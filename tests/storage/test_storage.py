# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable

import pytest

# from plaid.bridges import huggingface_bridge
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from plaid.storage import (
    init_from_disk,
    load_problem_definitions_from_disk,
    save_to_disk,
)
from plaid.storage.hf_datasets.bridge import (
    to_var_sample_dict,
)


def _yield_samples_from_ids(dataset: Dataset, ids: list[int]):
    for id_ in ids:
        yield dataset[id_]


def _yield_samples_from_local_ids(
    dataset: Dataset, split_global_ids: list[int], ids: list[int]
):
    for id_ in ids:
        yield dataset[split_global_ids[id_]]


def _yield_samples_from_shards_ids(dataset: Dataset, shards_ids):
    for ids in shards_ids:
        if isinstance(ids, int):
            ids = [ids]
        for id_ in ids:
            yield dataset[id_]


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


# %% Fixtures
@pytest.fixture()
def dataset(samples, infos) -> Dataset:
    samples_ = []
    for i, sample in enumerate(samples):
        sample_ = deepcopy(sample)
        if i == 0 or i == 2:
            sample_.add_scalar("toto", 1.0)
        samples_.append(sample_)

    dataset = Dataset(samples=samples_)
    dataset.set_infos(infos)
    return dataset


@pytest.fixture()
def main_splits() -> dict:
    return {"train": [0, 2], "test": [1, 3]}


@pytest.fixture()
def problem_definition(main_splits) -> ProblemDefinition:
    problem_definition = ProblemDefinition()
    problem_definition.set_task("regression")
    problem_definition.add_input_scalars_names(["feature_name_1", "feature_name_2"])
    problem_definition.set_split(main_splits)
    return problem_definition


@pytest.fixture()
def generator(dataset) -> Callable:
    def generator_():
        for sample in dataset:
            yield sample

    return generator_


@pytest.fixture()
def gen_kwargs(problem_definition) -> dict[str, dict]:
    gen_kwargs = {}
    for split_name, ids in problem_definition.get_split().items():
        mid = len(ids) // 2
        gen_kwargs[split_name] = {"shards_ids": [ids[:mid], ids[mid:]]}
    return gen_kwargs


@pytest.fixture()
def split_global_ids(problem_definition) -> dict[str, list[int]]:
    return problem_definition.get_split()


@pytest.fixture()
def split_n_samples(problem_definition) -> dict[str, int]:
    return {k: len(v) for k, v in problem_definition.get_split().items()}


@pytest.fixture()
def generator_split(dataset, problem_definition) -> dict[str, Callable]:
    generators_ = {}

    main_splits = problem_definition.get_split()

    for split_name, ids in main_splits.items():

        def generator_(ids):
            for id in ids:
                yield dataset[id]

        generators_[split_name] = partial(generator_, ids)

    return generators_


@pytest.fixture()
def generator_split_with_kwargs(dataset, gen_kwargs) -> dict[str, Callable]:
    generators_ = {}

    for split_name in gen_kwargs.keys():
        generators_[split_name] = partial(_yield_samples_from_shards_ids, dataset)

    return generators_


@pytest.fixture()
def generator_split_from_local_ids(dataset, split_global_ids) -> dict[str, Callable]:
    generators_ = {}

    for split_name, global_ids in split_global_ids.items():
        generators_[split_name] = partial(
            _yield_samples_from_local_ids, dataset, global_ids
        )

    return generators_


class Test_Storage:
    def assert_sample(self, sample):
        assert isinstance(sample, Sample)
        sorted_names = sorted(sample.get_scalar_names())
        for i in range(4):
            assert sorted_names[i] == f"global_{i}"
        assert "test_field_same_size" in sample.get_field_names()
        assert sample.get_field("test_field_same_size").shape[0] == 17

    # ------------------------------------------------------------------------------
    #     HUGGING FACE BRIDGE (with tree flattening and pyarrow tables)
    # ------------------------------------------------------------------------------

    def test_hf_datasets(
        self,
        dataset,
        tmp_path,
        generator_split,
        generator_split_with_kwargs,
        infos,
        problem_definition,
        gen_kwargs,
    ):
        test_dir = tmp_path / "test_hf"
        legacy_kwargs_test_dir = tmp_path / "test_hf_legacy_kwargs"

        save_to_disk(
            output_folder=legacy_kwargs_test_dir,
            generators=generator_split_with_kwargs,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            gen_kwargs=gen_kwargs,
            num_proc=2,
            overwrite=True,
            verbose=True,
        )

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
        )

        with pytest.raises(ValueError):
            save_to_disk(
                output_folder=test_dir,
                generators=generator_split,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                overwrite=False,
            )

        with pytest.raises(ValueError):
            problem_definition.set_name(None)
            save_to_disk(
                output_folder=test_dir,
                generators=generator_split,
                backend="hf_datasets",
                infos=infos,
                pb_defs=problem_definition,
                overwrite=True,
            )

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        load_problem_definitions_from_disk(test_dir)
        with pytest.raises(ValueError):
            load_problem_definitions_from_disk("dummy")

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        dataset = datasetdict["train"]
        converter = converterdict["train"]

        print(converter)

        plaid_sample = converter.to_plaid(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        plaid_sample = converter.to_plaid(dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        to_var_sample_dict(dataset, 0, enforce_shapes=False)

        for t in plaid_sample.get_all_time_values():
            for path in problem_definition.get_in_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)
            for path in problem_definition.get_out_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)

        converter.to_dict(dataset, 0)
        converter.sample_to_dict(dataset[0])

        converter.to_dict(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        converter.to_dict(
            dataset,
            0,
            features=["Base_Name/Zone_Name/VertexFields/test_field_same_size"],
        )
        converter.to_dict(dataset, 0, features=["Global/global_0"])
        with pytest.raises(KeyError):
            converter.to_dict(dataset, 0, features=["dummy"])

    def test_zarr(self, tmp_path, generator_split, infos, problem_definition):
        test_dir = tmp_path / "test_zarr"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        dataset = datasetdict["train"]
        converter = converterdict["train"]

        plaid_sample = converter.to_plaid(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        plaid_sample = converter.to_plaid(dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        # coverage of ZarrDataset classe
        for sample in dataset:
            sample
        len(dataset)
        dataset.zarr_group
        dataset.toto = 1.0
        print(dataset)

        for t in plaid_sample.get_all_time_values():
            for path in problem_definition.get_in_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)
            for path in problem_definition.get_out_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)

        converter.to_dict(dataset, 0)
        converter.sample_to_dict(dataset[0])

        converter.to_dict(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        with pytest.raises(KeyError):
            converter.to_dict(dataset, 0, features=["dummy"])

    def test_cgns(self, tmp_path, generator_split, infos, problem_definition):
        test_dir = tmp_path / "test_cgns"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="cgns",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        dataset = datasetdict["train"]
        converter = converterdict["train"]

        # coverage of CGNSDataset classe
        for sample in dataset:
            sample
        len(dataset)
        dataset.ids
        dataset.toto = 1.0
        print(dataset)

        plaid_sample = converter.to_plaid(dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        for t in plaid_sample.get_all_time_values():
            for path in problem_definition.get_in_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)
            for path in problem_definition.get_out_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)

        with pytest.raises(ValueError):
            converter.to_dict(dataset, 0)
        with pytest.raises(ValueError):
            converter.sample_to_dict(dataset[0])

    def test_registry(self):
        from plaid.storage import registry

        backends = registry.available_backends()
        assert "hf_datasets" in backends
        assert "zarr" in backends
        assert "cgns" in backends

        hf_module = registry.get_backend("hf_datasets")
        assert hf_module is not None

        zarr_module = registry.get_backend("zarr")
        assert zarr_module is not None

        cgns_module = registry.get_backend("cgns")
        assert cgns_module is not None

        with pytest.raises(ValueError):
            _ = registry.get_backend("non_existent_backend")

    def test_hf_datasets_with_split_n_samples_parallel(
        self,
        tmp_path,
        infos,
        problem_definition,
        split_n_samples,
        gen_kwargs,
        generator_split_from_local_ids,
    ):
        test_dir = tmp_path / "test_hf_split_n_samples"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split_from_local_ids,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            split_n_samples=split_n_samples,
            num_proc=2,
            overwrite=True,
            verbose=True,
        )

        datasetdict, _ = init_from_disk(test_dir)
        assert len(datasetdict["train"]) == split_n_samples["train"]
        assert len(datasetdict["test"]) == split_n_samples["test"]

        with pytest.raises(ValueError):
            save_to_disk(
                output_folder=test_dir,
                generators=generator_split_from_local_ids,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                split_n_samples=split_n_samples,
                gen_kwargs=gen_kwargs,
                num_proc=2,
                overwrite=True,
            )
