# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

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
from plaid.storage.common.preprocessor import preprocess
from plaid.storage.hf_datasets.bridge import (
    plaid_dataset_to_datasetdict,
    to_var_sample_dict,
)


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


# %% Fixtures
@pytest.fixture()
def dataset(samples, infos) -> Dataset:
    samples_ = []
    for i, sample in enumerate(samples):
        if i == 1:
            sample.add_scalar("toto", 1.0)
        samples_.append(sample)
        samples_.append(sample)
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
def generator_split(dataset, problem_definition) -> dict[str, Callable]:
    generators_ = {}

    main_splits = problem_definition.get_split()

    for split_name, ids in main_splits.items():

        def generator_():
            for id in ids:
                yield dataset[id]

        generators_[split_name] = generator_

    return generators_


@pytest.fixture()
def generator_split_with_kwargs(dataset, gen_kwargs) -> dict[str, Callable]:
    generators_ = {}

    for split_name in gen_kwargs.keys():

        def generator_(shards_ids):
            for ids in shards_ids:
                if isinstance(ids, int):
                    ids = [ids]
                for id in ids:
                    yield dataset[id]

        generators_[split_name] = generator_

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
        main_splits,
        tmp_path,
        generator_split,
        infos,
        problem_definition,
    ):
        _, variable_schema, _, _, _ = preprocess(generator_split)
        datasetdict = plaid_dataset_to_datasetdict(
            dataset, main_splits, variable_schema
        )

        test_dir = tmp_path / "test_hf"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
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

        with pytest.raises(ValueError):
            save_to_disk(
                output_folder=test_dir,
                generators=generator_split,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                overwrite=False,
            )

        load_problem_definitions_from_disk(test_dir)
        with pytest.raises(ValueError):
            load_problem_definitions_from_disk("dummy")

        datasetdict, converterdict = init_from_disk(test_dir)

        dataset = datasetdict["train"]
        converter = converterdict["train"]

        print(converter)

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

        # print(converter.flat_cst)
        converter.to_dict(
            dataset,
            0,
            features=[
                "TestBaseName/TestZoneName/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        converter.to_dict(
            dataset,
            0,
            features=["TestBaseName/TestZoneName/VertexFields/test_field_same_size"],
        )
        converter.to_dict(dataset, 0, features=["Global/global_0"])
        with pytest.raises(KeyError):
            converter.to_dict(dataset, 0, features=["dummy"])

    def test_zarr(self, tmp_path, generator_split, infos, problem_definition):
        test_dir = tmp_path / "test_hf"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir)

        dataset = datasetdict["train"]
        converter = converterdict["train"]

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
                "TestBaseName/TestZoneName/VertexFields/test_field_same_size",
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
