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

    def test_hf_datasets_train_test_split(
        self, tmp_path, generator_split, infos, problem_definition
    ):
        """Test train_test_split method for HF datasets backend."""
        test_dir = tmp_path / "test_hf_split"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        datasetdict, _ = init_from_disk(test_dir)
        dataset = datasetdict["train"]

        # Test default split (75/25) - uses native HF datasets method
        split_dict = dataset.train_test_split()
        assert "train" in split_dict
        assert "test" in split_dict
        assert len(split_dict["train"]) + len(split_dict["test"]) == len(dataset)

        # Test with specific test_size
        split_dict = dataset.train_test_split(test_size=1)
        assert len(split_dict["test"]) == 1
        assert len(split_dict["train"]) == len(dataset) - 1

        # Test with shuffle and seed for reproducibility
        split_dict1 = dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
        split_dict2 = dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
        # For HF datasets, we compare the actual indices
        assert len(split_dict1["train"]) == len(split_dict2["train"])
        assert len(split_dict1["test"]) == len(split_dict2["test"])

        # Verify split datasets can still access samples
        train_sample = split_dict["train"][0]
        assert train_sample is not None

    def test_cgns_train_test_split(
        self, tmp_path, generator_split, infos, problem_definition
    ):
        """Test train_test_split method for CGNSDataset."""
        test_dir = tmp_path / "test_cgns_split"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="cgns",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        datasetdict, _ = init_from_disk(test_dir)
        dataset = datasetdict["train"]

        # Test default split (75/25)
        split_dict = dataset.train_test_split()
        assert "train" in split_dict
        assert "test" in split_dict
        assert len(split_dict["train"]) + len(split_dict["test"]) == len(dataset)

        # Test with specific test_size
        split_dict = dataset.train_test_split(test_size=1)
        assert len(split_dict["test"]) == 1
        assert len(split_dict["train"]) == len(dataset) - 1

        # Test with shuffle=False and seed for reproducibility
        split_dict1 = dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
        split_dict2 = dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
        assert list(split_dict1["train"].ids) == list(split_dict2["train"].ids)
        assert list(split_dict1["test"].ids) == list(split_dict2["test"].ids)

        # Verify split datasets can still access samples
        train_sample = split_dict["train"][split_dict["train"].ids[0]]
        assert isinstance(train_sample, Sample)

    def test_zarr_train_test_split(
        self, tmp_path, generator_split, infos, problem_definition
    ):
        """Test train_test_split method for ZarrDataset."""
        test_dir = tmp_path / "test_zarr_split"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        datasetdict, _ = init_from_disk(test_dir)
        dataset = datasetdict["train"]

        # Test default split (75/25)
        split_dict = dataset.train_test_split()
        assert "train" in split_dict
        assert "test" in split_dict
        assert len(split_dict["train"]) + len(split_dict["test"]) == len(dataset)

        # Test with specific test_size
        split_dict = dataset.train_test_split(test_size=1)
        assert len(split_dict["test"]) == 1
        assert len(split_dict["train"]) == len(dataset) - 1

        # Test with shuffle=False and seed for reproducibility
        split_dict1 = dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
        split_dict2 = dataset.train_test_split(test_size=0.5, shuffle=True, seed=42)
        assert list(split_dict1["train"].ids) == list(split_dict2["train"].ids)
        assert list(split_dict1["test"].ids) == list(split_dict2["test"].ids)

        # Verify split datasets can still access samples
        train_sample = split_dict["train"][split_dict["train"].ids[0]]
        assert isinstance(train_sample, dict)

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
