# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import pickle
import shutil
from pathlib import Path
from typing import Callable

import pytest

from plaid.bridges import huggingface_bridge
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from plaid.utils import cgns_helper


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


# %% Fixtures
@pytest.fixture()
def dataset(samples, infos) -> Dataset:
    dataset = Dataset(samples=samples)
    dataset.set_infos(infos)
    return dataset


@pytest.fixture()
def problem_definition() -> ProblemDefinition:
    problem_definition = ProblemDefinition()
    problem_definition.set_task("regression")
    problem_definition.add_input_scalars_names(["feature_name_1", "feature_name_2"])
    problem_definition.set_split({"train": [0], "test": [1]})
    return problem_definition


@pytest.fixture()
def generator(dataset) -> Callable:
    def generator_():
        for sample in dataset:
            yield sample

    return generator_


@pytest.fixture()
def generator_split(dataset, problem_definition) -> dict[str, Callable]:
    generators_ = {}
    for split_name, ids in problem_definition.get_split().items():

        def generator_(ids=ids):
            for id in ids:
                yield dataset[id]

        generators_[split_name] = generator_
    return generators_


@pytest.fixture()
def generator_binary(dataset) -> Callable:
    def generator_():
        for sample in dataset:
            yield {
                "sample": pickle.dumps(sample),
            }

    return generator_


@pytest.fixture()
def generator_split_binary(dataset, problem_definition) -> dict[str, Callable]:
    generators_ = {}
    for split_name, ids in problem_definition.get_split().items():

        def generator_(ids=ids):
            for id in ids:
                yield {"sample": pickle.dumps(dataset[id])}

        generators_[split_name] = generator_
    return generators_


@pytest.fixture()
def hf_dataset(generator_binary) -> Dataset:
    hf_dataset = huggingface_bridge.plaid_generator_to_huggingface_binary(
        generator_binary
    )
    return hf_dataset


class Test_Huggingface_Bridge:
    def assert_sample(self, sample):
        assert isinstance(sample, Sample)
        assert sample.get_scalar_names()[0] == "test_scalar"
        assert "test_field_same_size" in sample.get_field_names()
        assert sample.get_field("test_field_same_size").shape[0] == 17

    def assert_hf_dataset_binary(self, hfds_binary):
        self.assert_sample(huggingface_bridge.to_plaid_sample_binary(hfds_binary[0]))

    def assert_plaid_dataset(self, ds):
        self.assert_sample(ds[0])

    # ------------------------------------------------------------------------------
    #     HUGGING FACE BRIDGE (with tree flattening and pyarrow tables)
    # ------------------------------------------------------------------------------

    def test_with_datasetdict(self, dataset, problem_definition):
        main_splits = problem_definition.get_split()
        hf_dataset_dict, flat_cst, key_mappings = (
            huggingface_bridge.plaid_dataset_to_huggingface_datasetdict(
                dataset, main_splits
            )
        )
        huggingface_bridge.to_plaid_sample(
            hf_dataset_dict["train"], 0, flat_cst, key_mappings["cgns_types"]
        )
        huggingface_bridge.to_plaid_sample(
            hf_dataset_dict["test"],
            0,
            flat_cst,
            key_mappings["cgns_types"],
            enforce_shapes=True,
        )
        huggingface_bridge.huggingface_dataset_to_plaid(
            hf_dataset_dict["train"], flat_cst, key_mappings["cgns_types"]
        )
        huggingface_bridge.huggingface_dataset_to_plaid(
            hf_dataset_dict["test"],
            flat_cst,
            key_mappings["cgns_types"],
            enforce_shapes=True,
        )
        cgns_helper.compare_cgns_trees(dataset[0].get_mesh(), dataset[0].get_mesh())
        cgns_helper.compare_cgns_trees_no_types(
            dataset[0].get_mesh(), dataset[0].get_mesh()
        )

    def test_with_generator(self, generator_split):
        hf_dataset_dict, flat_cst, key_mappings = (
            huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
                generator_split
            )
        )
        huggingface_bridge.to_plaid_sample(
            hf_dataset_dict["train"], 0, flat_cst, key_mappings["cgns_types"]
        )
        huggingface_bridge.to_plaid_sample(
            hf_dataset_dict["test"],
            0,
            flat_cst,
            key_mappings["cgns_types"],
            enforce_shapes=True,
        )

    # ------------------------------------------------------------------------------
    #     HUGGING FACE INTERACTIONS ON DISK
    # ------------------------------------------------------------------------------

    def test_save_load_to_disk(
        self, current_directory, generator_split, infos, problem_definition
    ):
        hf_dataset_dict, flat_cst, key_mappings = (
            huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
                generator_split
            )
        )

        test_dir = Path(current_directory) / Path("test")
        huggingface_bridge.save_dataset_dict_to_disk(test_dir, hf_dataset_dict)
        huggingface_bridge.save_infos_to_disk(test_dir, infos)
        huggingface_bridge.save_problem_definition_to_disk(
            test_dir, "task_1", problem_definition
        )
        huggingface_bridge.save_tree_struct_to_disk(test_dir, flat_cst, key_mappings)

        huggingface_bridge.load_dataset_from_disk(test_dir)
        huggingface_bridge.load_infos_from_disk(test_dir)
        huggingface_bridge.load_problem_definition_from_disk(test_dir, "task_1")
        huggingface_bridge.load_tree_struct_from_disk(test_dir)
        shutil.rmtree(test_dir)

    # ------------------------------------------------------------------------------
    #     OLD HUGGING FACE BRIDGE (binary blobs)
    # ------------------------------------------------------------------------------

    def test_save_load_to_disk_binary(
        self, current_directory, generator_split_binary, infos, problem_definition
    ):
        hf_dataset_dict = (
            huggingface_bridge.plaid_generator_to_huggingface_datasetdict_binary(
                generator_split_binary
            )
        )
        test_dir = Path(current_directory) / Path("test")
        huggingface_bridge.save_dataset_dict_to_disk(test_dir, hf_dataset_dict)
        huggingface_bridge.save_infos_to_disk(test_dir, infos)
        huggingface_bridge.save_problem_definition_to_disk(
            test_dir, "task_1", problem_definition
        )
        huggingface_bridge.load_dataset_from_disk(test_dir)
        huggingface_bridge.load_infos_from_disk(test_dir)
        huggingface_bridge.load_problem_definition_from_disk(test_dir, "task_1")
        shutil.rmtree(test_dir)

    def test_to_plaid_sample_binary(self, generator_binary):
        hfds = huggingface_bridge.plaid_generator_to_huggingface_binary(
            generator_binary
        )
        huggingface_bridge.to_plaid_sample_binary(hfds[0])

    def test_to_plaid_sample_binary_fallback_build_succeeds(self, dataset):
        sample = dataset[0]
        old_hf_sample = {
            "path": getattr(sample, "path", None),
            "scalars": {sn: sample.get_scalar(sn) for sn in sample.get_scalar_names()},
            "meshes": sample.features.data,
        }
        old_hf_sample = {"sample": pickle.dumps(old_hf_sample)}
        plaid_sample = huggingface_bridge.to_plaid_sample_binary(old_hf_sample)
        assert isinstance(plaid_sample, Sample)

    def test_plaid_dataset_to_huggingface_binary(self, dataset):
        hfds = huggingface_bridge.plaid_dataset_to_huggingface_binary(dataset)
        hfds = huggingface_bridge.plaid_dataset_to_huggingface_binary(
            dataset, ids=[0, 1]
        )
        self.assert_hf_dataset_binary(hfds)

    def test_plaid_dataset_to_huggingface_datasetdict_binary(
        self, dataset, problem_definition
    ):
        huggingface_bridge.plaid_dataset_to_huggingface_datasetdict_binary(
            dataset, main_splits=problem_definition.get_split()
        )

    def test_plaid_generator_to_huggingface_binary(self, generator_binary):
        hfds = huggingface_bridge.plaid_generator_to_huggingface_binary(
            generator_binary
        )
        hfds = huggingface_bridge.plaid_generator_to_huggingface_binary(
            generator_binary, processes_number=2
        )
        self.assert_hf_dataset_binary(hfds)

    def test_plaid_generator_to_huggingface_datasetdict_binary(
        self, generator_split_binary
    ):
        huggingface_bridge.plaid_generator_to_huggingface_datasetdict_binary(
            generator_split_binary
        )

    def test_huggingface_dataset_to_plaid_binary(self, hf_dataset):
        ds = huggingface_bridge.huggingface_dataset_to_plaid_binary(hf_dataset)
        self.assert_plaid_dataset(ds)

    def test_huggingface_dataset_to_plaid_with_ids_binary(self, hf_dataset):
        huggingface_bridge.huggingface_dataset_to_plaid_binary(hf_dataset, ids=[0, 1])

    def test_huggingface_dataset_to_plaid_large_binary(self, hf_dataset):
        huggingface_bridge.huggingface_dataset_to_plaid_binary(
            hf_dataset, processes_number=2, large_dataset=True
        )

    def test_huggingface_dataset_to_plaid_large_binary_2(self, hf_dataset):
        huggingface_bridge.huggingface_dataset_to_plaid_binary(
            hf_dataset, processes_number=2
        )

    def test_huggingface_dataset_to_plaid_with_ids_large_binary(self, hf_dataset):
        with pytest.raises(NotImplementedError):
            huggingface_bridge.huggingface_dataset_to_plaid_binary(
                hf_dataset, ids=[0, 1], processes_number=2, large_dataset=True
            )

    def test_huggingface_dataset_to_plaid_error_processes_number_binary(
        self, hf_dataset
    ):
        with pytest.raises(AssertionError):
            huggingface_bridge.huggingface_dataset_to_plaid_binary(
                hf_dataset, processes_number=128
            )

    def test_huggingface_dataset_to_plaid_error_processes_number_binary_2(
        self, hf_dataset
    ):
        with pytest.raises(AssertionError):
            huggingface_bridge.huggingface_dataset_to_plaid_binary(
                hf_dataset, ids=[0], processes_number=2
            )

    # ------------------------------------------------------------------------------
    #     DEPRECATED FUNCTIONS
    # ------------------------------------------------------------------------------

    def test_huggingface_description_to_problem_definition(self, hf_dataset):
        huggingface_bridge.huggingface_description_to_problem_definition(
            hf_dataset.description
        )

    def test_huggingface_description_to_infos(self, infos):
        hf_description = {}
        hf_description.update(infos)
        huggingface_bridge.huggingface_description_to_infos(hf_description)

    def test_create_string_for_huggingface_dataset_card(self, hf_dataset):
        huggingface_bridge.create_string_for_huggingface_dataset_card(
            description=hf_dataset.description,
            download_size_bytes=10,
            dataset_size_bytes=10,
            nb_samples=10,
            owner="Safran",
            license="cc-by-sa-4.0",
            zenodo_url="https://zenodo.org/records/10124594",
            arxiv_paper_url="https://arxiv.org/pdf/2305.12871",
            pretty_name="2D quasistatic non-linear structural mechanics solutions",
            size_categories=["n<1K"],
            task_categories=["graph-ml"],
            tags=["physics learning", "geometry learning"],
            dataset_long_description="my long description",
            url_illustration="url3",
        )
