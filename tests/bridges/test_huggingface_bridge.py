# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import pickle
from typing import Callable

import pytest

from plaid.bridges import huggingface_bridge
from plaid.bridges.huggingface_bridge import to_plaid_sample
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition


# %% Fixtures
@pytest.fixture()
def dataset(samples, infos) -> Dataset:
    dataset = Dataset()
    dataset.add_samples(samples[:2])
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
    def generator():
        for id in range(len(dataset)):
            yield {
                "sample": pickle.dumps(dataset[id]),
            }

    return generator


@pytest.fixture()
def hf_dataset(generator) -> Dataset:
    hf_dataset = huggingface_bridge.plaid_generator_to_huggingface(generator)
    return hf_dataset


class Test_Huggingface_Bridge:
    def assert_hf_dataset(self, hfds):
        # assert hfds.description["legal"] == {"owner": "PLAID2", "license": "BSD-3"}
        # assert hfds.description["task"] == "regression"
        # assert hfds.description["in_scalars_names"][0] == "feature_name_1"
        # assert hfds.description["in_scalars_names"][1] == "feature_name_2"
        self.assert_sample(to_plaid_sample(hfds[0]))

    def assert_plaid_dataset(self, ds):
        # assert ds.get_infos()["legal"] == {"owner": "PLAID2", "license": "BSD-3"}
        # assert pbdef.get_input_scalars_names()[0] == "feature_name_1"
        # assert pbdef.get_input_scalars_names()[1] == "feature_name_2"
        self.assert_sample(ds[0])

    def assert_sample(self, sample):
        assert isinstance(sample, Sample)
        assert sample.get_scalar_names()[0] == "test_scalar"
        assert "test_field_same_size" in sample.get_field_names()
        assert sample.get_field("test_field_same_size").shape[0] == 17

    def test_to_plaid_sample(self, generator):
        hfds = huggingface_bridge.plaid_generator_to_huggingface(generator)
        to_plaid_sample(hfds[0])

    def test_to_plaid_sample_fallback_build_succeeds(self, dataset):
        sample = dataset[0]
        old_hf_sample = {
            "path": getattr(sample, "path", None),
            "scalars": {sn: sample.get_scalar(sn) for sn in sample.get_scalar_names()},
            "meshes": sample.features.data,
        }
        old_hf_sample = {"sample": pickle.dumps(old_hf_sample)}
        plaid_sample = to_plaid_sample(old_hf_sample)
        assert isinstance(plaid_sample, Sample)

    def test_plaid_dataset_to_huggingface(self, dataset, problem_definition):
        hfds = huggingface_bridge.plaid_dataset_to_huggingface(
            dataset, problem_definition, split="train"
        )
        hfds = huggingface_bridge.plaid_dataset_to_huggingface(
            dataset, split="all_samples"
        )
        hfds = huggingface_bridge.plaid_dataset_to_huggingface(
            dataset, problem_definition
        )
        self.assert_hf_dataset(hfds)

    def test_plaid_dataset_to_huggingface_datasetdict(
        self, dataset, problem_definition
    ):
        huggingface_bridge.plaid_dataset_to_huggingface_datasetdict(
            dataset, problem_definition, main_splits=["train", "test"]
        )

    def test_plaid_generator_to_huggingface(self, generator):
        hfds = huggingface_bridge.plaid_generator_to_huggingface(
            generator, split="train"
        )
        hfds = huggingface_bridge.plaid_generator_to_huggingface(generator)
        self.assert_hf_dataset(hfds)

    def test_plaid_generator_to_huggingface_datasetdict(self, generator):
        huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
            generator, main_splits=["train", "test"]
        )

    def test_huggingface_dataset_to_plaid(self, hf_dataset):
        ds = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset)
        self.assert_plaid_dataset(ds)

    def test_huggingface_dataset_to_plaid_with_ids(self, hf_dataset):
        huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset, ids=[0, 1])

    def test_huggingface_dataset_to_plaid_large(self, hf_dataset):
        huggingface_bridge.huggingface_dataset_to_plaid(
            hf_dataset, processes_number=2, large_dataset=True
        )

    def test_huggingface_dataset_to_plaid_with_ids_large(self, hf_dataset):
        with pytest.raises(NotImplementedError):
            huggingface_bridge.huggingface_dataset_to_plaid(
                hf_dataset, ids=[0, 1], processes_number=2, large_dataset=True
            )

    def test_huggingface_dataset_to_plaid_error_processes_number(self, hf_dataset):
        with pytest.raises(AssertionError):
            huggingface_bridge.huggingface_dataset_to_plaid(
                hf_dataset, processes_number=128
            )

    def test_huggingface_dataset_to_plaid_error_processes_number_2(self, hf_dataset):
        with pytest.raises(AssertionError):
            huggingface_bridge.huggingface_dataset_to_plaid(
                hf_dataset, ids=[0], processes_number=2
            )

    # ----------------------------------------------------------------
    # deprecated function

    def test_huggingface_description_to_problem_definition(self, hf_dataset):
        huggingface_bridge.huggingface_description_to_problem_definition(
            hf_dataset.description
        )

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
