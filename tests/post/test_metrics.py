# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import shutil
from pathlib import Path

import pytest
import yaml

from plaid.containers.dataset import Dataset
from plaid.post.metrics import compute_metrics
from plaid.problem_definition import ProblemDefinition


@pytest.fixture()
def current_directory() -> Path:
    return Path(__file__).absolute().parent


@pytest.fixture()
def working_directory() -> Path:
    return Path.cwd()


class Test_Metrics:
    def test_compute_metrics_with_paths(self, current_directory, working_directory):
        ref_ds = current_directory / "dataset_ref"
        pred_ds = current_directory / "dataset_near_pred"
        problem = current_directory / "problem_definition"
        compute_metrics(ref_ds, pred_ds, problem, "first_metrics")
        shutil.move(
            working_directory / "first_metrics.yaml",
            current_directory / "first_metrics.yaml",
        )

    def test_compute_metrics_with_objects(self, current_directory, working_directory):
        ref_ds = Dataset(current_directory / "dataset_ref")
        pred_ds = Dataset(current_directory / "dataset_pred")
        problem = ProblemDefinition(current_directory / "problem_definition")
        compute_metrics(ref_ds, pred_ds, problem, "second_metrics", verbose=True)
        shutil.move(
            working_directory / "second_metrics.yaml",
            current_directory / "second_metrics.yaml",
        )

    def test_compute_metrics_mix(self, current_directory, working_directory):
        ref_ds = Dataset(current_directory / "dataset_ref")
        pred_ds = Dataset(current_directory / "dataset_ref")
        problem = ProblemDefinition(current_directory / "problem_definition")
        compute_metrics(ref_ds, pred_ds, problem, "third_metrics", verbose=True)
        shutil.move(
            working_directory / "third_metrics.yaml",
            current_directory / "third_metrics.yaml",
        )

    def test_compute_RMSE_data(self, current_directory):
        path = current_directory / "first_metrics.yaml"
        with path.open("r") as file:
            contenu_yaml = yaml.load(file, Loader=yaml.FullLoader)
        assert contenu_yaml["rRMSE for scalars"]["train"]["scalar_2"] < 0.2
        assert contenu_yaml["rRMSE for scalars"]["test"]["scalar_2"] < 0.2
        assert contenu_yaml["RMSE for scalars"]["train"]["scalar_2"] < 0.2
        assert contenu_yaml["RMSE for scalars"]["test"]["scalar_2"] < 0.2
        assert contenu_yaml["R2 for scalars"]["train"]["scalar_2"] > 0.8
        assert contenu_yaml["R2 for scalars"]["test"]["scalar_2"] > 0.8

    def test_compute_rRMSE_data(self, current_directory):
        path = current_directory / "second_metrics.yaml"
        with path.open("r") as file:
            contenu_yaml = yaml.load(file, Loader=yaml.FullLoader)
        assert contenu_yaml["rRMSE for scalars"]["train"]["scalar_2"] > 0.75
        assert contenu_yaml["rRMSE for scalars"]["test"]["scalar_2"] > 0.75
        assert contenu_yaml["RMSE for scalars"]["train"]["scalar_2"] > 0.75
        assert contenu_yaml["RMSE for scalars"]["test"]["scalar_2"] > 0.75
        assert contenu_yaml["R2 for scalars"]["train"]["scalar_2"] < 0.0
        assert contenu_yaml["R2 for scalars"]["test"]["scalar_2"] < 0.0

    def test_compute_R2_data(self, current_directory):
        path = current_directory / "third_metrics.yaml"
        with path.open("r") as file:
            contenu_yaml = yaml.load(file, Loader=yaml.FullLoader)
        assert contenu_yaml["rRMSE for scalars"]["train"]["scalar_2"] == 0.0
        assert contenu_yaml["rRMSE for scalars"]["test"]["scalar_2"] == 0.0
        assert contenu_yaml["RMSE for scalars"]["train"]["scalar_2"] == 0.0
        assert contenu_yaml["RMSE for scalars"]["test"]["scalar_2"] == 0.0
        assert contenu_yaml["R2 for scalars"]["train"]["scalar_2"] == 1.0
        assert contenu_yaml["R2 for scalars"]["test"]["scalar_2"] == 1.0
