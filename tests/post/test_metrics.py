# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import os
import shutil

import pytest
import yaml

from plaid.containers.dataset import Dataset
from plaid.post.metrics import compute_metrics
from plaid.problem_definition import ProblemDefinition


@pytest.fixture()
def current_directory() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def working_directory() -> str:
    return os.getcwd()


class Test_Metrics:
    def test_compute_metrics_with_paths(self, current_directory, working_directory):
        ref_ds = os.path.join(current_directory, "dataset_ref")
        pred_ds = os.path.join(current_directory, "dataset_near_pred")
        problem = os.path.join(current_directory, "problem_definition")
        compute_metrics(ref_ds, pred_ds, problem, "first_metrics")
        shutil.move(
            os.path.join(working_directory, "first_metrics.yaml"),
            os.path.join(current_directory, "first_metrics.yaml"),
        )

    def test_compute_metrics_with_objects(self, current_directory, working_directory):
        ref_ds = Dataset(os.path.join(current_directory, "dataset_ref"))
        pred_ds = Dataset(os.path.join(current_directory, "dataset_pred"))
        problem = ProblemDefinition(
            os.path.join(current_directory, "problem_definition")
        )
        compute_metrics(ref_ds, pred_ds, problem, "second_metrics", verbose=True)
        shutil.move(
            os.path.join(working_directory, "second_metrics.yaml"),
            os.path.join(current_directory, "second_metrics.yaml"),
        )

    def test_compute_metrics_mix(self, current_directory, working_directory):
        ref_ds = Dataset(os.path.join(current_directory, "dataset_ref"))
        pred_ds = Dataset(os.path.join(current_directory, "dataset_ref"))
        problem = ProblemDefinition(
            os.path.join(current_directory, "problem_definition")
        )
        compute_metrics(ref_ds, pred_ds, problem, "third_metrics", verbose=True)
        shutil.move(
            os.path.join(working_directory, "third_metrics.yaml"),
            os.path.join(current_directory, "third_metrics.yaml"),
        )

    def test_compute_RMSE_data(self, current_directory):
        path = os.path.join(current_directory, "first_metrics.yaml")
        with open(path, "r") as file:
            contenu_yaml = yaml.load(file, Loader=yaml.FullLoader)
        assert contenu_yaml["rRMSE for scalars"]["train"]["scalar_2"] < 0.2
        assert contenu_yaml["rRMSE for scalars"]["test"]["scalar_2"] < 0.2
        assert contenu_yaml["RMSE for scalars"]["train"]["scalar_2"] < 0.2
        assert contenu_yaml["RMSE for scalars"]["test"]["scalar_2"] < 0.2
        assert contenu_yaml["R2 for scalars"]["train"]["scalar_2"] > 0.8
        assert contenu_yaml["R2 for scalars"]["test"]["scalar_2"] > 0.8

    def test_compute_rRMSE_data(self, current_directory):
        path = os.path.join(current_directory, "second_metrics.yaml")
        with open(path, "r") as file:
            contenu_yaml = yaml.load(file, Loader=yaml.FullLoader)
        assert contenu_yaml["rRMSE for scalars"]["train"]["scalar_2"] > 0.75
        assert contenu_yaml["rRMSE for scalars"]["test"]["scalar_2"] > 0.75
        assert contenu_yaml["RMSE for scalars"]["train"]["scalar_2"] > 0.75
        assert contenu_yaml["RMSE for scalars"]["test"]["scalar_2"] > 0.75
        assert contenu_yaml["R2 for scalars"]["train"]["scalar_2"] < 0.0
        assert contenu_yaml["R2 for scalars"]["test"]["scalar_2"] < 0.0

    def test_compute_R2_data(self, current_directory):
        path = os.path.join(current_directory, "third_metrics.yaml")
        with open(path, "r") as file:
            contenu_yaml = yaml.load(file, Loader=yaml.FullLoader)
        assert contenu_yaml["rRMSE for scalars"]["train"]["scalar_2"] == 0.0
        assert contenu_yaml["rRMSE for scalars"]["test"]["scalar_2"] == 0.0
        assert contenu_yaml["RMSE for scalars"]["train"]["scalar_2"] == 0.0
        assert contenu_yaml["RMSE for scalars"]["test"]["scalar_2"] == 0.0
        assert contenu_yaml["R2 for scalars"]["train"]["scalar_2"] == 1.0
        assert contenu_yaml["R2 for scalars"]["test"]["scalar_2"] == 1.0
