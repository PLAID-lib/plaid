# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import os
import shutil

import pytest

from plaid.containers.dataset import Dataset
from plaid.post.bisect import plot_bisect
from plaid.problem_definition import ProblemDefinition


@pytest.fixture()
def current_directory() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def working_directory() -> str:
    return os.getcwd()


class Test_Bisect:
    def test_bisect_with_paths(self, current_directory, working_directory):
        ref_path = os.path.join(current_directory, "dataset_ref")
        pred_path = os.path.join(current_directory, "dataset_pred")
        problem_path = os.path.join(current_directory, "problem_definition")
        plot_bisect(ref_path, pred_path, problem_path, "scalar_2", "differ_bisect_plot")
        shutil.move(
            os.path.join(working_directory, "differ_bisect_plot.png"),
            os.path.join(current_directory, "differ_bisect_plot.png"),
        )

    def test_bisect_with_objects(self, current_directory, working_directory):
        ref_path = Dataset(os.path.join(current_directory, "dataset_pred"))
        pred_path = Dataset(os.path.join(current_directory, "dataset_pred"))
        problem_path = ProblemDefinition(
            os.path.join(current_directory, "problem_definition")
        )
        plot_bisect(ref_path, pred_path, problem_path, "scalar_2", "equal_bisect_plot")
        shutil.move(
            os.path.join(working_directory, "equal_bisect_plot.png"),
            os.path.join(current_directory, "equal_bisect_plot.png"),
        )

    def test_bisect_with_mix(self, current_directory, working_directory):
        scalar_index = 0
        ref_path = os.path.join(current_directory, "dataset_ref")
        pred_path = os.path.join(current_directory, "dataset_near_pred")
        problem_path = ProblemDefinition(
            os.path.join(current_directory, "problem_definition")
        )
        plot_bisect(
            ref_path,
            pred_path,
            problem_path,
            scalar_index,
            "converge_bisect_plot",
            verbose=True,
        )
        shutil.move(
            os.path.join(working_directory, "converge_bisect_plot.png"),
            os.path.join(current_directory, "converge_bisect_plot.png"),
        )

    def test_bisect_error(self, current_directory):
        ref_path = os.path.join(current_directory, "dataset_ref")
        pred_path = os.path.join(current_directory, "dataset_near_pred")
        problem_path = ProblemDefinition(
            os.path.join(current_directory, "problem_definition")
        )
        with pytest.raises(KeyError):
            plot_bisect(
                ref_path,
                pred_path,
                problem_path,
                "unknown_scalar",
                "converge_bisect_plot",
                verbose=True,
            )

    def test_generated_files(self, current_directory):
        path_1 = os.path.join(current_directory, "differ_bisect_plot.png")
        path_2 = os.path.join(current_directory, "equal_bisect_plot.png")
        path_3 = os.path.join(current_directory, "converge_bisect_plot.png")
        assert os.path.exists(path_1)
        assert os.path.exists(path_2)
        assert os.path.exists(path_3)
