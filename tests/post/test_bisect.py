# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import shutil
from pathlib import Path

import pytest

from plaid.containers.dataset import Dataset
from plaid.post.bisect import plot_bisect
from plaid.problem_definition import ProblemDefinition


@pytest.fixture()
def current_directory() -> Path:
    return Path(__file__).absolute().parent


@pytest.fixture()
def working_directory() -> Path:
    return Path.cwd()


class Test_Bisect:
    def test_bisect_with_paths(self, current_directory, working_directory):
        ref_path = current_directory / "dataset_ref"
        pred_path = current_directory / "dataset_pred"
        problem_path = current_directory / "problem_definition"
        plot_bisect(
            ref_path, pred_path, problem_path, "feature_2", "differ_bisect_plot"
        )
        shutil.move(
            working_directory / "differ_bisect_plot.png",
            current_directory / "differ_bisect_plot.png",
        )

    def test_bisect_with_objects(self, current_directory, working_directory):
        ref_path = Dataset(current_directory / "dataset_pred")
        pred_path = Dataset(current_directory / "dataset_pred")
        problem_path = ProblemDefinition(current_directory / "problem_definition")
        plot_bisect(ref_path, pred_path, problem_path, "feature_2", "equal_bisect_plot")
        shutil.move(
            working_directory / "equal_bisect_plot.png",
            current_directory / "equal_bisect_plot.png",
        )

    def test_bisect_with_mix(self, current_directory, working_directory):
        scalar_index = 0
        ref_path = current_directory / "dataset_ref"
        pred_path = current_directory / "dataset_near_pred"
        problem_path = ProblemDefinition(current_directory / "problem_definition")
        plot_bisect(
            ref_path,
            pred_path,
            problem_path,
            scalar_index,
            "converge_bisect_plot",
            verbose=True,
        )
        shutil.move(
            working_directory / "converge_bisect_plot.png",
            current_directory / "converge_bisect_plot.png",
        )

    def test_bisect_error(self, current_directory):
        ref_path = current_directory / "dataset_ref"
        pred_path = current_directory / "dataset_near_pred"
        problem_path = ProblemDefinition(current_directory / "problem_definition")
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
        path_1 = current_directory / "differ_bisect_plot.png"
        path_2 = current_directory / "equal_bisect_plot.png"
        path_3 = current_directory / "converge_bisect_plot.png"
        assert path_1.is_file()
        assert path_2.is_file()
        assert path_3.is_file()
