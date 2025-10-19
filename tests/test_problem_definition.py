# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import os
import subprocess
from pathlib import Path

import pytest
from packaging.version import Version

import plaid
from plaid.problem_definition import ProblemDefinition
from plaid.types.feature_types import FeatureIdentifier

# %% Fixtures


@pytest.fixture()
def problem_definition() -> ProblemDefinition:
    return ProblemDefinition()


@pytest.fixture()
def problem_definition_full(problem_definition: ProblemDefinition) -> ProblemDefinition:
    problem_definition.set_task("regression")

    # ----
    feature_identifier = FeatureIdentifier({"type": "scalar", "name": "feature"})
    predict_feature_identifier = FeatureIdentifier(
        {"type": "scalar", "name": "predict_feature"}
    )
    test_feature_identifier = FeatureIdentifier(
        {"type": "scalar", "name": "test_feature"}
    )
    problem_definition.add_in_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_in_feature_identifier(feature_identifier)
    problem_definition.add_out_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_out_feature_identifier(feature_identifier)
    # ----
    feature_identifier = "Base_2_2/Zone/PointData/U1"
    predict_feature_identifier = "Base_2_2/Zone/PointData/U2"
    test_feature_identifier = "Base_2_2/Zone/PointData/sig12"
    problem_definition.add_in_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_in_feature_identifier(feature_identifier)
    problem_definition.add_out_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_out_feature_identifier(feature_identifier)

    # ----
    problem_definition.add_input_scalars_names(["scalar", "test_scalar"])
    problem_definition.add_input_scalar_name("predict_scalar")
    problem_definition.add_output_scalars_names(["scalar", "test_scalar"])
    problem_definition.add_output_scalar_name("predict_scalar")

    problem_definition.add_input_fields_names(["field", "test_field"])
    problem_definition.add_input_field_name("predict_field")
    problem_definition.add_output_fields_names(["field", "test_field"])
    problem_definition.add_output_field_name("predict_field")

    problem_definition.add_input_timeseries_names(["timeseries", "test_timeseries"])
    problem_definition.add_input_timeseries_name("predict_timeseries")
    problem_definition.add_output_timeseries_names(["timeseries", "test_timeseries"])
    problem_definition.add_output_timeseries_name("predict_timeseries")

    problem_definition.add_input_meshes_names(["mesh", "test_mesh"])
    problem_definition.add_input_mesh_name("predict_mesh")
    problem_definition.add_output_meshes_names(["mesh", "test_mesh"])
    problem_definition.add_output_mesh_name("predict_mesh")

    new_split = {"train": [0, 1, 2], "test": [3, 4]}
    problem_definition.set_split(new_split)
    return problem_definition


@pytest.fixture()
def current_directory() -> Path:
    return Path(__file__).absolute().parent


@pytest.fixture(scope="session", autouse=True)
def clean_tests():
    base_dir = Path(__file__).absolute().parent
    if os.name == "nt":
        # Windows
        script_path = base_dir / "clean.bat"
        retcode = subprocess.call(["cmd", "/c", str(script_path)])
    else:
        # Unix
        script_path = base_dir / "clean.sh"
        retcode = subprocess.call(["sh", str(script_path)])
    assert retcode == 0, "Test cleanup script failed"


# %% Tests


class Test_ProblemDefinition:
    def test__init__(self, problem_definition):
        assert problem_definition.get_task() is None
        print(problem_definition)

    def test__init__path(self, current_directory):
        d_path = current_directory / "problem_definition"
        ProblemDefinition(path=d_path)

    def test__init__directory_path(self, current_directory):
        d_path = current_directory / "problem_definition"
        ProblemDefinition(directory_path=d_path)

    def test__init__both_path_and_directory_path(self, current_directory):
        d_path = current_directory / "problem_definition"
        with pytest.raises(ValueError):
            ProblemDefinition(path=d_path, directory_path=d_path)

    # -------------------------------------------------------------------------#
    def test_version(self, problem_definition):
        # Unauthorized version
        assert problem_definition.get_version() == Version(plaid.__version__)

    # -------------------------------------------------------------------------#
    def test_task(self, problem_definition):
        # Unauthorized task
        with pytest.raises(TypeError):
            problem_definition.set_task("ighyurgv")
        problem_definition.set_task("classification")
        with pytest.raises(ValueError):
            problem_definition.set_task("regression")
        assert problem_definition.get_task() == "classification"
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_score_function(self, problem_definition):
        # Unauthorized task
        with pytest.raises(TypeError):
            problem_definition.set_score_function("ighyurgv")
        problem_definition.set_score_function("RRMSE")
        with pytest.raises(ValueError):
            problem_definition.set_score_function("RRMSE")
        assert problem_definition.get_score_function() == "RRMSE"
        print(problem_definition)

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # -------------------------------------------------------------------------#
    def test_get_in_features_identifiers(self, problem_definition):
        assert problem_definition.get_in_features_identifiers() == []

    def test_add_in_features_identifiers_fail_same_identifier(self, problem_definition):
        dummy_identifier = FeatureIdentifier({"type": "scalar", "name": "dummy"})
        with pytest.raises(ValueError):
            problem_definition.add_in_features_identifiers(
                [dummy_identifier, dummy_identifier]
            )
        problem_definition.add_in_feature_identifier(dummy_identifier)
        with pytest.raises(ValueError):
            problem_definition.add_in_feature_identifier(dummy_identifier)

    def test_add_in_features_identifiers(self, problem_definition):
        dummy_identifier_1 = FeatureIdentifier({"type": "scalar", "name": "dummy_1"})
        dummy_identifier_2 = FeatureIdentifier({"type": "scalar", "name": "dummy_2"})
        dummy_identifier_3 = FeatureIdentifier({"type": "scalar", "name": "dummy_3"})
        problem_definition.add_in_features_identifiers(
            [dummy_identifier_1, dummy_identifier_2]
        )
        problem_definition.add_in_feature_identifier(dummy_identifier_3)
        inputs = problem_definition.get_in_features_identifiers()
        assert len(inputs) == 3
        assert set(inputs) == set(
            [dummy_identifier_1, dummy_identifier_2, dummy_identifier_3]
        )
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_get_out_features_identifiers(self, problem_definition):
        assert problem_definition.get_out_features_identifiers() == []

    def test_add_out_features_identifiers_fail(self, problem_definition):
        dummy_identifier = FeatureIdentifier({"type": "scalar", "name": "dummy"})
        with pytest.raises(ValueError):
            problem_definition.add_out_features_identifiers(
                [dummy_identifier, dummy_identifier]
            )
        problem_definition.add_out_feature_identifier(dummy_identifier)
        with pytest.raises(ValueError):
            problem_definition.add_out_feature_identifier(dummy_identifier)

    def test_add_out_features_identifiers(self, problem_definition):
        dummy_identifier_1 = FeatureIdentifier({"type": "scalar", "name": "dummy_1"})
        dummy_identifier_2 = FeatureIdentifier({"type": "scalar", "name": "dummy_2"})
        dummy_identifier_3 = FeatureIdentifier({"type": "scalar", "name": "dummy_3"})
        problem_definition.add_out_features_identifiers(
            [dummy_identifier_1, dummy_identifier_2]
        )
        problem_definition.add_out_feature_identifier(dummy_identifier_3)
        outputs = problem_definition.get_out_features_identifiers()
        assert len(outputs) == 3
        assert set(outputs) == set(
            [dummy_identifier_1, dummy_identifier_2, dummy_identifier_3]
        )
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_filter_features_identifiers(self, current_directory):
        d_path = current_directory / "problem_definition"
        problem = ProblemDefinition(d_path)
        predict_feature_identifier = FeatureIdentifier(
            {"type": "scalar", "name": "predict_feature"}
        )
        test_feature_identifier = FeatureIdentifier(
            {"type": "scalar", "name": "test_feature"}
        )
        filter_in = problem.filter_in_features_identifiers(
            [predict_feature_identifier, test_feature_identifier]
        )
        filter_out = problem.filter_out_features_identifiers(
            [predict_feature_identifier, test_feature_identifier]
        )
        assert len(filter_in) == 2 and filter_in == [
            predict_feature_identifier,
            test_feature_identifier,
        ]
        assert filter_in != [test_feature_identifier, predict_feature_identifier], (
            "common inputs not sorted"
        )

        assert len(filter_out) == 2 and filter_out == [
            predict_feature_identifier,
            test_feature_identifier,
        ]
        assert filter_out != [test_feature_identifier, predict_feature_identifier], (
            "common outputs not sorted"
        )

        inexisting_feature_identifier = FeatureIdentifier(
            {"type": "scalar", "name": "inexisting_feature"}
        )
        fail_filter_in = problem.filter_in_features_identifiers(
            [inexisting_feature_identifier]
        )
        fail_filter_out = problem.filter_out_features_identifiers(
            [inexisting_feature_identifier]
        )

        assert fail_filter_in == []
        assert fail_filter_out == []

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # -------------------------------------------------------------------------#
    def test_get_input_scalars_names(self, problem_definition):
        assert problem_definition.get_input_scalars_names() == []

    def test_add_input_scalars_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_scalars_names(["feature_name", "feature_name"])
        problem_definition.add_input_scalar_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_input_scalar_name("feature_name")

    def test_add_input_scalars_names(self, problem_definition):
        problem_definition.add_input_scalars_names(["scalar", "test_scalar"])
        problem_definition.add_input_scalar_name("predict_scalar")
        inputs = problem_definition.get_input_scalars_names()
        assert len(inputs) == 3
        assert set(inputs) == set(["predict_scalar", "scalar", "test_scalar"])
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_get_output_scalars_names(self, problem_definition):
        assert problem_definition.get_output_scalars_names() == []

    def test_add_output_scalars_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_scalars_names(
                ["feature_name", "feature_name"]
            )
        problem_definition.add_output_scalar_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_output_scalar_name("feature_name")

    def test_add_output_scalars_names(self, problem_definition):
        problem_definition.add_output_scalars_names(["scalar", "test_scalar"])
        problem_definition.add_output_scalar_name("predict_scalar")
        outputs = problem_definition.get_output_scalars_names()
        assert len(outputs) == 3
        assert set(outputs) == set(["predict_scalar", "scalar", "test_scalar"])
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_filter_scalars_names(self, current_directory):
        d_path = current_directory / "problem_definition"
        problem = ProblemDefinition(d_path)
        filter_in = problem.filter_input_scalars_names(
            ["predict_scalar", "test_scalar"]
        )
        filter_out = problem.filter_output_scalars_names(
            ["predict_scalar", "test_scalar"]
        )
        assert len(filter_in) == 2 and filter_in == ["predict_scalar", "test_scalar"]
        assert filter_in != ["test_scalar", "predict_scalar"], (
            "common inputs not sorted"
        )

        assert len(filter_out) == 2 and filter_out == ["predict_scalar", "test_scalar"]
        assert filter_out != ["test_scalar", "predict_scalar"], (
            "common outputs not sorted"
        )

        fail_filter_in = problem.filter_input_scalars_names(["a_scalar"])
        fail_filter_out = problem.filter_output_scalars_names(["b_scalar"])

        assert fail_filter_in == []
        assert fail_filter_out == []

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # -------------------------------------------------------------------------#
    def test_get_input_fields_names(self, problem_definition):
        assert problem_definition.get_input_fields_names() == []

    def test_add_input_fields_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_fields_names(["feature_name", "feature_name"])
        problem_definition.add_input_field_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_input_field_name("feature_name")

    def test_add_input_fields_names(self, problem_definition):
        problem_definition.add_input_fields_names(["field", "test_field"])
        problem_definition.add_input_field_name("predict_field")
        inputs = problem_definition.get_input_fields_names()
        assert len(inputs) == 3
        assert set(inputs) == set(["predict_field", "field", "test_field"])
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_get_output_fields_names(self, problem_definition):
        assert problem_definition.get_output_fields_names() == []

    def test_add_output_fields_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_fields_names(["feature_name", "feature_name"])
        problem_definition.add_output_field_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_output_field_name("feature_name")

    def test_add_output_fields_names(self, problem_definition):
        problem_definition.add_output_fields_names(["field", "test_field"])
        problem_definition.add_output_field_name("predict_field")
        outputs = problem_definition.get_output_fields_names()
        assert len(outputs) == 3
        assert set(outputs) == set(["predict_field", "field", "test_field"])
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_filter_fields_names(self, current_directory):
        d_path = current_directory / "problem_definition"
        problem = ProblemDefinition(d_path)
        filter_in = problem.filter_input_fields_names(["predict_field", "test_field"])
        filter_out = problem.filter_output_fields_names(["predict_field", "test_field"])
        assert len(filter_in) == 2 and filter_in == ["predict_field", "test_field"]
        assert filter_in != ["test_field", "predict_field"], "common inputs not sorted"

        assert len(filter_out) == 2 and filter_out == ["predict_field", "test_field"]
        assert filter_out != ["test_field", "predict_field"], (
            "common outputs not sorted"
        )

        fail_filter_in = problem.filter_input_fields_names(["a_field"])
        fail_filter_out = problem.filter_output_fields_names(["b_field"])

        assert fail_filter_in == []
        assert fail_filter_out == []

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # -------------------------------------------------------------------------#
    def test_get_input_timeseries_names(self, problem_definition):
        assert problem_definition.get_input_timeseries_names() == []

    def test_add_input_timeseries_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_timeseries_names(
                ["feature_name", "feature_name"]
            )
        problem_definition.add_input_timeseries_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_input_timeseries_name("feature_name")

    def test_add_input_timeseries_names(self, problem_definition):
        problem_definition.add_input_timeseries_names(["timeseries", "test_timeseries"])
        problem_definition.add_input_timeseries_name("predict_timeseries")
        inputs = problem_definition.get_input_timeseries_names()
        assert len(inputs) == 3
        assert set(inputs) == set(
            ["predict_timeseries", "timeseries", "test_timeseries"]
        )
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_get_output_timeseries_names(self, problem_definition):
        assert problem_definition.get_output_timeseries_names() == []

    def test_add_output_timeseries_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_timeseries_names(
                ["feature_name", "feature_name"]
            )
        problem_definition.add_output_timeseries_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_output_timeseries_name("feature_name")

    def test_add_output_timeseries_names(self, problem_definition):
        problem_definition.add_output_timeseries_names(
            ["timeseries", "test_timeseries"]
        )
        problem_definition.add_output_timeseries_name("predict_timeseries")
        outputs = problem_definition.get_output_timeseries_names()
        assert len(outputs) == 3
        assert set(outputs) == set(
            ["predict_timeseries", "timeseries", "test_timeseries"]
        )
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_filter_timeseries_names(self, current_directory):
        d_path = current_directory / "problem_definition"
        problem = ProblemDefinition(d_path)
        filter_in = problem.filter_input_timeseries_names(
            ["predict_timeseries", "test_timeseries"]
        )
        filter_out = problem.filter_output_timeseries_names(
            ["predict_timeseries", "test_timeseries"]
        )
        assert len(filter_in) == 2 and filter_in == [
            "predict_timeseries",
            "test_timeseries",
        ]
        assert filter_in != ["test_timeseries", "predict_timeseries"], (
            "common inputs not sorted"
        )

        assert len(filter_out) == 2 and filter_out == [
            "predict_timeseries",
            "test_timeseries",
        ]
        assert filter_out != ["test_timeseries", "predict_timeseries"], (
            "common outputs not sorted"
        )

        fail_filter_in = problem.filter_input_timeseries_names(["a_timeseries"])
        fail_filter_out = problem.filter_output_timeseries_names(["b_timeseries"])

        assert fail_filter_in == []
        assert fail_filter_out == []

    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    # -------------------------------------------------------------------------#
    def test_get_input_meshes_names(self, problem_definition):
        assert problem_definition.get_input_meshes_names() == []

    def test_add_input_meshes_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_meshes_names(["feature_name", "feature_name"])
        problem_definition.add_input_mesh_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_input_mesh_name("feature_name")

    def test_add_input_meshes_names(self, problem_definition):
        problem_definition.add_input_meshes_names(["mesh", "test_mesh"])
        problem_definition.add_input_mesh_name("predict_mesh")
        inputs = problem_definition.get_input_meshes_names()
        assert len(inputs) == 3
        assert set(inputs) == set(["predict_mesh", "mesh", "test_mesh"])
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_get_output_meshes_names(self, problem_definition):
        assert problem_definition.get_output_meshes_names() == []

    def test_add_output_meshes_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_meshes_names(["feature_name", "feature_name"])
        problem_definition.add_output_mesh_name("feature_name")
        with pytest.raises(ValueError):
            problem_definition.add_output_mesh_name("feature_name")

    def test_add_output_meshes_names(self, problem_definition):
        problem_definition.add_output_meshes_names(["mesh", "test_mesh"])
        problem_definition.add_output_mesh_name("predict_mesh")
        outputs = problem_definition.get_output_meshes_names()
        assert len(outputs) == 3
        assert set(outputs) == set(["predict_mesh", "mesh", "test_mesh"])
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_filter_meshes_names(self, current_directory):
        d_path = current_directory / "problem_definition"
        problem = ProblemDefinition(d_path)
        print(f"{problem=}")
        print(f"{problem.get_input_meshes_names()=}")
        filter_in = problem.filter_input_meshes_names(["predict_mesh", "test_mesh"])
        filter_out = problem.filter_output_meshes_names(["predict_mesh", "test_mesh"])
        assert len(filter_in) == 2 and filter_in == ["predict_mesh", "test_mesh"]
        assert filter_in != ["test_mesh", "predict_mesh"], "common inputs not sorted"

        assert len(filter_out) == 2 and filter_out == ["predict_mesh", "test_mesh"]
        assert filter_out != ["test_mesh", "predict_mesh"], "common outputs not sorted"

        fail_filter_in = problem.filter_input_meshes_names(["a_mesh"])
        fail_filter_out = problem.filter_output_meshes_names(["b_mesh"])

        assert fail_filter_in == []
        assert fail_filter_out == []

    # -------------------------------------------------------------------------#
    def test_split(self, problem_definition):
        new_split = {"train": [0, 1, 2], "test": [3, 4]}
        problem_definition.set_split(new_split)
        assert problem_definition.get_split("train") == [0, 1, 2]
        assert problem_definition.get_split("test") == [3, 4]

        all_split = problem_definition.get_split()
        assert all_split["train"] == [0, 1, 2] and all_split["test"] == [3, 4]
        assert problem_definition.get_all_indices() == [0, 1, 2, 3, 4]

    def test_train_split(self, problem_definition):
        train_split = {"train1": [0, 1, 2], "train2": [3, 4]}
        problem_definition.set_train_split(train_split)
        problem_definition.get_train_split()
        assert problem_definition.get_train_split("train1") == [0, 1, 2]
        assert problem_definition.get_train_split("train2") == [3, 4]

    def test_test_split(self, problem_definition):
        test_split = {"test1": [0, 1, 2], "test2": [3, 4]}
        problem_definition.set_test_split(test_split)
        problem_definition.get_test_split()
        assert problem_definition.get_test_split("test1") == [0, 1, 2]
        assert problem_definition.get_test_split("test2") == [3, 4]

    # -------------------------------------------------------------------------#
    def test__save_to_dir_(
        self, problem_definition_full: ProblemDefinition, tmp_path: Path
    ):
        problem_definition_full._save_to_dir_(tmp_path / "problem_definition")

    def test_load_path_object(self, current_directory):
        my_dir = Path(current_directory)
        ProblemDefinition(my_dir / "problem_definition")

    def test___init___path(
        self, problem_definition_full: ProblemDefinition, tmp_path: Path
    ):
        d_path = tmp_path / "problem_definition"
        problem_definition_full._save_to_dir_(d_path)
        #
        problem = ProblemDefinition(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_input_scalars_names()) == set(
            ["predict_scalar", "scalar", "test_scalar"]
        )
        assert set(problem.get_output_scalars_names()) == set(
            ["predict_scalar", "scalar", "test_scalar"]
        )
        all_split = problem.get_split()
        assert all_split["train"] == [0, 1, 2] and all_split["test"] == [3, 4]

    def test__load_from_dir_(
        self, problem_definition_full: ProblemDefinition, tmp_path: Path
    ):
        d_path = tmp_path / "problem_definition"
        problem_definition_full._save_to_dir_(d_path)
        #
        problem = ProblemDefinition()
        problem._load_from_dir_(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_input_scalars_names()) == set(
            ["predict_scalar", "scalar", "test_scalar"]
        )
        assert set(problem.get_output_scalars_names()) == set(
            ["predict_scalar", "scalar", "test_scalar"]
        )
        all_split = problem.get_split()
        assert all_split["train"] == [0, 1, 2] and all_split["test"] == [3, 4]

    def test_load(self, problem_definition_full: ProblemDefinition, tmp_path: Path):
        d_path = tmp_path / "problem_definition"
        problem_definition_full._save_to_dir_(d_path)
        #
        problem = ProblemDefinition.load(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_input_scalars_names()) == set(
            ["predict_scalar", "scalar", "test_scalar"]
        )
        assert set(problem.get_output_scalars_names()) == set(
            ["predict_scalar", "scalar", "test_scalar"]
        )
        all_split = problem.get_split()
        assert all_split["train"] == [0, 1, 2] and all_split["test"] == [3, 4]

    def test__load_from_dir__old_version(
        self, problem_definition_full: ProblemDefinition, tmp_path: Path
    ):
        d_path = tmp_path / "problem_definition"
        problem_definition_full._save_to_dir_(d_path)
        # Modify the plaid version in saved file
        infos_path = d_path / "problem_infos.yaml"
        with infos_path.open("r") as f:
            text = f.read().splitlines()
        text.pop()
        text.append("version: 0.1.7")
        text.append("")
        infos_path.write_text("\n".join(text))

        # Load the problem definition from the directory
        problem = ProblemDefinition.load(d_path)
        assert problem.get_version() == Version("0.1.7")

    def test__load_from_dir__empty_dir(self, tmp_path):
        problem = ProblemDefinition()
        with pytest.raises(FileNotFoundError):
            problem._load_from_dir_(tmp_path)

    def test__load_from_dir__non_existing_dir(self):
        problem = ProblemDefinition()
        non_existing_dir = Path("non_existing_path")
        with pytest.raises(FileNotFoundError):
            problem._load_from_dir_(non_existing_dir)

    def test__load_from_dir__path_is_file(self, tmp_path):
        problem = ProblemDefinition()
        file_path = tmp_path / "file.yaml"
        file_path.touch()  # Create an empty file
        with pytest.raises(FileExistsError):
            problem._load_from_dir_(file_path)

    def test_extract_problem_definition_from_identifiers(self, problem_definition):
        in_id_1 = FeatureIdentifier({"type": "scalar", "name": "in_1"})
        in_id_2 = FeatureIdentifier({"type": "scalar", "name": "in_2"})
        out_id_1 = FeatureIdentifier({"type": "scalar", "name": "out_1"})
        out_id_2 = FeatureIdentifier({"type": "scalar", "name": "out_2"})

        problem_definition.add_in_features_identifiers([in_id_1, in_id_2])
        problem_definition.add_out_features_identifiers([out_id_1, out_id_2])
        problem_definition.set_task("regression")
        problem_definition.set_split({"train": [0, 1], "test": [2, 3]})

        sub_problem_definition = (
            problem_definition.extract_problem_definition_from_identifiers(
                [in_id_1, out_id_1]
            )
        )

        assert sub_problem_definition.get_in_features_identifiers() == [in_id_1]
        assert sub_problem_definition.get_out_features_identifiers() == [out_id_1]
        assert sub_problem_definition.get_version() == problem_definition.get_version()
        assert sub_problem_definition.get_task() == "regression"
        assert sub_problem_definition.get_split() == {"train": [0, 1], "test": [2, 3]}


# %%
