# %% Imports

import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from plaid.problem_definition import ProblemDefinition

# %% Fixtures


@pytest.fixture()
def problem_definition() -> ProblemDefinition:
    return ProblemDefinition()


@pytest.fixture()
def problem_definition_full(problem_definition: ProblemDefinition) -> ProblemDefinition:
    problem_definition.task = "regression"
    problem_definition.name = "regression_1"

    # ----
    feature_identifier = "Global/feature"
    predict_feature_identifier = "Global/predict_feature"
    test_feature_identifier = "Global/test_feature"

    problem_definition.add_in_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_in_features_identifiers(feature_identifier)
    problem_definition.add_out_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_out_features_identifiers(feature_identifier)
    # ----
    feature_identifier = "Base_2_2/Zone/PointData/U1"
    predict_feature_identifier = "Base_2_2/Zone/PointData/U2"
    test_feature_identifier = "Base_2_2/Zone/PointData/sig12"
    problem_definition.add_in_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_in_features_identifiers(feature_identifier)
    problem_definition.add_out_features_identifiers(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.train_split = {"train_1": [0, 1, 2]}
    problem_definition.test_split = {"test_1": "all"}

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
        assert problem_definition.task is None
        print(problem_definition)

    def test__init__both_path_and_directory_path(self, current_directory):
        d_path = current_directory / "problem_definition"
        with pytest.raises(ValueError):
            ProblemDefinition(path=d_path, directory_path=d_path)

    # -------------------------------------------------------------------------#
    def test_task(self, problem_definition):
        # Unauthorized task
        with pytest.raises(ValidationError):
            problem_definition.task = "ighyurgv"
        problem_definition.task = "classification"
        assert problem_definition.task == "classification"
        print(problem_definition)

    # -------------------------------------------------------------------------#
    def test_score_function(self, problem_definition):
        # Unauthorized task
        with pytest.raises(ValidationError):
            problem_definition.score_function = "ighyurgv"
        problem_definition.score_function = "RRMSE"
        # can be set again to the same value
        problem_definition.score_function = "RRMSE"
        assert problem_definition.score_function == "RRMSE"
        print(problem_definition)

    def test_from_path_single_definition(self, monkeypatch, tmp_path):
        expected = ProblemDefinition(
            name="pb_single",
            task="regression",
            input_features=["in_a"],
            output_features=["out_a"],
            train_split={"train_0": [0, 1]},
            test_split={"test_0": [2]},
        )

        def fake_loader(path):
            assert path == tmp_path
            return {"pb_single": expected}

        monkeypatch.setattr(
            "plaid.storage.load_problem_definitions_from_disk", fake_loader
        )

        loaded = ProblemDefinition.from_path(tmp_path)
        assert loaded.name == "pb_single"
        assert loaded.task == "regression"
        assert loaded.input_features == ["in_a"]
        assert loaded.output_features == ["out_a"]
        assert loaded.get_train_split_name() == "train_0"
        assert loaded.get_test_split_name() == "test_0"
        assert loaded.get_train_split_indices() == [0, 1]
        assert loaded.get_test_split_indices() == [2]

    def test_from_path_single_definition_with_override(self, monkeypatch, tmp_path):
        expected = ProblemDefinition(
            name="pb_single",
            task="regression",
            input_features=["in_a"],
            output_features=["out_a"],
            train_split={"train_0": [0, 1]},
            test_split={"test_0": [2]},
        )

        monkeypatch.setattr(
            "plaid.storage.load_problem_definitions_from_disk",
            lambda path: {"pb_single": expected},
        )

        loaded = ProblemDefinition.from_path(tmp_path)
        assert loaded.name == "pb_single"

    def test_from_path_named_definition_and_override(self, monkeypatch, tmp_path):
        pb_1 = ProblemDefinition(
            name="pb_1",
            task="regression",
            input_features=["in_a"],
            output_features=["out_a"],
            train_split={"train_0": [0, 1]},
            test_split={"test_0": [2]},
        )
        pb_2 = ProblemDefinition(
            name="pb_2",
            task="classification",
            input_features=["in_b"],
            output_features=["out_b"],
            train_split={"train_1": [3, 4]},
            test_split={"test_1": [5]},
        )

        def fake_loader(path):
            assert path == tmp_path
            return {"pb_1": pb_1, "pb_2": pb_2}

        monkeypatch.setattr(
            "plaid.storage.load_problem_definitions_from_disk", fake_loader
        )

        loaded = ProblemDefinition.from_path(
            tmp_path,
            name="pb_2",
            score_function="RRMSE",
        )
        assert loaded.name == "pb_2"
        assert loaded.task == "classification"
        assert loaded.score_function == "RRMSE"

    def test_from_path_unknown_name_raises(self, monkeypatch, tmp_path):
        pb = ProblemDefinition(
            name="existing",
            task="regression",
            input_features=["in_a"],
            output_features=["out_a"],
            train_split={"train_0": [0, 1]},
            test_split={"test_0": [2]},
        )

        monkeypatch.setattr(
            "plaid.storage.load_problem_definitions_from_disk",
            lambda path: {"existing": pb},
        )

        with pytest.raises(ValueError, match="Problem definition 'missing' not found"):
            ProblemDefinition.from_path(tmp_path, name="missing")

    def test_from_path_requires_name_when_multiple(self, monkeypatch, tmp_path):
        pb_1 = ProblemDefinition(
            name="pb_1",
            task="regression",
            input_features=["in_a"],
            output_features=["out_a"],
            train_split={"train_0": [0, 1]},
            test_split={"test_0": [2]},
        )
        pb_2 = ProblemDefinition(
            name="pb_2",
            task="classification",
            input_features=["in_b"],
            output_features=["out_b"],
            train_split={"train_1": [3, 4]},
            test_split={"test_1": [5]},
        )

        monkeypatch.setattr(
            "plaid.storage.load_problem_definitions_from_disk",
            lambda path: {"pb_1": pb_1, "pb_2": pb_2},
        )

        with pytest.raises(RuntimeError, match="more than one Problem definition"):
            ProblemDefinition.from_path(tmp_path)

    def test_from_path_error_lists_sorted_available_names(self, monkeypatch, tmp_path):
        pb_a = ProblemDefinition(name="a", input_features=["in"], output_features=["out"])
        pb_b = ProblemDefinition(name="b", input_features=["in"], output_features=["out"])

        monkeypatch.setattr(
            "plaid.storage.load_problem_definitions_from_disk",
            lambda path: {"b": pb_b, "a": pb_a},
        )

        with pytest.raises(ValueError, match="Available definitions: a, b"):
            ProblemDefinition.from_path(tmp_path, name="missing")

    def test_feature_validators_reject_duplicates(self):
        with pytest.raises(
            ValidationError, match="duplicated values in input_features"
        ):
            ProblemDefinition(input_features=["a", "a"])

        with pytest.raises(
            ValidationError, match="duplicated values in output_features"
        ):
            ProblemDefinition(output_features=["a", "a"])

    def test_split_validator_rejects_more_than_one_key(self):
        with pytest.raises(ValidationError, match="Splits only support one element"):
            ProblemDefinition(train_split={"train_1": [0], "train_2": [1]})

    def test_non_overwritable_attributes_raise(self, problem_definition):
        problem_definition.name = "problem_a"
        with pytest.raises(AttributeError, match="'name' is already set"):
            problem_definition.name = "problem_b"

        problem_definition.task = "regression"
        with pytest.raises(AttributeError, match="'task' is already set"):
            problem_definition.task = "classification"

        problem_definition.score_function = "RRMSE"
        with pytest.raises(AttributeError, match="'score_function' is already set"):
            problem_definition.score_function = "MSE"

    def test_split_replacement_logs_warning(self, problem_definition, caplog):
        problem_definition.train_split = {"train_0": [0, 1]}
        with caplog.at_level("WARNING"):
            problem_definition.train_split = {"train_1": [2, 3]}

        assert "already exists -> data will be replaced" in caplog.text

    def test_get_split_paths(self, problem_definition):
        problem_definition.train_split = {"train_0": [0, 1, 2]}
        problem_definition.test_split = {"test_0": [3, 4]}

        assert problem_definition.get_train_split_name() == "train_0"
        assert problem_definition.get_test_split_name() == "test_0"
        assert problem_definition.get_train_split_indices() == [0, 1, 2]
        assert problem_definition.get_test_split_indices() == [3, 4]

    def test_get_split_paths_raise_when_not_defined(self, problem_definition):
        with pytest.raises(ValueError, match="train_split is not defined"):
            problem_definition.get_train_split_name()
        with pytest.raises(ValueError, match="train_split is not defined"):
            problem_definition.get_train_split_indices()
        with pytest.raises(ValueError, match="test_split is not defined"):
            problem_definition.get_test_split_name()
        with pytest.raises(ValueError, match="test_split is not defined"):
            problem_definition.get_test_split_indices()

    def test_add_feature_identifiers_duplicate_checks(self, problem_definition):
        problem_definition.add_in_features_identifiers(["in_1", "in_2"])
        with pytest.raises(ValueError, match="in_1 is already in"):
            problem_definition.add_in_features_identifiers("in_1")
        with pytest.raises(
            ValueError, match="Some input features share the same identifier"
        ):
            problem_definition.add_in_features_identifiers(["x", "x"])

        problem_definition.add_out_features_identifiers(["out_1", "out_2"])
        with pytest.raises(ValueError, match="out_1 is already in"):
            problem_definition.add_out_features_identifiers("out_1")
        with pytest.raises(
            ValueError, match="Some output features share the same identifier"
        ):
            problem_definition.add_out_features_identifiers(["y", "y"])

    # -------------------------------------------------------------------------#
    def test_split(self, problem_definition):
        problem_definition.train_split = {"train_0": [0, 1, 2]}
        problem_definition.test_split = {"test-1": [3, 4]}
        assert problem_definition.train_split == {"train_0": [0, 1, 2]}
        assert problem_definition.test_split == {"test-1": [3, 4]}

    def test__load_from_file_(
        self, problem_definition_full: ProblemDefinition, tmp_path: Path
    ):

        path = tmp_path / "pb_def"
        problem_definition_full.save_to_file(path)
        problem = ProblemDefinition()
        problem._load_from_file_(path)
        assert problem.task == "regression"
        assert set(problem.input_features) == set(
            [
                "Base_2_2/Zone/PointData/sig12",
                "Base_2_2/Zone/PointData/U1",
                "Base_2_2/Zone/PointData/U2",
                "Global/predict_feature",
                "Global/test_feature",
                "Global/feature",
            ]
        )
        assert set(problem.output_features) == set(
            [
                "Global/predict_feature",
                "Base_2_2/Zone/PointData/sig12",
                "Global/feature",
                "Base_2_2/Zone/PointData/U2",
                "Global/test_feature",
            ]
        )

    def test_save_and_load_keep_yaml_suffix(self, tmp_path: Path):
        problem = ProblemDefinition(
            name="pb",
            task="regression",
            input_features=["in_1"],
            output_features=["out_1"],
            train_split={"train": [0]},
            test_split={"test": [1]},
        )
        file_path = tmp_path / "problem.yaml"
        problem.save_to_file(file_path)

        loaded = ProblemDefinition()
        loaded._load_from_file_(file_path)

        assert loaded.name == "pb"
        assert loaded.task == "regression"
        assert loaded.get_train_split_name() == "train"
        assert loaded.get_test_split_name() == "test"

    def test__load_from_file__unknown_field_warns_and_raises(self, tmp_path: Path, caplog):
        file_path = tmp_path / "problem_with_unknown.yaml"
        file_path.write_text(
            "name: pb\n"
            "task: regression\n"
            "input_features:\n"
            "  - in_1\n"
            "output_features:\n"
            "  - out_1\n"
            "unknown_key: value\n",
            encoding="utf-8",
        )

        problem = ProblemDefinition()
        with caplog.at_level("WARNING"):
            problem._load_from_file_(file_path)

        assert "Data ignored! : unknown_key: value" in caplog.text

    def test__load_from_file__non_existing_file(self):
        problem = ProblemDefinition()
        non_existing_path = Path("non_existing_path")
        with pytest.raises(FileNotFoundError):
            problem._load_from_file_(non_existing_path)
