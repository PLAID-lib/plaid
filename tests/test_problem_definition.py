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
    return ProblemDefinition.model_construct(
        input_features=[],
        output_features=[],
        train_split=None,
        test_split=None,
    )


@pytest.fixture()
def problem_definition_full(problem_definition: ProblemDefinition) -> ProblemDefinition:
    # ----
    feature_identifier = "Global/feature"
    predict_feature_identifier = "Global/predict_feature"
    test_feature_identifier = "Global/test_feature"

    problem_definition.add_input_features(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_input_features(feature_identifier)
    problem_definition.add_output_features(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_output_features(feature_identifier)
    # ----
    feature_identifier = "Base_2_2/Zone/PointData/U1"
    predict_feature_identifier = "Base_2_2/Zone/PointData/U2"
    test_feature_identifier = "Base_2_2/Zone/PointData/sig12"
    problem_definition.add_input_features(
        [predict_feature_identifier, test_feature_identifier]
    )
    problem_definition.add_input_features(feature_identifier)
    problem_definition.add_output_features(
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
        print(problem_definition)

    def test_required_fields(self):
        with pytest.raises(ValidationError, match="Field required"):
            ProblemDefinition()

    def test_feature_lists_must_not_be_empty(self):
        base = {
            "train_split": {"train": "all"},
            "test_split": {"test": "all"},
        }
        with pytest.raises(ValidationError, match="input_features must not be empty"):
            ProblemDefinition(
                **base,
                input_features=[],
                output_features=["out"],
            )
        with pytest.raises(ValidationError, match="output_features must not be empty"):
            ProblemDefinition(
                **base,
                input_features=["in"],
                output_features=[],
            )

    # -------------------------------------------------------------------------#

    def test_from_mapping_validates_and_normalizes(self):
        loaded = ProblemDefinition.model_validate(
            {
                "input_features": ["in_b", "in_a"],
                "output_features": ["out_b", "out_a"],
                "train_split": {"train_0": [0, 1]},
                "test_split": {"test_0": [2]},
            }
        )

        assert loaded.input_features == ["in_a", "in_b"]
        assert loaded.output_features == ["out_a", "out_b"]
        assert loaded.train_split == {"train_0": [0, 1]}
        assert loaded.test_split == {"test_0": [2]}

    def test_from_path_loads_single_yaml_file(self, tmp_path: Path):
        file_path = tmp_path / "problem.yaml"
        file_path.write_text(
            "input_features:\n"
            "  - in_1\n"
            "output_features:\n"
            "  - out_1\n"
            "train_split:\n"
            "  train: [0]\n"
            "test_split:\n"
            "  test: [1]\n",
            encoding="utf-8",
        )

        loaded = ProblemDefinition.from_path(file_path)

        assert loaded.train_split == {"train": [0]}
        assert loaded.test_split == {"test": [1]}

    def test_from_path_adds_yaml_suffix(self, tmp_path: Path):
        file_path = tmp_path / "problem.yaml"
        file_path.write_text(
            "input_features: [in_1]\n"
            "output_features: [out_1]\n"
            "train_split:\n"
            "  train: all\n"
            "test_split:\n"
            "  test: all\n",
            encoding="utf-8",
        )

        loaded = ProblemDefinition.from_path(tmp_path / "problem")

        assert loaded.input_features == ["in_1"]

    def test_from_path_rejects_old_name_key(self, tmp_path: Path):
        file_path = tmp_path / "problem_with_name.yaml"
        file_path.write_text(
            "name: pb\n"
            "input_features: [in_1]\n"
            "output_features: [out_1]\n"
            "train_split:\n"
            "  train: all\n"
            "test_split:\n"
            "  test: all\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="extra_forbidden"):
            ProblemDefinition.from_path(file_path)

    def test_from_path_unknown_key_raises(self, tmp_path: Path):
        file_path = tmp_path / "problem_with_unknown.yaml"
        file_path.write_text(
            "input_features: [in_1]\n"
            "output_features: [out_1]\n"
            "train_split:\n"
            "  train: all\n"
            "test_split:\n"
            "  test: all\n"
            "unknown_key: value\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="extra_forbidden"):
            ProblemDefinition.from_path(file_path)

    def test_from_path_non_existing_file(self):
        with pytest.raises(FileNotFoundError):
            ProblemDefinition.from_path(Path("non_existing_path"))

    def test_from_path_rejects_directory(self, tmp_path: Path):
        with pytest.raises(IsADirectoryError, match="Expected a YAML file path"):
            ProblemDefinition.from_path(tmp_path)

    def test_feature_validators_reject_duplicates(self):
        with pytest.raises(
            ValidationError, match="duplicated values in input_features"
        ):
            ProblemDefinition(
                input_features=["a", "a"],
                output_features=["out"],
                train_split={"train": "all"},
                test_split={"test": "all"},
            )

        with pytest.raises(
            ValidationError, match="duplicated values in output_features"
        ):
            ProblemDefinition(
                input_features=["in"],
                output_features=["a", "a"],
                train_split={"train": "all"},
                test_split={"test": "all"},
            )

    def test_feature_validators_reject_input_output_overlap(self):
        with pytest.raises(
            ValidationError, match="features cannot be both input and output"
        ):
            ProblemDefinition(
                input_features=["shared"],
                output_features=["shared"],
                train_split={"train": "all"},
                test_split={"test": "all"},
            )

    def test_split_replacement_logs_warning(self, problem_definition, caplog):
        problem_definition.train_split = {"train_0": [0, 1]}
        with caplog.at_level("WARNING"):
            problem_definition.train_split = {"train_1": [2, 3]}

        assert "already exists -> data will be replaced" in caplog.text

    def test_split_fields_are_plain_dictionaries(self, problem_definition):
        problem_definition.train_split = {"train_0": [0, 1, 2]}
        problem_definition.test_split = {"test_0": [3, 4]}

        assert problem_definition.train_split == {"train_0": [0, 1, 2]}
        assert problem_definition.test_split == {"test_0": [3, 4]}

    def test_add_feature_identifiers_duplicate_checks(self, problem_definition):
        problem_definition.add_input_features(["in_1", "in_2"])
        with pytest.raises(ValueError, match="in_1 is already in"):
            problem_definition.add_input_features("in_1")
        with pytest.raises(
            ValueError, match="Some input features share the same identifier"
        ):
            problem_definition.add_input_features(["x", "x"])

        problem_definition.add_output_features(["out_1", "out_2"])
        with pytest.raises(ValueError, match="out_1 is already in"):
            problem_definition.add_output_features("out_1")
        with pytest.raises(
            ValueError, match="Some output features share the same identifier"
        ):
            problem_definition.add_output_features(["y", "y"])

    # -------------------------------------------------------------------------#
    def test_split(self, problem_definition):
        problem_definition.train_split = {"train_0": [0, 1, 2]}
        problem_definition.test_split = {"test-1": [3, 4]}
        assert problem_definition.train_split == {"train_0": [0, 1, 2]}
        assert problem_definition.test_split == {"test-1": [3, 4]}

    def test_save_and_load_keep_yaml_suffix(self, tmp_path: Path):
        problem = ProblemDefinition(
            input_features=["in_1"],
            output_features=["out_1"],
            train_split={"train": [0]},
            test_split={"test": [1]},
        )
        file_path = tmp_path / "problem.yaml"
        problem.save_to_file(file_path)

        loaded = ProblemDefinition.from_path(file_path)

        saved_text = file_path.read_text(encoding="utf-8")
        assert "name:" not in saved_text
        assert loaded.train_split == {"train": [0]}
        assert loaded.test_split == {"test": [1]}

    def test_save_to_file_rejects_directory(self, tmp_path: Path):
        problem = ProblemDefinition(
            input_features=["in_1"],
            output_features=["out_1"],
            train_split={"train": [0]},
            test_split={"test": [1]},
        )

        with pytest.raises(IsADirectoryError, match="Expected a YAML file path"):
            problem.save_to_file(tmp_path)
