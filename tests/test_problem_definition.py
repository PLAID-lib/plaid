# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from pathlib import Path

import pytest
from packaging.version import Version

import plaid
from plaid.containers import FeatureIdentifier
from plaid.problem_definition import ProblemDefinition


@pytest.fixture()
def problem_definition() -> ProblemDefinition:
    return ProblemDefinition()


@pytest.fixture()
def problem_definition_full(problem_definition: ProblemDefinition) -> ProblemDefinition:
    problem_definition.set_task("regression")
    problem_definition.set_name("regression_1")

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

    str_feature = "Base_2_2/Zone/PointData/U1"
    predict_str_feature = "Base_2_2/Zone/PointData/U2"
    test_str_feature = "Base_2_2/Zone/PointData/sig12"
    problem_definition.add_in_features_identifiers(
        [predict_str_feature, test_str_feature]
    )
    problem_definition.add_in_feature_identifier(str_feature)
    problem_definition.add_out_features_identifiers(
        [predict_str_feature, test_str_feature]
    )
    problem_definition.add_constant_feature_identifier(str_feature)
    problem_definition.add_constant_features_identifiers(
        [predict_str_feature, test_str_feature]
    )

    new_split = {"train": [0, 1, 2], "test": [3, 4]}
    problem_definition.set_split(new_split)

    new_train_split = {"train_1": {"train": [0, 1]}, "train_2": {"train": "all"}}
    problem_definition.set_train_split(new_train_split)

    new_test_split = {"test_1": {"test": "all"}, "test_2": {"test": [0, 2]}}
    problem_definition.set_test_split(new_test_split)

    return problem_definition


@pytest.fixture()
def current_directory() -> Path:
    return Path(__file__).absolute().parent


class TestProblemDefinition:
    def test_init(self, problem_definition: ProblemDefinition):
        assert problem_definition.get_task() is None
        assert problem_definition.get_version() == Version(plaid.__version__)

    def test_load_from_dir(self, current_directory: Path):
        d_path = current_directory / "problem_definition"
        pb = ProblemDefinition.load(d_path)
        assert isinstance(pb, ProblemDefinition)

    def test_task(self, problem_definition: ProblemDefinition):
        with pytest.raises(TypeError):
            problem_definition.set_task("not_valid")
        problem_definition.set_task("classification")
        with pytest.raises(ValueError):
            problem_definition.set_task("regression")
        assert problem_definition.get_task() == "classification"

    def test_score_function(self, problem_definition: ProblemDefinition):
        with pytest.raises(TypeError):
            problem_definition.set_score_function("not_valid")
        problem_definition.set_score_function("RRMSE")
        with pytest.raises(ValueError):
            problem_definition.set_score_function("RRMSE")
        assert problem_definition.get_score_function() == "RRMSE"

    def test_add_in_features_identifiers(self, problem_definition: ProblemDefinition):
        dummy_identifier_1 = FeatureIdentifier({"type": "scalar", "name": "dummy_1"})
        dummy_identifier_2 = FeatureIdentifier({"type": "scalar", "name": "dummy_2"})
        dummy_identifier_3 = FeatureIdentifier({"type": "scalar", "name": "dummy_3"})
        problem_definition.add_in_features_identifiers(
            [dummy_identifier_1, dummy_identifier_2]
        )
        problem_definition.add_in_feature_identifier(dummy_identifier_3)
        inputs = problem_definition.get_in_features_identifiers()
        assert len(inputs) == 3
        assert set(inputs) == {
            dummy_identifier_1,
            dummy_identifier_2,
            dummy_identifier_3,
        }
        with pytest.raises(ValueError):
            problem_definition.add_in_feature_identifier(dummy_identifier_1)

    def test_add_out_features_identifiers(self, problem_definition: ProblemDefinition):
        dummy_identifier_1 = FeatureIdentifier({"type": "scalar", "name": "dummy_1"})
        dummy_identifier_2 = FeatureIdentifier({"type": "scalar", "name": "dummy_2"})
        dummy_identifier_3 = FeatureIdentifier({"type": "scalar", "name": "dummy_3"})
        problem_definition.add_out_features_identifiers(
            [dummy_identifier_1, dummy_identifier_2]
        )
        problem_definition.add_out_feature_identifier(dummy_identifier_3)
        outputs = problem_definition.get_out_features_identifiers()
        assert len(outputs) == 3
        assert set(outputs) == {
            dummy_identifier_1,
            dummy_identifier_2,
            dummy_identifier_3,
        }
        with pytest.raises(ValueError):
            problem_definition.add_out_feature_identifier(dummy_identifier_1)

    def test_constant_features(self, problem_definition: ProblemDefinition):
        dummy_identifier_1 = "Base_2_2/Zone/PointData/U1"
        dummy_identifier_2 = "Base_2_2/Zone/PointData/U2"
        dummy_identifier_3 = "Base_2_2/Zone/PointData/sig12"
        problem_definition.add_constant_features_identifiers(
            [dummy_identifier_1, dummy_identifier_2]
        )
        problem_definition.add_constant_feature_identifier(dummy_identifier_3)
        constants = problem_definition.get_constant_features_identifiers()
        assert len(constants) == 3
        assert set(constants) == {
            dummy_identifier_1,
            dummy_identifier_2,
            dummy_identifier_3,
        }
        with pytest.raises(ValueError):
            problem_definition.add_constant_feature_identifier(dummy_identifier_1)

    def test_split(self, problem_definition: ProblemDefinition):
        new_split = {"train": [0, 1, 2], "test": [3, 4]}
        problem_definition.set_split(new_split)
        assert set(problem_definition.get_split().keys()) == {"train", "test"}
        assert set(problem_definition.get_all_indices()) == {0, 1, 2, 3, 4}
        assert problem_definition.get_split("train") == [0, 1, 2]
        with pytest.raises(KeyError):
            problem_definition.get_split("val")

    def test_train_test_split(self, problem_definition: ProblemDefinition):
        train_split = {"train_1": {"train": [0, 1]}, "train_2": {"train": "all"}}
        test_split = {"test_1": {"test": "all"}, "test_2": {"test": [0, 2]}}
        problem_definition.set_train_split(train_split)
        problem_definition.set_test_split(test_split)
        assert problem_definition.get_train_split("train_1") == {"train": [0, 1]}
        assert problem_definition.get_test_split("test_2") == {"test": [0, 2]}
        with pytest.raises(KeyError):
            problem_definition.get_test_split("missing")

    def test_save_load_roundtrip(
        self, problem_definition_full: ProblemDefinition, tmp_path: Path
    ):
        out_dir = tmp_path / "pb_def"
        problem_definition_full.save_to_dir(out_dir)
        reloaded = ProblemDefinition.load(out_dir)
        assert reloaded.model_dump() == problem_definition_full.model_dump()
