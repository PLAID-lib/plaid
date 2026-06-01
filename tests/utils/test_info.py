import pytest
from pydantic import ValidationError

from plaid.infos import Infos


def test_verify_info_accepts_special_internal_keys():
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
        "num_samples": {"train": 10},
        "storage_backend": "zarr",
    }
    Infos(**infos)


def test_verify_info_rejects_unknown_category():
    with pytest.raises(ValidationError):
        Infos(**{"unknown": {"x": "y"}})


def test_verify_info_rejects_unknown_key():
    with pytest.raises(ValidationError):
        Infos(**{"legal": {"unknown_key": "v"}})


def test_validate_required_only_missing_required_key():
    with pytest.raises(ValueError):
        Infos(**{"legal": {"owner": "someone"}})


def test_normalize_infos_strips_legacy_plaid_section_and_copies():
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
        "plaid": {"version": "x"},
    }
    normalized = Infos.normalize_mapping(infos)

    # The legacy ``plaid`` section is dropped from the validated payload.
    assert "plaid" not in normalized
    # And the input mapping is not mutated.
    assert "plaid" in infos


def test_model_validate_rejects_plaid_section():
    with pytest.raises(ValidationError):
        Infos.model_validate(
            {
                "legal": {"owner": "owner", "license": "cc-by-4.0"},
                "plaid": {"version": "x"},
            }
        )


def test_dataset_info_model_validate_success():
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
        "num_samples": {"train": 10},
        "storage_backend": "zarr",
    }

    model = Infos.model_validate(infos)

    assert model.legal.owner == "owner"
    assert model.storage_backend == "zarr"


def test_dataset_info_model_validate_rejects_extra_top_level_key():
    with pytest.raises(ValueError):
        Infos.model_validate(
            {
                "legal": {"owner": "owner", "license": "cc-by-4.0"},
                "unknown": {},
            }
        )


def test_infos_save_and_load_roundtrip(tmp_path):
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
        "num_samples": {"train": 10},
        "storage_backend": "zarr",
    }
    model = Infos.from_mapping(infos)

    target = tmp_path / "infos.yaml"
    model.save_to_file(target)
    assert target.is_file()

    reloaded = Infos.from_path(target)
    assert reloaded.legal.owner == "owner"
    assert reloaded.storage_backend == "zarr"
    assert reloaded.num_samples == {"train": 10}


def test_infos_from_path_directory(tmp_path):
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
    }
    Infos.from_mapping(infos).save_to_file(tmp_path / "infos.yaml")
    reloaded = Infos.from_path(tmp_path)
    assert reloaded.legal.license == "cc-by-4.0"
