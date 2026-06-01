import pytest
from pydantic import ValidationError

from plaid.info import Info


def test_verify_info_accepts_special_internal_keys():
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
        "num_samples": {"train": 10},
        "storage_backend": "zarr",
    }
    Info(**infos)


def test_verify_info_rejects_unknown_category():
    with pytest.raises(ValidationError):
        Info(**{"unknown": {"x": "y"}})


def test_verify_info_rejects_unknown_key():
    with pytest.raises(ValidationError):
        Info(**{"legal": {"unknown_key": "v"}})


def test_validate_required_only_missing_required_key():
    with pytest.raises(ValueError):
        Info(**{"legal": {"owner": "someone"}})


def test_normalize_infos_adds_plaid_section_and_copies():
    infos = {"legal": {"owner": "owner", "license": "cc-by-4.0"}}
    normalized = Info.normalize_mapping(infos)

    assert "plaid" in normalized
    assert normalized["plaid"] == {}
    assert "plaid" not in infos


def test_dataset_info_model_validate_success():
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
        "num_samples": {"train": 10},
        "storage_backend": "zarr",
    }

    model = Info.model_validate({**infos, "plaid": {}})

    assert model.legal.owner == "owner"
    assert model.storage_backend == "zarr"


def test_dataset_info_model_validate_rejects_extra_top_level_key():
    with pytest.raises(ValueError):
        Info.model_validate(
            {
                "legal": {"owner": "owner", "license": "cc-by-4.0"},
                "unknown": {},
            }
        )
