#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import pytest

from plaid.utils.info import normalize_infos, validate_required_infos, verify_info


def test_verify_info_accepts_special_internal_keys():
    infos = {
        "legal": {"owner": "owner", "license": "cc-by-4.0"},
        "num_samples": {"train": 10},
        "storage_backend": "zarr",
    }
    verify_info(infos)


def test_verify_info_rejects_unknown_category():
    with pytest.raises(KeyError):
        verify_info({"unknown": {"x": "y"}})


def test_verify_info_rejects_unknown_key():
    with pytest.raises(KeyError):
        verify_info({"legal": {"unknown_key": "v"}})


def test_validate_required_infos_missing_required_key():
    with pytest.raises(ValueError):
        validate_required_infos({"legal": {"owner": "someone"}})


def test_normalize_infos_adds_plaid_section_and_copies():
    infos = {"legal": {"owner": "owner", "license": "cc-by-4.0"}}
    normalized = normalize_infos(infos)

    assert "plaid" in normalized
    assert normalized["plaid"] == {}
    assert "plaid" not in infos
