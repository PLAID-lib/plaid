import pytest
from pydantic import ValidationError

from plaid.infos import Infos, Legal


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


def test_validate_authorized_only_allows_missing_legal():
    model = Infos.validate_authorized_only(
        {"num_samples": {"train": 1}, "storage_backend": "zarr"}
    )
    # Missing legal must be filled with empty placeholder values.
    assert model.legal.owner == ""
    assert model.legal.license == ""
    assert model.storage_backend == "zarr"


def test_validate_authorized_only_with_legal_present():
    model = Infos.validate_authorized_only({"legal": {"owner": "o", "license": "l"}})
    assert model.legal.owner == "o"


def test_validate_authorized_only_rejects_unauthorized_key():
    with pytest.raises(KeyError):
        Infos.validate_authorized_only(
            {"legal": {"owner": "o", "license": "l"}, "unknown": {}}
        )


def test_validate_authorized_only_reraises_other_validation_errors():
    # ``num_samples`` expects dict[str, int] - giving a string triggers a
    # validation error that is *not* of the unauthorized-key kind, so it
    # must be re-raised as ValidationError.
    with pytest.raises(ValidationError):
        Infos.validate_authorized_only(
            {"legal": {"owner": "o", "license": "l"}, "num_samples": "nope"}
        )


def test_validate_required_only_accepts_valid_mapping():
    Infos.validate_required_only({"legal": {"owner": "o", "license": "l"}})


def test_validate_required_only_missing_legal():
    with pytest.raises(ValidationError):
        Infos.validate_required_only({})


def test_to_dict_returns_plain_mapping():
    model = Infos.from_mapping(
        {"legal": {"owner": "o", "license": "l"}, "storage_backend": "zarr"}
    )
    d = model.to_dict()
    assert d["legal"] == {"owner": "o", "license": "l"}
    assert d["storage_backend"] == "zarr"


def test_getitem_returns_plain_value_and_unwraps_nested_dataclasses():
    model = Infos.from_mapping(
        {
            "legal": {"owner": "o", "license": "l"},
            "storage_backend": "zarr",
        }
    )
    # Plain field.
    assert model["storage_backend"] == "zarr"
    # Nested dataclass is returned as a dict, with None entries dropped.
    legal_dict = model["legal"]
    assert legal_dict == {"owner": "o", "license": "l"}


def test_getitem_raises_key_error_for_unknown_field():
    model = Infos.from_mapping({"legal": {"owner": "o", "license": "l"}})
    with pytest.raises(KeyError):
        model["does_not_exist"]


def test_contains_handles_strings_and_non_strings():
    model = Infos.from_mapping(
        {"legal": {"owner": "o", "license": "l"}, "storage_backend": "zarr"}
    )
    assert "legal" in model
    assert "storage_backend" in model
    # Field exists but is None -> not "in" the model.
    assert "data_production" not in model
    # Unknown attribute name.
    assert "nope" not in model
    # Non-string keys are never in the model.
    assert (123 in model) is False


def test_get_returns_default_when_missing():
    model = Infos.from_mapping({"legal": {"owner": "o", "license": "l"}})
    assert model.get("storage_backend", "fallback") == "fallback"
    assert model.get("legal")["owner"] == "o"


def test_save_to_file_treats_suffixless_path_as_directory(tmp_path):
    target = tmp_path / "myinfos"
    Infos(legal=Legal(owner="o", license="l")).save_to_file(target)
    # Suffix-less, non-existing paths are treated as directories that
    # will hold an ``infos.yaml``.
    assert (target / "infos.yaml").is_file()


def test_save_to_file_into_existing_directory(tmp_path):
    Infos(legal=Legal(owner="o", license="l")).save_to_file(tmp_path)
    assert (tmp_path / "infos.yaml").is_file()


def test_save_to_file_replaces_non_yaml_suffix(tmp_path):
    target = tmp_path / "weird.txt"
    Infos(legal=Legal(owner="o", license="l")).save_to_file(target)
    # ``.txt`` suffix is replaced by ``.yaml``.
    assert (tmp_path / "weird.yaml").is_file()
    assert not target.exists()


def test_save_to_file_preserves_unknown_future_keys(tmp_path, monkeypatch):
    """Fields outside ``_KEY_ORDER`` should still be serialised.

    Emptying ``_KEY_ORDER`` forces every dumped key to take the
    "future field" branch in ``save_to_file`` so we cover the
    "preserve any future fields" loop body.
    """
    from plaid import infos as infos_mod

    monkeypatch.setattr(infos_mod, "_KEY_ORDER", ())
    model = Infos.from_mapping(
        {"legal": {"owner": "o", "license": "l"}, "storage_backend": "zarr"}
    )
    target = tmp_path / "out.yaml"
    model.save_to_file(target)
    text = target.read_text(encoding="utf-8")
    assert "storage_backend" in text
    assert "legal" in text


def test_from_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Infos.from_path(tmp_path / "missing.yaml")
