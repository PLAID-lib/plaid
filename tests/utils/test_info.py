import pytest
from pydantic import ValidationError

from plaid.infos import Infos


def _valid_infos(**overrides):
    data = {
        "owner": "owner",
        "license": "cc-by-4.0",
        "num_samples": {"train": 10},
        "storage_backend": "zarr",
    }
    data.update(overrides)
    return data


def test_verify_info_accepts_flat_owner_license_keys():
    Infos(**_valid_infos())


def test_infos_allows_draft_without_storage_derived_fields():
    model = Infos(owner="owner", license="cc-by-4.0")

    assert model.owner == "owner"
    assert model.license == "cc-by-4.0"
    assert model.num_samples == {}
    assert model.storage_backend is None


def test_verify_info_rejects_unknown_category():
    with pytest.raises(ValidationError):
        Infos(**{"unknown": {"x": "y"}})


def test_verify_info_rejects_legacy_legal_key():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Infos(**{"legal": {"owner": "owner", "license": "cc-by-4.0"}})


def test_validate_required_only_missing_required_key():
    with pytest.raises(ValueError):
        Infos(**{"owner": "someone"})


def test_validate_required_only_requires_persisted_fields():
    with pytest.raises(ValueError, match="num_samples"):
        Infos.validate_required_only({"owner": "owner", "license": "cc-by-4.0"})

    with pytest.raises(ValueError, match="storage_backend"):
        Infos.validate_required_only(
            {
                "owner": "owner",
                "license": "cc-by-4.0",
                "num_samples": {},
            }
        )


def test_normalize_infos_rejects_legacy_plaid_section_and_copies():
    infos = {"owner": "owner", "license": "cc-by-4.0", "plaid": {"version": "x"}}

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Infos.normalize_mapping(infos)

    # The input mapping is not mutated before validation raises.
    assert "plaid" in infos


def test_model_validate_rejects_plaid_section():
    with pytest.raises(ValidationError):
        Infos.model_validate(
            {
                "owner": "owner",
                "license": "cc-by-4.0",
                "plaid": {"version": "x"},
            }
        )


def test_dataset_info_model_validate_success():
    model = Infos.model_validate(_valid_infos())

    assert model.owner == "owner"
    assert model.license == "cc-by-4.0"
    assert model.storage_backend == "zarr"


def test_dataset_info_model_validate_rejects_extra_top_level_key():
    with pytest.raises(ValueError):
        Infos.model_validate({"owner": "owner", "license": "cc-by-4.0", "unknown": {}})


def test_infos_save_and_load_roundtrip(tmp_path):
    model = Infos.model_validate(_valid_infos())

    target = tmp_path / "infos.yaml"
    model.save_to_file(target)
    assert target.is_file()

    reloaded = Infos.from_path(target)
    assert reloaded.owner == "owner"
    assert reloaded.license == "cc-by-4.0"
    assert reloaded.storage_backend == "zarr"
    assert reloaded.num_samples == {"train": 10}


def test_infos_from_path_directory(tmp_path):
    Infos.model_validate(_valid_infos()).save_to_file(tmp_path / "infos.yaml")
    reloaded = Infos.from_path(tmp_path)
    assert reloaded.license == "cc-by-4.0"


def test_infos_from_path_requires_persisted_fields_by_default(tmp_path):
    Infos(owner="o", license="l").save_to_file(tmp_path)

    with pytest.raises(ValueError, match="num_samples"):
        Infos.from_path(tmp_path)


def test_infos_from_path_can_load_draft_infos(tmp_path):
    Infos(owner="o", license="l").save_to_file(tmp_path)

    reloaded = Infos.from_path(tmp_path, require_persisted=False)

    assert reloaded.owner == "o"
    assert reloaded.license == "l"
    assert reloaded.num_samples == {}
    assert reloaded.storage_backend is None


def test_validate_authorized_only_allows_missing_owner_license():
    model = Infos.validate_authorized_only(
        {"num_samples": {"train": 1}, "storage_backend": "zarr"}
    )
    # Missing user-authored required fields are filled with empty placeholder values.
    assert model.owner == ""
    assert model.license == ""
    assert model.storage_backend == "zarr"


def test_validate_authorized_only_with_owner_license_present():
    model = Infos.validate_authorized_only({"owner": "o", "license": "l"})
    assert model.owner == "o"
    assert model.license == "l"


def test_validate_authorized_only_rejects_unauthorized_key():
    with pytest.raises(KeyError):
        Infos.validate_authorized_only({"owner": "o", "license": "l", "unknown": {}})


def test_validate_authorized_only_reraises_other_validation_errors():
    # ``num_samples`` expects dict[str, int] - giving a string triggers a
    # validation error that is *not* of the unauthorized-key kind, so it
    # must be re-raised as ValidationError.
    with pytest.raises(ValidationError):
        Infos.validate_authorized_only(
            {"owner": "o", "license": "l", "num_samples": "nope"}
        )


def test_validate_required_only_accepts_valid_mapping():
    Infos.validate_required_only(
        {
            "owner": "o",
            "license": "l",
            "num_samples": {"train": 1},
            "storage_backend": "zarr",
        }
    )


def test_validate_required_only_missing_owner_license():
    with pytest.raises(ValidationError):
        Infos.validate_required_only({})


def test_model_dump_returns_plain_mapping():
    model = Infos.model_validate(
        {
            "owner": "o",
            "license": "l",
            "num_samples": {},
            "storage_backend": "zarr",
        }
    )
    d = model.model_dump(exclude_none=True)
    assert d["owner"] == "o"
    assert d["license"] == "l"
    assert d["storage_backend"] == "zarr"


def test_attribute_access_returns_typed_values():
    model = Infos.model_validate(
        {
            "owner": "o",
            "license": "l",
            "num_samples": {},
            "storage_backend": "zarr",
        }
    )
    assert model.storage_backend == "zarr"
    assert model.owner == "o"
    assert model.license == "l"


def test_save_to_file_treats_suffixless_path_as_directory(tmp_path):
    target = tmp_path / "myinfos"
    Infos(owner="o", license="l", num_samples={}, storage_backend="zarr").save_to_file(
        target
    )
    # Suffix-less, non-existing paths are treated as directories that
    # will hold an ``infos.yaml``.
    assert (target / "infos.yaml").is_file()


def test_save_to_file_into_existing_directory(tmp_path):
    Infos(owner="o", license="l", num_samples={}, storage_backend="zarr").save_to_file(
        tmp_path
    )
    assert (tmp_path / "infos.yaml").is_file()


def test_save_to_file_replaces_non_yaml_suffix(tmp_path):
    target = tmp_path / "weird.txt"
    Infos(owner="o", license="l", num_samples={}, storage_backend="zarr").save_to_file(
        target
    )
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
    model = Infos.model_validate(
        {
            "owner": "o",
            "license": "l",
            "num_samples": {},
            "storage_backend": "zarr",
        }
    )
    target = tmp_path / "out.yaml"
    model.save_to_file(target)
    text = target.read_text(encoding="utf-8")
    assert "storage_backend" in text
    assert "owner" in text
    assert "license" in text


def test_from_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Infos.from_path(tmp_path / "missing.yaml")
