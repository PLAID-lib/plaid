from typing import Any

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


def test_infos_accepts_data_production_mapping_constructor():
    kwargs: Any = {
        "owner": "owner",
        "license": "cc-by-4.0",
        "data_production": {
            "type": "simulation",
            "physics": "fluid dynamics",
            "simulator": "ExampleSolver",
        },
    }
    model = Infos(**kwargs)

    assert model.data_production is not None
    assert model.data_production.type == "simulation"
    assert model.data_production.physics == "fluid dynamics"
    assert model.data_production.simulator == "ExampleSolver"


def test_infos_data_production_mapping_constructor_preserves_nested_serialization():
    kwargs: Any = {
        "owner": "owner",
        "license": "cc-by-4.0",
        "data_production": {"type": "simulation"},
    }
    model = Infos(**kwargs)

    assert model.model_dump(exclude_none=True) == {
        "owner": "owner",
        "license": "cc-by-4.0",
        "data_production": {"type": "simulation"},
        "num_samples": {},
    }


def test_infos_print_available_fields(capsys):
    Infos.print_available_fields()

    assert capsys.readouterr().out.splitlines() == [
        "Infos fields:",
        "  - owner",
        "  - license",
        "  - data_production",
        "    subfields:",
        "      - type",
        "      - physics",
        "      - simulator",
        "      - hardware",
        "      - computation_duration",
        "      - script",
        "      - contact",
        "  - data_description",
        "  - num_samples",
        "    note: automatically filled when calling save_to_disk",
        "  - storage_backend",
        "    note: automatically filled when calling save_to_disk",
    ]


def test_verify_info_rejects_unknown_category():
    kwargs: Any = {"unknown": {"x": "y"}}

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Infos(**kwargs)


def test_verify_info_rejects_legacy_legal_key():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Infos.model_validate({"legal": {"owner": "owner", "license": "cc-by-4.0"}})


def test_infos_constructor_missing_required_license():
    kwargs: Any = {"owner": "someone"}

    with pytest.raises(ValidationError):
        Infos(**kwargs)


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
    with pytest.raises(ValidationError, match="extra_forbidden"):
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

    Infos.from_path(tmp_path / "infos")
    reloaded = Infos.from_path(target)
    assert reloaded.owner == "owner"
    assert reloaded.license == "cc-by-4.0"
    assert reloaded.storage_backend == "zarr"
    assert reloaded.num_samples == {"train": 10}


def test_infos_from_path_rejects_directory(tmp_path):
    Infos.model_validate(_valid_infos()).save_to_file(tmp_path / "infos.yaml")

    with pytest.raises(IsADirectoryError, match="Expected a YAML file path"):
        Infos.from_path(tmp_path)


def test_infos_from_path_requires_persisted_fields_by_default(tmp_path):
    (tmp_path / "infos.yaml").write_text("owner: o\nlicense: l\n", encoding="utf-8")

    with pytest.raises(ValueError, match="num_samples"):
        Infos.from_path(tmp_path / "infos.yaml")


def test_infos_from_path_can_load_draft_infos(tmp_path):
    (tmp_path / "infos.yaml").write_text("owner: o\nlicense: l\n", encoding="utf-8")

    reloaded = Infos.from_path(tmp_path / "infos.yaml", require_persisted=False)

    assert reloaded.owner == "o"
    assert reloaded.license == "l"
    assert reloaded.num_samples == {}
    assert reloaded.storage_backend is None


def test_infos_save_to_file_requires_persisted_fields(tmp_path):
    with pytest.raises(ValueError, match="num_samples"):
        Infos(owner="o", license="l").save_to_file(tmp_path)


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


def test_save_to_file_treats_suffixless_path_as_file_stem(tmp_path):
    target = tmp_path / "myinfos"
    Infos(owner="o", license="l", num_samples={}, storage_backend="zarr").save_to_file(
        target
    )
    assert target.with_suffix(".yaml").is_file()
    assert not target.exists()


def test_save_to_file_rejects_existing_directory(tmp_path):
    with pytest.raises(IsADirectoryError, match="Expected a YAML file path"):
        Infos(
            owner="o", license="l", num_samples={}, storage_backend="zarr"
        ).save_to_file(tmp_path)


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
