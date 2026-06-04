"""Pydantic models and helpers for dataset ``infos`` metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

_PD_CONFIG = ConfigDict(
    extra="forbid", str_strip_whitespace=True, validate_assignment=True
)


@dataclass(config=_PD_CONFIG)
class Legal:
    """Legal ownership and licensing metadata."""

    owner: str
    license: str


@dataclass(config=_PD_CONFIG)
class DataProduction:
    """Dataset production context metadata."""

    owner: str | None = None
    license: str | None = None
    type: str | None = None
    physics: str | None = None
    simulator: str | None = None
    hardware: str | None = None
    computation_duration: str | None = None
    script: str | None = None
    contact: str | None = None


# Order used when serializing to YAML.
_KEY_ORDER = (
    "legal",
    "data_production",
    "data_description",
    "num_samples",
    "storage_backend",
)


class Infos(BaseModel):
    """Structured representation of a PLAID dataset ``infos`` payload."""

    model_config = _PD_CONFIG

    legal: Legal
    data_production: DataProduction | None = None
    data_description: str | None = None
    num_samples: dict[str, int]
    storage_backend: str

    @classmethod
    def validate_authorized_only(cls, infos: dict[str, Any]) -> "Infos":
        """Validate schema/authorized keys without enforcing required sections."""
        normalized = dict(infos)
        had_legal = "legal" in normalized
        had_num_samples = "num_samples" in normalized
        had_storage_backend = "storage_backend" in normalized
        if not had_legal:
            normalized["legal"] = {
                "owner": "__placeholder__",
                "license": "__placeholder__",
            }
        if not had_num_samples:
            normalized["num_samples"] = {}
        if not had_storage_backend:
            normalized["storage_backend"] = "__placeholder__"
        try:
            model = cls.model_validate(normalized)
        except ValidationError as exc:
            for error in exc.errors():
                if error.get("type") in {
                    "extra_forbidden",
                    "unexpected_keyword_argument",
                }:
                    loc = ".".join(str(p) for p in error.get("loc", ()))
                    raise KeyError(f"Unauthorized infos key: {loc!r}") from exc
            raise

        if not had_legal:
            model.legal = Legal(owner="", license="")
        if not had_num_samples:
            model.num_samples = {}
        if not had_storage_backend:
            model.storage_backend = ""
        return model

    @classmethod
    def validate_required_only(cls, infos: dict[str, Any]) -> None:
        """Validate required entries using pydantic-required fields."""
        cls.model_validate(infos)

    @classmethod
    def normalize_mapping(cls, infos: dict[str, Any]) -> dict[str, Any]:
        """Validate and return a normalized deep copy of infos."""
        model = cls.model_validate(infos)
        return model.model_dump(exclude_none=True)

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "Infos":
        """Load and validate an :class:`Infos` from a YAML file.

        Args:
            path: Path to the YAML file (typically ``infos.yaml``) or to a
                directory containing it.

        Returns:
            Validated :class:`Infos` instance.

        Raises:
            FileNotFoundError: If the resolved YAML file does not exist.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "infos.yaml"
        if not path.exists():
            raise FileNotFoundError(f'File "{path}" does not exist. Abort')

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        return cls.model_validate(data)

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save infos to ``path`` as a YAML file.

        Args:
            path: File path (or directory) where the YAML will be written. If
                ``path`` is a directory it will be extended with ``infos.yaml``.
        """
        path = Path(path)
        if path.suffix == "" and not path.exists():
            # Treat suffix-less paths as directories.
            path = path / "infos.yaml"
        elif path.is_dir():
            path = path / "infos.yaml"
        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")

        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(exclude_none=True)
        ordered_data = {key: data[key] for key in _KEY_ORDER if key in data}
        # Preserve any future fields.
        for key, value in data.items():
            if key not in ordered_data:
                ordered_data[key] = value

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                ordered_data,
                file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
