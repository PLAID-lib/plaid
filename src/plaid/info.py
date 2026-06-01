"""Pydantic models and helpers for dataset ``infos`` metadata."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.dataclasses import dataclass

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
    location: str | None = None


@dataclass(config=_PD_CONFIG)
class DataDescription:
    """Dataset shape/DOE metadata."""

    number_of_samples: int | None = None
    number_of_splits: int | None = None
    DOE: str | None = None
    inputs: str | None = None
    outputs: str | None = None


class Info(BaseModel):
    """Structured representation of a PLAID dataset ``infos`` payload."""

    model_config = _PD_CONFIG

    legal: Legal
    data_production: DataProduction | None = None
    data_description: DataDescription | None = None
    plaid: dict[str, Any] = Field(default_factory=dict)
    num_samples: dict[str, int] = Field(default_factory=dict)
    storage_backend: str | None = None

    @classmethod
    def _normalize_top_level(cls, infos: dict[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(infos)
        normalized.setdefault("plaid", {})
        return normalized

    @classmethod
    def validate_authorized_only(cls, infos: dict[str, Any]) -> "Info":
        """Validate schema/authorized keys without enforcing required sections."""
        normalized = cls._normalize_top_level(infos)
        if "legal" not in normalized:
            normalized["legal"] = {
                "owner": "__placeholder__",
                "license": "__placeholder__",
            }
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

        if "legal" not in infos:
            model.legal = Legal(owner="", license="")
        return model

    @classmethod
    def validate_required_only(cls, infos: dict[str, Any]) -> None:
        """Validate required entries using pydantic-required fields."""
        normalized = cls._normalize_top_level(infos)
        cls.model_validate(normalized)

    @classmethod
    def normalize_mapping(cls, infos: dict[str, Any]) -> dict[str, Any]:
        """Validate and return a normalized deep copy of infos."""
        normalized = cls._normalize_top_level(infos)
        model = cls.model_validate(normalized)
        return model.model_dump(exclude_none=True)
