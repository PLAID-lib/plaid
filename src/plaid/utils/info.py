"""Helpers for validating and normalizing dataset infos metadata."""
import copy
from typing import Any

from ..constants import AUTHORIZED_INFO_KEYS, REQUIRED_INFOS_KEYS


def verify_info(infos: dict[str, dict[str, Any]]) -> None:
    """Validate infos keys against authorized categories and entries.

    Args:
        infos: Metadata dictionary grouped by category.

    Raises:
        KeyError: If a category or an info key is not authorized.
    """
    for cat_key in infos.keys():  # Format checking on "infos"
        if cat_key not in {"plaid", "num_samples", "storage_backend"}:
            if cat_key not in AUTHORIZED_INFO_KEYS:
                raise KeyError(
                    f"{cat_key=} not among authorized keys. Maybe you want to try among these keys {list(AUTHORIZED_INFO_KEYS.keys())}"
                )
            for info_key in infos[cat_key].keys():
                if info_key not in AUTHORIZED_INFO_KEYS[cat_key]:
                    raise KeyError(
                        f"{info_key=} not among authorized keys. Maybe you want to try among these keys {AUTHORIZED_INFO_KEYS[cat_key]}"
                    )


def validate_required_infos(infos: dict[str, Any]) -> None:
    """Validate that required infos categories and keys are present.

    Args:
        infos: Dataset infos dictionary loaded from disk.

    Raises:
        ValueError: If a required infos category or key is missing.
    """
    assert isinstance(infos, dict)

    missing_entries: list[str] = []
    for category, required_keys in REQUIRED_INFOS_KEYS.items():
        category_infos = infos.get(category)
        assert isinstance(category_infos, dict)

        for key in required_keys:
            if key not in category_infos:
                missing_entries.append(f"{category}.{key}")

    if missing_entries:
        raise ValueError(
            "Missing required infos entries: "
            + ", ".join(sorted(missing_entries))
            + f". Required entries are defined by {REQUIRED_INFOS_KEYS!r}."
        )


def normalize_infos(infos: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a validated deep copy of infos with guaranteed ``plaid`` section.

    Args:
        infos: Metadata dictionary grouped by category.

    Returns:
        dict[str, dict[str, Any]]: Validated infos with a guaranteed
        ``plaid`` section.
    """
    verify_info(infos)

    normalized = copy.deepcopy(infos)
    normalized.setdefault("plaid", {})
    return normalized
