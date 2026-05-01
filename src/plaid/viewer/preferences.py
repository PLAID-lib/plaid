"""Persistent user preferences for the dataset viewer.

The viewer stores a tiny JSON document under the OS-standard user config
directory so a handful of settings (currently only the last local
``datasets_root``) survive across sessions. The file is best-effort:
read/write errors are silently swallowed so a broken preferences file
never prevents the viewer from starting.

Location: ``$XDG_CONFIG_HOME/plaid/viewer.json`` (falling back to
``~/.config/plaid/viewer.json``), overridable by setting
``PLAID_VIEWER_CONFIG_FILE``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _preferences_path() -> Path:
    """Return the path to the persistent preferences file."""
    override = os.environ.get("PLAID_VIEWER_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "plaid" / "viewer.json"


def load_preferences() -> dict[str, object]:
    """Return the persisted preferences dict, or an empty dict on failure."""
    path = _preferences_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        logger.debug("Ignoring unreadable viewer preferences at %s: %s", path, exc)
        return {}


def save_preferences(data: dict[str, object]) -> None:
    """Persist ``data`` to the preferences file, creating parents as needed."""
    path = _preferences_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError as exc:  # noqa: BLE001
        logger.debug("Failed to persist viewer preferences to %s: %s", path, exc)


def update_preferences(**updates: object) -> dict[str, object]:
    """Merge ``updates`` into the persisted preferences and return the result.

    Keys whose value is ``None`` are removed from the stored document so
    clearing a setting (e.g. the datasets root) does not leave a stale
    entry behind.
    """
    current = load_preferences()
    for key, value in updates.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    save_preferences(current)
    return current


def get_last_datasets_root() -> Path | None:
    """Return the persisted last-used datasets root, or ``None``."""
    value = load_preferences().get("datasets_root")
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_dir() else None


def set_last_datasets_root(path: Path | str | None) -> None:
    """Persist (or clear) the last-used datasets root."""
    if path is None:
        update_preferences(datasets_root=None)
        return
    update_preferences(datasets_root=str(Path(path).expanduser().resolve()))
