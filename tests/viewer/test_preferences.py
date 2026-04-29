"""Tests for viewer preference persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plaid.viewer import preferences as prefs


def test_preferences_path_uses_xdg_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PLAID_VIEWER_CONFIG_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert prefs._preferences_path() == tmp_path / "plaid" / "viewer.json"


def test_load_preferences_handles_missing_invalid_and_valid_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "viewer.json"
    monkeypatch.setenv("PLAID_VIEWER_CONFIG_FILE", str(path))
    assert prefs.load_preferences() == {}
    path.write_text("not json")
    assert prefs.load_preferences() == {}
    path.write_text(json.dumps({"datasets_root": str(tmp_path)}))
    assert prefs.load_preferences() == {"datasets_root": str(tmp_path)}


def test_update_and_last_datasets_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "viewer.json"
    monkeypatch.setenv("PLAID_VIEWER_CONFIG_FILE", str(path))

    prefs.save_preferences({"datasets_root": str(tmp_path), "other": 1})
    assert prefs.get_last_datasets_root() == tmp_path
    updated = prefs.update_preferences(datasets_root=None)
    assert updated == {"other": 1}
    assert prefs.get_last_datasets_root() is None
    prefs.set_last_datasets_root(tmp_path)
    assert prefs.get_last_datasets_root() == tmp_path.resolve()
    prefs.set_last_datasets_root(None)
    assert prefs.get_last_datasets_root() is None


def test_get_last_datasets_root_rejects_bad_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "viewer.json"
    monkeypatch.setenv("PLAID_VIEWER_CONFIG_FILE", str(path))
    path.write_text(json.dumps({"datasets_root": ""}))
    assert prefs.get_last_datasets_root() is None
    path.write_text(json.dumps({"datasets_root": str(tmp_path / "missing")}))
    assert prefs.get_last_datasets_root() is None


def test_save_preferences_ignores_os_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadPath:
        parent = Path("/")

        def write_text(self, _text: str) -> None:
            raise OSError("boom")

    monkeypatch.setattr(prefs, "_preferences_path", lambda: BadPath())
    prefs.save_preferences({"x": 1})
