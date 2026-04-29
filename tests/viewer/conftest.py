"""Shared fixtures for viewer tests.

The viewer persists user preferences (currently the last-used datasets
root) to ``$XDG_CONFIG_HOME/plaid/viewer.json``. Tests that exercise
:meth:`PlaidDatasetService.set_datasets_root` would otherwise mutate the
real user preferences file, polluting interactive sessions with a path
from ``tmp_path``. We redirect preference persistence to a temporary
location for every viewer test through the
``PLAID_VIEWER_CONFIG_FILE`` environment variable honoured by
:mod:`plaid.viewer.preferences`.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_viewer_preferences(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect viewer preference persistence to a unique temporary file."""
    prefs_file = tmp_path_factory.mktemp("viewer_prefs") / "viewer.json"
    monkeypatch.setenv("PLAID_VIEWER_CONFIG_FILE", str(prefs_file))
    return prefs_file
