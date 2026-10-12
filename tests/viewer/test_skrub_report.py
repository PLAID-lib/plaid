"""Exercise the standalone skrub report used by the Globals viewer."""

from __future__ import annotations

from pathlib import Path

from plaid.viewer.global_plots import globals_dataframe
from plaid.viewer.trame_app.server import _write_globals_report


def test_write_globals_report_from_all_extracted_splits(tmp_path: Path) -> None:
    """The browser report must contain the same rows as the full CSV export."""
    records = {
        "train": [
            {
                "sample_id": "0",
                "values": {"Global/pressure": 1.5, "Global/vector": [2.0, 3.0]},
            }
        ],
        "test": [
            {
                "sample_id": "1",
                "values": {"Global/pressure": 4.5, "Global/vector": [5.0, 6.0]},
            }
        ],
    }
    dataframe = globals_dataframe(records)
    destination = tmp_path / "globals.html"

    _write_globals_report(dataframe, destination, "Example Globals")

    content = destination.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content or "<!doctype html>" in content.lower()
    assert "Example Globals" in content
    assert "Global/vector_1" in content
    assert "Global/vector_2" in content
    assert "train" in content
    assert "test" in content
