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


def test_report_columns_put_text_globals_first_without_truncation() -> None:
    """Reports contain exactly the selection, text Globals first."""
    from plaid.viewer.trame_app import server as srv

    columns = srv._report_columns(
        ["Global/a", "Global/case", "Global/b", "Global/c"], ["Global/case"]
    )
    assert columns == ["Global/case", "Global/a", "Global/b", "Global/c"]
    assert srv._report_columns([], ["Global/case"]) == ["Global/case"]


def test_report_blocker_refuses_too_wide_selections(monkeypatch) -> None:
    """Wide selections are refused with an explanation, never truncated."""
    from plaid.viewer.trame_app import server as srv

    monkeypatch.setattr(srv, "SKRUB_MAX_COLUMNS", 3)
    assert srv._report_blocker([], ["Global/case"]) == ""
    assert srv._report_blocker(["Global/a", "Global/b"], ["Global/case"]) == ""
    message = srv._report_blocker(["Global/a", "Global/b", "Global/c"], ["Global/case"])
    assert "4 columns selected" in message
    assert "limited to 3" in message


def test_write_globals_report_enables_plots_for_small_selections(
    tmp_path: Path, monkeypatch
) -> None:
    """Plots exceed skrub's 30-column default but stop for wide selections."""
    import skrub

    from plaid.viewer.trame_app import server as srv

    calls = []

    class _Report:
        def __init__(self, _dataframe, **kwargs):
            calls.append(kwargs)

        def write_html(self, path):
            Path(path).write_text("report")

    monkeypatch.setattr(skrub, "TableReport", _Report)
    monkeypatch.setattr(srv, "SKRUB_MAX_PLOT_COLUMNS", 2)
    monkeypatch.setattr(srv, "SKRUB_MAX_ASSOCIATION_COLUMNS", 1)
    records = {"train": [{"sample_id": "0", "values": {"Global/v": [1.0, 2.0, 3.0]}}]}
    for count in (1, 2, 3):
        columns = [f"Global/v_{i}" for i in range(1, count + 1)]
        srv._write_globals_report(
            globals_dataframe(records, columns=columns), tmp_path / "r.html", "T"
        )
    assert [(c["plot_distributions"], c["compute_associations"]) for c in calls] == [
        (True, True),
        (True, False),
        (False, False),
    ]
