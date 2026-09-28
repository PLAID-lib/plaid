"""Tests for Global selection and interactive exploration plots."""

import json

import numpy as np
import pytest
from plotly.utils import PlotlyJSONEncoder
from trame.widgets.plotly import Figure as TrameFigure

from plaid.viewer.global_plots import (
    _component_values,
    _parallel_values,
    build_globals_figure,
    format_point_label,
    global_names,
    label_names,
    scalar_value,
)


@pytest.fixture
def records():
    """Return two small splits with missing and vector values."""
    return {
        "train": [
            {
                "sample_id": "0",
                "values": {
                    "Global/a": [1.0],
                    "Global/b": 2.0,
                    "Global/c": 3.0,
                    "Global/vector": [1, 2],
                },
            },
            {
                "sample_id": "1",
                "values": {"Global/a": np.nan, "Global/b": 4.0, "Global/c": 5.0},
            },
        ],
        "test": [
            {
                "sample_id": "0",
                "values": {"Global/a": 6.0, "Global/b": 7.0, "Global/c": 8.0},
            },
        ],
    }


def test_scalar_value_filters_missing_and_non_scalars():
    assert scalar_value([3]) == 3.0
    assert scalar_value(np.array(2)) == 2.0
    assert scalar_value([1, 2]) is None
    assert scalar_value(float("inf")) is None
    assert scalar_value(None) is None
    assert scalar_value("text") is None


def test_names_include_only_plot_eligible_globals(records):
    assert global_names(records) == [
        "Global/a",
        "Global/b",
        "Global/c",
        "Global/vector_1",
        "Global/vector_2",
    ]
    assert label_names(records) == [
        "Global/a",
        "Global/b",
        "Global/c",
        "Global/vector_1",
        "Global/vector_2",
    ]


def test_array_globals_expand_to_one_based_scalar_components():
    records = {
        "train": [
            {
                "sample_id": "0",
                "values": {
                    "Global/vector": np.array([1.5, 2.5, 3.5]),
                    "Global/scalar_array": np.array([4.5]),
                    "Global/text": "case-a",
                },
            }
        ],
        "test": [
            {
                "sample_id": "0",
                "values": {"Global/vector": np.array([6.5, 7.5])},
            }
        ],
    }
    assert global_names(records) == [
        "Global/scalar_array",
        "Global/vector_1",
        "Global/vector_2",
        "Global/vector_3",
    ]
    assert label_names(records) == [
        "Global/scalar_array",
        "Global/text",
        "Global/vector_1",
        "Global/vector_2",
        "Global/vector_3",
    ]
    components = _component_values(records["train"][0]["values"])
    assert components == {
        "Global/vector_1": 1.5,
        "Global/vector_2": 2.5,
        "Global/vector_3": 3.5,
        "Global/scalar_array": 4.5,
        "Global/text": "case-a",
    }


def test_plot_uses_selected_array_component_and_parcoords_expands_components():
    records = {
        "train": [
            {
                "sample_id": "0",
                "values": {"Global/vector": np.array([1.0, 2.0, 3.0])},
            },
            {
                "sample_id": "1",
                "values": {"Global/vector": np.array([4.0, 5.0, 6.0])},
            },
        ]
    }
    figure = build_globals_figure(records, ["train"], "1D", ["Global/vector_2"])
    assert figure is not None
    assert figure.data[0].y == (2.0, 5.0)
    parallel = build_globals_figure(
        records,
        ["train"],
        "parallel",
        [],
        parallel_fields=["Global/vector_1", "Global/vector_3"],
        parallel_mode="parcoords",
    )
    assert parallel is not None
    assert [dimension.label for dimension in parallel.data[0].dimensions] == [
        "vector_1",
        "vector_3",
    ]
    assert parallel.data[0].dimensions[1].values == (3.0, 6.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1.234567890123456], "1.235"),
        (np.array([1.23456]), "1.235"),
        (np.array([[2.5]]), "2.500"),
        (3.1415926535, "3.142"),
        ([4], "4"),
        (["name"], "name"),
        ("plain text", "plain text"),
        (np.array([1.23456, 7.89123]), "[1.235 7.891]"),
    ],
)
def test_format_point_label(value, expected):
    assert format_point_label(value) == expected


@pytest.mark.parametrize("kind", ["1D", "2D", "3D", "parallel"])
def test_every_plot_serializes_for_trame(records, kind):
    figure = build_globals_figure(
        records,
        ["train", "test"],
        kind,
        ["Global/a", "Global/b", "Global/c"],
        "sample_id",
    )
    assert figure is not None
    assert [trace.name for trace in figure.data if trace.showlegend is not False] == [
        "train",
        "test",
    ]
    assert figure.data[0].type == ("scatter3d" if kind == "3D" else "scatter")
    assert figure.layout.autosize is True
    assert figure.layout.height is None
    assert figure.layout.showlegend is True
    data = TrameFigure.to_data(figure)
    assert data["data"]
    json.dumps(data, cls=PlotlyJSONEncoder, allow_nan=False)


def test_scatter_coordinates_and_formatted_labels(records):
    records["train"][0]["values"]["Global/b"] = [2.123456789012345]
    one_d = build_globals_figure(records, ["train"], "1D", ["Global/a"], "Global/b")
    assert one_d.data[0].x == (0,)
    assert one_d.data[0].y == (1.0,)
    assert one_d.data[0].text == ("2.123",)
    assert one_d.layout.xaxis.title.text == "Sample id"
    two_d = build_globals_figure(records, ["train"], "2D", ["Global/a", "Global/b"])
    assert two_d.data[0].x == (1.0,)
    assert two_d.data[0].y == (2.123456789012345,)
    three_d = build_globals_figure(
        records, ["train"], "3D", ["Global/a", "Global/b", "Global/c"]
    )
    assert three_d.data[0].z == (3.0,)
    assert three_d.layout.scene.zaxis.title.text == "Global/c"


@pytest.mark.parametrize("kind", ["1D", "2D", "3D", "parallel"])
def test_single_selected_split_has_a_visible_legend(records, kind):
    figure = build_globals_figure(
        records,
        ["train"],
        kind,
        ["Global/a", "Global/b", "Global/c"],
        parallel_fields=["Global/a", "Global/b", "Global/c"],
    )
    assert figure is not None
    assert figure.layout.showlegend is True
    assert [trace.name for trace in figure.data if trace.showlegend is not False] == [
        "train"
    ]


@pytest.mark.parametrize("fields", [["Global/b"], ["Global/c", "Global/a"]])
def test_parallel_plot_shows_only_checked_globals(records, fields):
    figure = build_globals_figure(
        records, ["train", "test"], "parallel", [], parallel_fields=fields
    )
    assert figure is not None
    assert figure.layout.xaxis.ticktext == tuple(
        f.removeprefix("Global/") for f in fields
    )
    assert figure.data[0].x == tuple(range(len(fields)))


def test_parallel_plot_preserves_partial_samples_from_different_splits():
    records = {
        "train": [{"sample_id": "0", "values": {"Global/a": 1.0, "Global/c": 3.0}}],
        "test": [{"sample_id": "0", "values": {"Global/b": 2.0}}],
    }
    figure = build_globals_figure(
        records,
        ["train", "test"],
        "parallel",
        [],
        parallel_fields=["Global/a", "Global/b", "Global/c"],
    )
    assert figure is not None
    assert [trace.y for trace in figure.data] == [
        (0.5, None, 0.5),
        (None, 0.5, None),
    ]
    assert all(trace.connectgaps is False for trace in figure.data)
    assert figure.data[0].customdata == (
        ["0", 1.0],
        ["0", None],
        ["0", 3.0],
    )
    json.dumps(TrameFigure.to_data(figure), cls=PlotlyJSONEncoder, allow_nan=False)


def test_parcoords_renders_complete_samples_with_selected_axes():
    records = {
        "train": [
            {
                "sample_id": "0",
                "values": {"Global/a": 1.0, "Global/b": 2.0, "Global/c": 3.0},
            },
            {"sample_id": "1", "values": {"Global/a": 4.0, "Global/b": None}},
        ],
        "test": [
            {"sample_id": "0", "values": {"Global/a": 6.0, "Global/c": 8.0}},
        ],
    }
    figure = build_globals_figure(
        records,
        ["train", "test"],
        "parallel",
        [],
        parallel_fields=["Global/a", "Global/c"],
        parallel_mode="parcoords",
    )
    assert figure is not None
    assert [trace.type for trace in figure.data] == ["parcoords"]
    trace = figure.data[0]
    assert trace.dimensions[0].label == "a"
    assert trace.dimensions[0].values == (1.0, 6.0)
    assert trace.dimensions[1].range == (3.0, 8.0)
    assert trace.customdata == ("0", "0")
    assert trace.line.color == (0, 1)
    assert trace.line.colorbar.ticktext == ("train", "test")
    assert trace.line.showscale is True
    json.dumps(TrameFigure.to_data(figure), cls=PlotlyJSONEncoder, allow_nan=False)


def test_parcoords_skips_split_without_complete_selected_values():
    records = {
        "train": [{"sample_id": "0", "values": {"Global/a": 1.0}}],
        "test": [
            {"sample_id": "0", "values": {"Global/a": 2.0, "Global/b": 3.0}},
        ],
    }
    figure = build_globals_figure(
        records,
        ["train", "test"],
        "parallel",
        [],
        parallel_fields=["Global/a", "Global/b"],
        parallel_mode="parcoords",
    )
    assert figure is not None
    assert len(figure.data) == 1
    assert figure.data[0].type == "parcoords"
    assert figure.data[0].dimensions[0].values == (2.0,)
    assert figure.data[0].line.colorbar.ticktext == ("test",)
    assert figure.data[0].line.showscale is False


def test_parallel_renderer_rejects_unknown_mode(records):
    with pytest.raises(ValueError, match="Unknown parallel plot mode"):
        build_globals_figure(
            records,
            ["train"],
            "parallel",
            [],
            parallel_mode="invalid",
        )


def test_parallel_plot_keeps_axis_missing_from_selected_splits(records):
    records["test"][0]["values"]["Global/only_test"] = 9.0
    figure = build_globals_figure(
        records,
        ["train"],
        "parallel",
        [],
        parallel_fields=["Global/a", "Global/only_test", "Global/b"],
    )
    assert figure is not None
    assert figure.layout.xaxis.ticktext == ("a", "only_test", "b")
    assert all(trace.y[1] is None for trace in figure.data)


def test_parallel_plot_skips_samples_without_selected_values(records):
    records["train"].append({"sample_id": "2", "values": {"Global/vector": [1, 2]}})
    figure = build_globals_figure(
        records, ["train"], "parallel", [], parallel_fields=["Global/a"]
    )
    assert figure is not None
    assert len(figure.data) == 1
    assert figure.data[0].y == (0.5,)
    assert figure.data[0].showlegend is True


def test_plotly_widget_updates_3d_figure_in_trame_state(records):
    from trame.app import get_server  # noqa: PLC0415
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout  # noqa: PLC0415

    server = get_server("globals_plotly_widget_test", client_type="vue3")
    with SinglePageWithDrawerLayout(server) as layout:
        with layout.content:
            widget = TrameFigure()
    figure = build_globals_figure(
        records,
        ["train"],
        "3D",
        ["Global/a", "Global/b", "Global/c"],
    )
    widget.update(figure)
    assert server.state[widget.key]["data"][0]["type"] == "scatter3d"


def test_parallel_values_have_gaps_without_interpolating_over_missing_axes():
    assert _parallel_values(
        [2.0, None, 5.0, None],
        [(0.0, 4.0), None, (0.0, 10.0), (0.0, 1.0)],
    ) == [0.5, None, 0.5, None]
    assert _parallel_values([3.0], [(3.0, 3.0)]) == [0.5]


def test_empty_selection_and_invalid_kind(records):
    assert build_globals_figure(records, [], "1D", ["Global/a"]) is None
    assert (
        build_globals_figure(records, ["train"], "2D", ["missing", "Global/b"]) is None
    )
    assert (
        build_globals_figure(records, ["train"], "parallel", [], parallel_fields=[])
        is None
    )
    assert (
        build_globals_figure(
            records, ["train"], "parallel", [], parallel_fields=["Global/not_present"]
        )
        is None
    )
    with pytest.raises(ValueError, match="Unknown plot type"):
        build_globals_figure(records, ["train"], "heatmap", [])
