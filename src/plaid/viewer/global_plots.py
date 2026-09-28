"""Prepare scalar Global values and build interactive exploration plots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative


def scalar_value(value: object) -> float | None:
    """Return a finite numeric scalar, or ``None`` for unplottable values.

    Args:
        value: Raw Global value, possibly a one-element array.

    Returns:
        A finite float if the value contains exactly one number, else ``None``.
    """
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.size != 1 or array.dtype.kind not in "iuf":
            return None
        number = float(array.reshape(-1)[0])
        return number if np.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _component_values(
    values: dict[str, object],
) -> dict[str, object]:
    """Convert scalar and numeric-vector Globals to named scalar components.

    Args:
        values: Raw Global values for one sample.

    Returns:
        Scalar values keyed by Global name, with vectors expanded using 1-based
        suffixes. Invalid or non-finite components are represented by ``None``.
    """
    components: dict[str, float | None] = {}
    for name, value in values.items():
        if name.endswith("_times"):
            continue
        try:
            array = np.asarray(value)
            if array.size == 1 and array.dtype.kind not in "iuf":
                components[name] = value  # Preserve text Globals for point labels.
                continue
            if array.dtype.kind not in "iuf" or array.size == 0:
                continue
            flat = array.reshape(-1)
            if flat.size == 1:
                components[name] = scalar_value(flat[0])
            else:
                components.update(
                    {
                        f"{name}_{index}": scalar_value(element)
                        for index, element in enumerate(flat, start=1)
                    }
                )
        except (TypeError, ValueError, OverflowError):
            continue
    return components


def _expanded_records(
    records: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    """Expand vector Globals to scalar components in copied sample records.

    Args:
        records: Raw extracted records, left unmodified.

    Returns:
        Records with plottable scalar Global values and 1-based component names.
    """
    return {
        split: [
            {
                **row,
                "values": _component_values(row["values"]),
            }
            for row in rows
        ]
        for split, rows in records.items()
    }


def global_names(records: dict[str, list[dict[str, object]]]) -> list[str]:
    """List Global paths with at least one plottable value.

    Args:
        records: Extracted split records.

    Returns:
        Sorted names of numeric scalar Globals (excluding time companions).
    """
    expanded = _expanded_records(records)
    return sorted(
        {
            name
            for rows in expanded.values()
            for row in rows
            for name, value in row["values"].items()
            if scalar_value(value) is not None
        }
    )


def label_names(records: dict[str, list[dict[str, object]]]) -> list[str]:
    """List all extracted Global paths usable as point labels.

    Args:
        records: Extracted split records.

    Returns:
        Sorted paths, including non-numeric values but excluding time arrays.
    """
    expanded = _expanded_records(records)
    return sorted(
        {name for rows in expanded.values() for row in rows for name in row["values"]}
    )


def format_point_label(value: object) -> str:
    """Format a Global annotation with three decimal places for floats.

    Args:
        value: A scalar or array-like Global used as a point label.

    Returns:
        The readable label, without brackets for a one-element vector.
    """
    array = np.asarray(value)
    if array.size == 1:
        element = array.reshape(-1)[0]
        if isinstance(element, (float, np.floating)):
            return f"{element:.3f}"
        return str(element)
    if array.ndim and array.dtype.kind == "f":
        return np.array2string(
            array, formatter={"float_kind": lambda number: f"{number:.3f}"}
        )
    return str(value)


def _parallel_values(
    numbers: list[float | None], ranges: list[tuple[float, float] | None]
) -> list[float | None]:
    """Normalize parallel values and leave gaps where a Global is absent.

    Args:
        numbers: Scalar values in selected axis order, including missing values.
        ranges: Minimum and maximum for each axis, or ``None`` if absent.

    Returns:
        Values normalized to 0–1, with ``None`` for unavailable axes.
    """
    scaled = []
    for number, bounds in zip(numbers, ranges):
        if number is None or bounds is None:
            scaled.append(None)
        elif bounds[1] > bounds[0]:
            scaled.append((number - bounds[0]) / (bounds[1] - bounds[0]))
        else:
            scaled.append(0.5)
    return scaled


def build_globals_figure(
    records: dict[str, list[dict[str, object]]],
    splits: list[str],
    kind: str,
    axes: list[str],
    label: str | None = None,
    parallel_fields: list[str] | None = None,
    parallel_mode: str = "lines",
) -> go.Figure | None:
    """Build an interactive Plotly scatter or parallel-coordinates figure.

    Args:
        records: Raw Global values keyed by split.
        splits: Selected split names in display order.
        kind: ``1D``, ``2D``, ``3D`` or ``parallel``.
        axes: Selected Global paths for scatter axes.
        label: Optional Global path for point annotations; ``sample_id`` is
            supported as a special label.
        parallel_fields: Selected parallel axes, or ``None`` for all axes.
        parallel_mode: Parallel renderer, either ``lines`` or ``parcoords``.

    Returns:
        A Plotly figure, or ``None`` if there is nothing to plot.
    """
    records = _expanded_records(records)
    names = global_names(records)
    dimensions = {"1D": 1, "2D": 2, "3D": 3, "parallel": len(names)}
    if kind not in dimensions:
        raise ValueError(f"Unknown plot type: {kind}")
    if parallel_mode not in {"lines", "parcoords"}:
        raise ValueError(f"Unknown parallel plot mode: {parallel_mode}")
    if kind == "parallel":
        fields = names if parallel_fields is None else parallel_fields
    else:
        fields = axes[: dimensions[kind]]
    if (
        not splits
        or (kind != "parallel" and len(fields) != dimensions[kind])
        or not fields
        or any(field not in names for field in fields)
    ):
        return None

    ranges: list[tuple[float, float] | None] = []
    if kind == "parallel":
        for field in fields:
            present = [
                number
                for split in splits
                for row in records.get(split, [])
                if (number := scalar_value(row["values"].get(field))) is not None
            ]
            ranges.append((min(present), max(present)) if present else None)

    figure = go.Figure()
    dashes = ["solid", "dash", "dot", "dashdot"]
    symbols = ["circle", "square", "diamond", "cross", "x"]
    if kind == "parallel" and parallel_mode == "parcoords":
        dimensions_data = [[] for _ in fields]
        sample_ids = []
        split_values = []
        plotted_splits = []
        for split in splits:
            split_rows = []
            for row in records.get(split, []):
                numbers = [scalar_value(row["values"].get(field)) for field in fields]
                if any(number is None for number in numbers):
                    continue
                split_rows.append((row, numbers))
            if not split_rows:
                continue
            split_index = len(plotted_splits)
            plotted_splits.append(split)
            for row, numbers in split_rows:
                sample_ids.append(str(row["sample_id"]))
                split_values.append(split_index)
                for dimension, number in zip(dimensions_data, numbers):
                    dimension.append(number)
        if not sample_ids:
            return None

        trace_dimensions = [
            {
                "label": field.removeprefix("Global/"),
                "values": values,
                "range": list(axis_range) if axis_range is not None else None,
            }
            for field, values, axis_range in zip(fields, dimensions_data, ranges)
        ]
        colors = [
            qualitative.Plotly[i % len(qualitative.Plotly)]
            for i in range(len(plotted_splits))
        ]
        color_max = max(len(colors) - 1, 1)
        colorscale = [[0, colors[0]]]
        for i, color in enumerate(colors[1:], start=1):
            boundary = (i - 0.5) / color_max
            colorscale.extend([[boundary, colors[i - 1]], [boundary, color]])
        colorscale.append([1, colors[-1]])
        figure.add_trace(
            go.Parcoords(
                dimensions=trace_dimensions,
                customdata=sample_ids,
                line={
                    "color": split_values,
                    "cmin": 0,
                    "cmax": color_max,
                    "colorscale": colorscale,
                    "showscale": len(plotted_splits) > 1,
                    "colorbar": {
                        "title": "Split",
                        "tickmode": "array",
                        "tickvals": list(range(len(plotted_splits))),
                        "ticktext": plotted_splits,
                    },
                },
            )
        )
    for split_index, split in enumerate(splits):
        color = qualitative.Plotly[split_index % len(qualitative.Plotly)]
        rows = records.get(split, [])
        if kind == "parallel":
            if parallel_mode == "parcoords":
                continue
            plotted = False
            for row in rows:
                numbers = [scalar_value(row["values"].get(f)) for f in fields]
                if all(number is None for number in numbers):
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=list(range(len(fields))),
                        y=_parallel_values(numbers, ranges),
                        mode="lines+markers",
                        name=split,
                        legendgroup=split,
                        showlegend=not plotted,
                        connectgaps=False,
                        line={"color": color, "dash": dashes[split_index % 4]},
                        marker={"color": color, "size": 5},
                        opacity=0.65,
                        customdata=[
                            [str(row["sample_id"]), number] for number in numbers
                        ],
                        hovertemplate=(
                            "Sample %{customdata[0]}<br>"
                            "Value: %{customdata[1]}<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
                plotted = True
        else:
            points = []
            for row in rows:
                numbers = [scalar_value(row["values"].get(f)) for f in fields]
                if any(number is None for number in numbers):
                    continue
                value = (
                    row["sample_id"]
                    if label == "sample_id"
                    else row["values"].get(label)
                    if label
                    else None
                )
                points.append((row, numbers, value))
            if not points:
                continue
            x = [
                int(row["sample_id"]) if kind == "1D" else nums[0]
                for row, nums, _ in points
            ]
            y = [nums[0] if kind == "1D" else nums[1] for _, nums, _ in points]
            text = [
                format_point_label(value) if value is not None else ""
                for _, _, value in points
            ]
            props = {
                "x": x,
                "y": y,
                "name": split,
                "mode": "markers+text" if label else "markers",
                "text": text if label else None,
                "textposition": "top center",
                "marker": {
                    "color": color,
                    "symbol": symbols[split_index % len(symbols)],
                    "size": 8,
                },
                "customdata": [str(row["sample_id"]) for row, _, _ in points],
            }
            if kind == "3D":
                props["z"] = [nums[2] for _, nums, _ in points]
                figure.add_trace(go.Scatter3d(**props))
            else:
                figure.add_trace(go.Scatter(**props))

    if kind == "parallel":
        figure.update_layout(
            xaxis={
                "tickmode": "array",
                "tickvals": list(range(len(fields))),
                "ticktext": [name.removeprefix("Global/") for name in fields],
            },
            yaxis={"title": "Normalized Global value (per axis)", "range": [0, 1]},
        )
    elif kind == "3D":
        figure.update_layout(
            scene={
                "xaxis_title": fields[0],
                "yaxis_title": fields[1],
                "zaxis_title": fields[2],
            }
        )
    else:
        figure.update_layout(
            xaxis_title="Sample id" if kind == "1D" else fields[0],
            yaxis_title=fields[0] if kind == "1D" else fields[1],
        )
    figure.update_layout(
        autosize=True,
        height=None,
        showlegend=True,
        margin={"l": 65, "r": 20, "t": 35, "b": 90},
    )
    return figure
