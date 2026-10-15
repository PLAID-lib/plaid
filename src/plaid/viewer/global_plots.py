"""Prepare scalar Global values and build interactive exploration plots.

Extracted Globals are stored column-wise: datasets with many vector Globals
expand to tens of thousands of scalar components, and one dictionary per
sample (or a pandas table of the whole extraction) does not fit in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

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
    except (TypeError, ValueError, OverflowError):  # pragma: no cover
        return None


def _text_value(array: np.ndarray) -> str | None:
    """Decode CGNS character arrays, or return ``None`` for other values."""
    if array.dtype.kind == "S" and array.dtype.itemsize == 1:
        return array.tobytes().decode("ascii")
    if array.dtype.kind == "U" and array.dtype.itemsize == np.dtype("U1").itemsize:
        return "".join(array.reshape(-1).tolist())
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
    components: dict[str, object] = {}
    for name, value in values.items():
        if name.endswith("_times"):  # pragma: no cover
            continue
        try:
            array = np.asarray(value)
            text = _text_value(array)
            if text is not None:
                components[name] = text
                continue
            if array.size == 1 and array.dtype.kind not in "iuf":
                components[name] = value  # Preserve text Globals for point labels.
                continue
            if array.dtype.kind not in "iuf" or array.size == 0:  # pragma: no cover
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
        except (TypeError, ValueError, OverflowError):  # pragma: no cover
            continue
    return components


RawGlobals: TypeAlias = dict[str, list[dict[str, object]]]


@dataclass
class _SplitColumns:
    """Column-wise Globals of one split.

    Numeric Globals are stored as ``(samples, components)`` float matrices
    with ``NaN`` for missing or non-finite values; other values are kept as
    one Python object per sample (``None`` when absent).
    """

    sample_ids: list[str]
    numeric: dict[str, np.ndarray]
    objects: dict[str, list[object]]


@dataclass(frozen=True)
class PreparedGlobals:
    """Cached plot inputs derived once from raw extracted Global records.

    Attributes:
        splits: Column-wise values keyed by split name.
        names: Sorted plottable scalar components (``Global/x`` or
            ``Global/x_<i>`` for vectors).
        label_names: Sorted components usable as point labels and exported
            as table columns.
        text_names: Sorted non-numeric Globals (e.g. case names), included
            in label and table columns but never plotted.
    """

    splits: dict[str, _SplitColumns]
    names: list[str]
    label_names: list[str]
    text_names: list[str]

    def sample_ids(self, split: str) -> list[str]:
        """Return the sample identifiers of ``split`` in extraction order."""
        columns = self.splits.get(split)
        return columns.sample_ids if columns is not None else []

    def column(self, split: str, name: str) -> np.ndarray | list[object] | None:
        """Return one expanded component of ``split``.

        Args:
            split: Split name.
            name: A name from :attr:`label_names`.

        Returns:
            A float array (``NaN`` for missing values) for numeric components,
            a list of objects for text or other values, or ``None`` when the
            component does not exist in the split.
        """
        columns = self.splits.get(split)
        if columns is None:
            return None
        located = _component_location(name, columns)
        if located is None:
            return None
        source, component = located
        if component is None:
            if source in columns.numeric:
                return columns.numeric[source][:, 0]
            return columns.objects[source]
        return columns.numeric[source][:, component]

    def numeric_column(self, split: str, name: str) -> np.ndarray | None:
        """Return a numeric component of ``split`` as floats, else ``None``."""
        column = self.column(split, name)
        return column if isinstance(column, np.ndarray) else None

    def value(self, split: str, index: int, name: str) -> object:
        """Return the prepared value of one sample, ``None`` when missing."""
        column = self.column(split, name)
        if column is None:
            return None
        value = column[index]
        if isinstance(value, np.floating):
            return float(value) if np.isfinite(value) else None
        return value


def _component_location(
    name: str, columns: _SplitColumns
) -> tuple[str, int | None] | None:
    """Map an expanded component name to its stored Global and column index."""
    if name in columns.objects:
        return name, None
    matrix = columns.numeric.get(name)
    if matrix is not None and matrix.shape[1] == 1:
        return name, None
    source, _, suffix = name.rpartition("_")
    if source and suffix.isdigit():
        matrix = columns.numeric.get(source)
        index = int(suffix) - 1
        if matrix is not None and matrix.shape[1] > 1 and 0 <= index < matrix.shape[1]:
            return source, index
    return None


def _component_names(name: str, width: int) -> list[str]:
    """Return expanded names of a stored Global with ``width`` components."""
    if width == 1:
        return [name]
    return [f"{name}_{index}" for index in range(1, width + 1)]


def _split_columns(rows: list[dict[str, object]]) -> _SplitColumns:
    """Build column-wise storage for one split of raw records.

    Args:
        rows: Raw records with ``sample_id`` and ``values``.

    Returns:
        Numeric Globals as float matrices and other Globals as object lists.
    """
    count = len(rows)
    widths: dict[str, int] = {}
    numeric: dict[str, dict[int, np.ndarray]] = {}
    objects: dict[str, list[object]] = {}
    for index, row in enumerate(rows):
        values: dict[str, object] = row["values"]  # type: ignore[assignment]
        for name, value in values.items():
            if name.endswith("_times") or value is None:
                continue
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):  # pragma: no cover
                continue
            text = _text_value(array)
            if text is not None or (array.size == 1 and array.dtype.kind not in "iuf"):
                objects.setdefault(name, [None] * count)[index] = (
                    text if text is not None else value
                )
                continue
            if array.dtype.kind not in "iuf" or array.size == 0:
                continue
            flat = array.reshape(-1)
            widths[name] = max(widths.get(name, 0), flat.size)
            numeric.setdefault(name, {})[index] = flat
    matrices: dict[str, np.ndarray] = {}
    for name, width in widths.items():
        if name in objects:  # pragma: no cover - mixed text and numbers
            for index, flat in numeric[name].items():
                objects[name][index] = flat[0] if flat.size == 1 else flat
            continue
        # Keep single precision when stored so: wide float32 Globals would
        # otherwise double in memory. Integers and doubles use float64.
        dtype = (
            np.float32
            if all(flat.dtype == np.float32 for flat in numeric[name].values())
            else np.float64
        )
        # Shorter vectors leave NaN gaps in trailing components.
        matrix = np.full((count, width), np.nan, dtype=dtype)
        for index, flat in numeric[name].items():
            matrix[index, : flat.size] = flat
        matrix[~np.isfinite(matrix)] = np.nan
        matrices[name] = matrix
    return _SplitColumns([str(row["sample_id"]) for row in rows], matrices, objects)


def prepare_globals(records: RawGlobals) -> PreparedGlobals:
    """Store raw records column-wise and cache numeric and label names."""
    splits = {split: _split_columns(rows) for split, rows in records.items()}
    names: set[str] = set()
    labels: set[str] = set()
    texts: set[str] = set()
    for columns in splits.values():
        for name, matrix in columns.numeric.items():
            expanded = _component_names(name, matrix.shape[1])
            labels.update(expanded)
            present = ~np.isnan(matrix).all(axis=0)
            names.update(
                component for component, keep in zip(expanded, present) if keep
            )
        labels.update(columns.objects)
        texts.update(columns.objects)
    return PreparedGlobals(splits, sorted(names), sorted(labels), sorted(texts))


def _prepared(records: RawGlobals | PreparedGlobals) -> PreparedGlobals:
    return records if isinstance(records, PreparedGlobals) else prepare_globals(records)


def global_names(records: RawGlobals | PreparedGlobals) -> list[str]:
    """List Global paths with at least one plottable value.

    Args:
        records: Extracted split records.

    Returns:
        Sorted names of numeric scalar Globals (excluding time companions).
    """
    return _prepared(records).names


def label_names(records: RawGlobals | PreparedGlobals) -> list[str]:
    """List all extracted Global paths usable as point labels.

    Args:
        records: Extracted split records.

    Returns:
        Sorted paths, including non-numeric values but excluding time arrays.
    """
    return _prepared(records).label_names


def _table_rows(
    prepared: PreparedGlobals, selected: set[tuple[str, str]] | None
) -> list[tuple[str, list[int]]]:
    """List exported row indices per split, preserving extraction order."""
    return [
        (
            split,
            [
                index
                for index, sample_id in enumerate(columns.sample_ids)
                if selected is None or (split, sample_id) in selected
            ],
        )
        for split, columns in prepared.splits.items()
    ]


def _arrow_block(
    prepared: PreparedGlobals,
    split: str,
    indices: list[int],
    columns: list[str],
):
    """Build an Arrow table for ``indices`` rows of ``split``.

    Args:
        prepared: Column-wise Globals.
        split: Split name.
        indices: Row indices within the split.
        columns: Expanded Global names, in output order.

    Returns:
        A ``pyarrow.Table`` with ``split``, ``sample_id`` and ``columns``.
    """
    import pyarrow as pa  # noqa: PLC0415

    split_columns = prepared.splits[split]
    rows = np.asarray(indices, dtype=np.int64)
    arrays = [
        pa.array([split] * len(indices), pa.string()),
        pa.array([split_columns.sample_ids[i] for i in indices], pa.string()),
    ]
    for name in columns:
        column = prepared.column(split, name)
        if column is None:
            arrays.append(pa.nulls(len(indices), pa.float64()))
        elif isinstance(column, np.ndarray):
            values = column[rows]
            arrays.append(pa.array(values, mask=np.isnan(values)))
        else:
            arrays.append(
                pa.array([_cell(column[i]) for i in indices], pa.string(), safe=False)
            )
    return pa.Table.from_arrays(arrays, names=["split", "sample_id", *columns])


def _cell(value: object) -> str | None:
    """Format a non-numeric Global for tabular exports."""
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("ascii", "replace")
    return str(value)


def iter_globals_csv(
    records: RawGlobals | PreparedGlobals,
    selected: set[tuple[str, str]] | None = None,
    *,
    columns: list[str] | None = None,
    chunk_rows: int | None = None,
) -> Iterator[bytes]:
    """Stream the extracted Globals as UTF-8 CSV without a full table.

    Args:
        records: Extracted Globals, raw or prepared.
        selected: Optional ``(split, sample_id)`` pairs to retain.
        columns: Optional expanded Global columns, defaulting to all labels.
        chunk_rows: Number of samples encoded per chunk. By default, chunks
            hold about four million cells so wide exports use bounded memory.

    Yields:
        CSV bytes: a UTF-8 BOM and header, then blocks of rows.
    """
    import csv  # noqa: PLC0415
    import io  # noqa: PLC0415

    import pyarrow.csv as pa_csv  # noqa: PLC0415

    prepared = _prepared(records)
    columns = prepared.label_names if columns is None else columns
    header = io.StringIO()
    csv.writer(header, lineterminator="\n").writerow(["split", "sample_id", *columns])
    yield ("﻿" + header.getvalue()).encode("utf-8")
    if chunk_rows is None:
        chunk_rows = max(1, min(4096, 4_000_000 // (len(columns) + 2)))
    options = pa_csv.WriteOptions(include_header=False)
    for split, indices in _table_rows(prepared, selected):
        for start in range(0, len(indices), chunk_rows):
            block = _arrow_block(
                prepared, split, indices[start : start + chunk_rows], columns
            )
            buffer = io.BytesIO()
            pa_csv.write_csv(block, buffer, options)
            yield buffer.getvalue()


def globals_dataframe(
    records: RawGlobals | PreparedGlobals,
    selected: set[tuple[str, str]] | None = None,
    columns: list[str] | None = None,
):
    """Return one tabular row per extracted sample.

    Args:
        records: Extracted Globals, raw or expanded for plotting.
        selected: Optional ``(split, sample_id)`` pairs to retain.
        columns: Optional expanded Global columns, defaulting to all labels.

    Returns:
        A pandas DataFrame with split, sample_id and expanded Global columns.
    """
    import pandas as pd  # noqa: PLC0415

    prepared = _prepared(records)
    columns = prepared.label_names if columns is None else columns
    frames = []
    for split, indices in _table_rows(prepared, selected):
        rows = np.asarray(indices, dtype=np.int64)
        data: dict[str, object] = {
            "split": [split] * len(indices),
            "sample_id": [prepared.splits[split].sample_ids[i] for i in indices],
        }
        for name in columns:
            column = prepared.column(split, name)
            if column is None:
                data[name] = np.full(len(indices), np.nan)
            elif isinstance(column, np.ndarray):
                data[name] = column[rows]
            else:
                data[name] = pd.Series([column[i] for i in indices], dtype=object)
        frames.append(pd.DataFrame(data, columns=["split", "sample_id", *columns]))
    if not frames:
        return pd.DataFrame(columns=["split", "sample_id", *columns])
    return pd.concat(frames, ignore_index=True)


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


def _split_matrix(
    prepared: PreparedGlobals, split: str, fields: list[str]
) -> np.ndarray:
    """Stack selected numeric components of ``split`` (``NaN`` when missing)."""
    count = len(prepared.sample_ids(split))
    matrix = np.full((count, len(fields)), np.nan)
    for axis, field in enumerate(fields):
        column = prepared.numeric_column(split, field)
        if column is not None:
            matrix[:, axis] = column
    return matrix


def _optional(numbers: np.ndarray) -> list[float | None]:
    """Convert ``NaN`` gaps to ``None`` for Plotly and JSON serialization."""
    return [None if np.isnan(number) else float(number) for number in numbers]


def build_globals_figure(
    records: RawGlobals | PreparedGlobals,
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
    prepared = _prepared(records)
    names = prepared.names
    dimensions = {"1D": 1, "2D": 2, "3D": 3, "parallel": len(names)}
    if kind not in dimensions:
        raise ValueError(f"Unknown plot type: {kind}")
    if parallel_mode not in {"lines", "parcoords"}:
        raise ValueError(f"Unknown parallel plot mode: {parallel_mode}")
    if kind == "parallel":
        fields = names if parallel_fields is None else parallel_fields
    else:
        fields = axes[: dimensions[kind]]
    known = set(names)
    if (
        not splits
        or (kind != "parallel" and len(fields) != dimensions[kind])
        or not fields
        or any(field not in known for field in fields)
    ):
        return None

    matrices = {split: _split_matrix(prepared, split, fields) for split in splits}
    ranges: list[tuple[float, float] | None] = []
    if kind == "parallel":
        stacked = np.vstack(list(matrices.values()))
        for axis in range(len(fields)):
            present = stacked[:, axis][~np.isnan(stacked[:, axis])]
            ranges.append(
                (float(present.min()), float(present.max())) if present.size else None
            )

    figure = go.Figure()
    dashes = ["solid", "dash", "dot", "dashdot"]
    symbols = ["circle", "square", "diamond", "cross", "x"]
    if kind == "parallel" and parallel_mode == "parcoords":
        blocks = []
        sample_ids: list[str] = []
        split_values: list[int] = []
        plotted_splits = []
        for split in splits:
            matrix = matrices[split]
            complete = ~np.isnan(matrix).any(axis=1)
            if not complete.any():
                continue
            split_index = len(plotted_splits)
            plotted_splits.append(split)
            blocks.append(matrix[complete])
            ids = prepared.sample_ids(split)
            sample_ids.extend(ids[i] for i in np.flatnonzero(complete))
            split_values.extend([split_index] * int(complete.sum()))
        if not sample_ids:  # pragma: no cover
            return None
        values = np.vstack(blocks)

        trace_dimensions = [
            {
                "label": field.removeprefix("Global/"),
                "values": values[:, axis].tolist(),
                "range": list(axis_range) if axis_range is not None else None,
            }
            for axis, (field, axis_range) in enumerate(zip(fields, ranges))
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
        matrix = matrices[split]
        ids = prepared.sample_ids(split)
        if kind == "parallel":
            if parallel_mode == "parcoords":
                continue
            plotted = False
            for index in np.flatnonzero(~np.isnan(matrix).all(axis=1)):
                numbers = _optional(matrix[index])
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
                        customdata=[[ids[index], number] for number in numbers],
                        hovertemplate=(
                            "Sample %{customdata[0]}<br>"
                            "Value: %{customdata[1]}<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
                plotted = True
        else:
            points = np.flatnonzero(~np.isnan(matrix).any(axis=1))
            if not points.size:  # pragma: no cover
                continue
            text = None
            if label:
                text = [
                    format_point_label(value) if value is not None else ""
                    for value in (
                        ids[index]
                        if label == "sample_id"
                        else prepared.value(split, int(index), label)
                        for index in points
                    )
                ]
            if kind == "1D":
                x = [int(ids[index]) for index in points]
                y = matrix[points, 0].tolist()
            else:
                x = matrix[points, 0].tolist()
                y = matrix[points, 1].tolist()
            props = {
                "x": x,
                "y": y,
                "name": split,
                "mode": "markers+text" if label else "markers",
                "text": text,
                "textposition": "top center",
                "marker": {
                    "color": color,
                    "symbol": symbols[split_index % len(symbols)],
                    "size": 8,
                },
                "customdata": [ids[index] for index in points],
            }
            if kind == "3D":
                props["z"] = matrix[points, 2].tolist()
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
