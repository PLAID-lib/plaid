"""Extract, merge, and repartition local PLAID datasets."""

import argparse
import math
import tempfile
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from plaid.containers.sample import Sample
from plaid.infos import Infos
from plaid.storage import init_from_disk, save_to_disk
from plaid.storage.registry import available_backends

SampleId = tuple[int, str, int]


def _load(sources: Sequence[str | Path]) -> list[tuple[dict, dict, Infos]]:
    """Load source collections, converters, and infos for the given paths."""
    loaded = []
    for source in sources:
        source = Path(source).expanduser().resolve()
        datasets, converters = init_from_disk(source)
        loaded.append((datasets, converters, Infos.from_path(source / "infos.yaml")))
        definitions = source / "problem_definitions"
        if definitions.is_dir() and any(definitions.iterdir()):
            warnings.warn(
                f"Problem definitions from {source} are omitted: their split/index "
                "references may no longer be valid.",
                UserWarning,
                stacklevel=2,
            )
    return loaded


def _split_name(name: str) -> str:
    """Validate and return a split name safe for all storage backends."""
    if not name or name in {".", ".."} or any(c in name for c in "/\\:"):
        raise ValueError(f"Invalid split name: {name!r}")
    return name


def _indices(expression: str, size: int) -> list[int]:
    """Expand comma-separated indices and stop-exclusive slices within size."""
    indices = []
    for token in expression.split(","):
        if ":" not in token:
            index = int(token)
            if not 0 <= index < size:
                raise ValueError(f"Sample index {index} outside [0, {size})")
            indices.append(index)
        else:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid sample slice: {token!r}")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if parts[1] else size
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if not 0 <= start <= stop <= size or step <= 0:
                raise ValueError(f"Invalid sample slice {token!r} for {size} samples")
            indices.extend(range(start, stop, step))
    return indices


def _selection(datasets: dict, select: Sequence[str] | None) -> dict[str, list[int]]:
    """Resolve split or split:indices selectors, rejecting repeated samples."""
    result: dict[str, list[int]] = {}
    for selector in select if select is not None else list(datasets):
        split, separator, expression = selector.partition(":")
        _split_name(split)
        if split not in datasets:
            raise ValueError(f"Unknown split {split!r}; available: {list(datasets)}")
        indices = (
            _indices(expression, len(datasets[split]))
            if separator
            else list(range(len(datasets[split])))
        )
        result.setdefault(split, []).extend(indices)
    for split, indices in result.items():
        if not indices:
            raise ValueError(f"Selection for split {split!r} is empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"Repeated samples in split {split!r}")
    return result


def _write(
    loaded: list[tuple[dict, dict, Infos]],
    ids: dict[str, list[SampleId]],
    output: str | Path,
    backend: str | None,
) -> None:
    """Write identifiers to a new output, publishing only after success."""
    if not ids or any(not values for values in ids.values()):
        raise ValueError("The output must contain nonempty splits")
    for split in ids:
        _split_name(split)
    backend = backend if backend is not None else loaded[0][2].storage_backend
    if backend not in available_backends():
        raise ValueError(f"Unknown backend {backend!r}: {available_backends()}")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    def sample_constructor(identifier: SampleId) -> Sample:
        """Return the source sample identified by dataset, split, and index."""
        source, split, index = identifier
        datasets, converters, _ = loaded[source]
        return converters[split].to_plaid(datasets[split], index)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".plaid-tools-", dir=output.parent) as tmp:
        staged = Path(tmp) / "dataset"
        save_to_disk(
            output_folder=staged,
            sample_constructor=sample_constructor,
            ids=ids,
            backend=backend,
            infos=loaded[0][2],
            num_proc=1,
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output already exists: {output}")
        staged.rename(output)


def subset(
    source: str | Path,
    output: str | Path,
    *,
    select: Sequence[str] | None = None,
    backend: str | None = None,
) -> None:
    """Extract splits or samples without modifying the source.

    Args:
        source: Existing local PLAID dataset directory.
        output: New dataset directory; must not exist.
        select: Split names or ``split:0,2,5:10`` selectors. None selects all.
        backend: Output backend override; defaults to the source backend.

    Returns:
        None. Problem definitions are omitted with a warning.
    """
    loaded = _load([source])
    selection = _selection(loaded[0][0], select)
    ids = {
        split: [(0, split, i) for i in indices] for split, indices in selection.items()
    }
    _write(loaded, ids, output, backend)


def merge(
    sources: Sequence[str | Path],
    output: str | Path,
    *,
    backend: str | None = None,
    metadata: str = "strict",
) -> None:
    """Concatenate same-named splits from at least two datasets in input order.

    Args:
        sources: Dataset directories, possibly using different backends.
        output: New dataset directory; must not exist.
        backend: Output backend; defaults to the first source's backend.
        metadata: ``strict`` rejects conflicting descriptive infos; ``first``
            explicitly uses the first source's infos. Counts are recomputed.

    Returns:
        None. Samples are not deduplicated; problem definitions are omitted.
    """
    if len(sources) < 2:
        raise ValueError("Merge requires at least two source datasets")
    if metadata not in {"strict", "first"}:
        raise ValueError("metadata must be 'strict' or 'first'")
    loaded = _load(sources)
    descriptive = [
        infos.model_dump(exclude={"num_samples", "storage_backend"})
        for _, _, infos in loaded
    ]
    if metadata == "strict" and any(info != descriptive[0] for info in descriptive[1:]):
        raise ValueError(
            "Conflicting dataset metadata; use --metadata first to accept it"
        )
    ids: dict[str, list[SampleId]] = {}
    for source, (datasets, _, _) in enumerate(loaded):
        for split, dataset in datasets.items():
            ids.setdefault(split, []).extend(
                (source, split, i) for i in range(len(dataset))
            )
    _write(loaded, ids, output, backend)


def _sizes(specification: Sequence[str], total: int, ratios: bool) -> dict[str, int]:
    """Validate named sizes and apportion all samples, resolving ties in order."""
    values = {}
    for entry in specification:
        name, separator, value = entry.partition("=")
        _split_name(name)
        if not separator or name in values:
            raise ValueError(f"Expected unique NAME=VALUE entries, got {entry!r}")
        number = float(value) if ratios else int(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("Partition sizes must be finite and positive")
        values[name] = number
    if not values:
        raise ValueError("Specify at least one output split")
    if ratios:
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Partition ratios must sum to 1")
        weight = sum(values.values())
        quotas = {name: total * value / weight for name, value in values.items()}
        sizes = {name: math.floor(value) for name, value in quotas.items()}
        order = sorted(
            values, key=lambda name: quotas[name] - sizes[name], reverse=True
        )
        for name in order[: total - sum(sizes.values())]:
            sizes[name] += 1
    else:
        sizes = values
        if sum(sizes.values()) != total:
            raise ValueError(
                f"Partition counts must sum to the selected sample count ({total})"
            )
    if any(size == 0 for size in sizes.values()):
        raise ValueError("Not enough samples: every output split must be nonempty")
    return sizes


def partition(
    source: str | Path,
    output: str | Path,
    *,
    ratios: Sequence[str] | None = None,
    counts: Sequence[str] | None = None,
    select: Sequence[str] | None = None,
    seed: int = 42,
    shuffle: bool = True,
    backend: str | None = None,
) -> None:
    """Redistribute selected samples into disjoint splits covering the selection.

    Args:
        source: Existing local PLAID dataset directory.
        output: New dataset directory; must not exist.
        ratios: ``NAME=RATIO`` entries summing to one; exclusive of counts.
        counts: ``NAME=COUNT`` entries summing to the selected sample count.
        select: Source selectors as in subset; None pools all source splits.
        seed: Nonnegative random seed, used when shuffle is True.
        shuffle: Shuffle first; otherwise preserve selection order.
        backend: Output backend override; defaults to the source backend.

    Returns:
        None. Unselected samples and original problem definitions are omitted.
    """
    if (ratios is None) == (counts is None):
        raise ValueError("Specify exactly one of ratios or counts")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    loaded = _load([source])
    selection = _selection(loaded[0][0], select)
    pool = [(0, split, i) for split, indices in selection.items() for i in indices]
    sizes = _sizes(
        ratios if ratios is not None else counts, len(pool), ratios is not None
    )
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(pool))
        pool = [pool[int(i)] for i in order]
    ids = {}
    start = 0
    for name, size in sizes.items():
        ids[name] = pool[start : start + size]
        start += size
    _write(loaded, ids, output, backend)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset CLI.

    Args:
        argv: Arguments excluding the executable, or None for sys.argv.

    Returns:
        Zero on success. Invalid arguments or operations exit with status two.
    """
    parser = argparse.ArgumentParser(description=__doc__, prog="plaid-tools")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("subset", "Extract selected splits or samples"),
        ("merge", "Concatenate two or more datasets"),
        ("partition", "Generate disjoint splits from selected samples"),
    ):
        command = commands.add_parser(name, help=help_text)
        if name == "merge":
            command.add_argument("sources", nargs="+", type=Path)
            command.add_argument(
                "--metadata", choices=("strict", "first"), default="strict"
            )
        else:
            command.add_argument("source", type=Path)
            command.add_argument(
                "--select",
                action="append",
                metavar="SPLIT[:INDICES]",
                help="Repeat selections; indices use commas and Python slices",
            )
        command.add_argument("output", type=Path)
        command.add_argument(
            "--backend", choices=available_backends(), help="Output backend override"
        )
        if name == "partition":
            sizes = command.add_mutually_exclusive_group(required=True)
            sizes.add_argument("--ratios", nargs="+", metavar="NAME=RATIO")
            sizes.add_argument("--counts", nargs="+", metavar="NAME=COUNT")
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    options = vars(parser.parse_args(argv))
    operation = {"subset": subset, "merge": merge, "partition": partition}[
        options.pop("command")
    ]
    try:
        operation(**options)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved dataset to {options['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
