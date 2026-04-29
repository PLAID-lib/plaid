"""Dataset discovery and sample introspection for the PLAID viewer.

This service owns all PLAID-facing logic used by the viewer:

- Discover datasets under a configured root directory.
- Load a split-wise ``(dataset_dict, converter_dict)`` pair through
  :func:`plaid.storage.init_from_disk` and cache it for subsequent calls.
- Materialize PLAID :class:`plaid.Sample` instances via
  ``converter.to_plaid(dataset, index)``, regardless of the underlying
  backend (``hf_datasets``, ``cgns``, ``zarr`` ...).
- Summarize sample contents (bases, zones, fields, times, scalars).
- Report basic validation status via :meth:`Sample.check_completeness`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from plaid.viewer.config import ViewerConfig
from plaid.viewer.models import (
    DatasetDetail,
    DatasetInfo,
    SampleRef,
    SampleRefDTO,
    SampleSummary,
    ValidationResult,
)

logger = logging.getLogger(__name__)


# Sentinel ``sample_id`` used for streaming datasets, where the only
# addressable sample is "the one currently produced by the iterator".
STREAM_CURSOR_ID = "cursor"


@dataclass
class _StreamCursor:
    """Forward-only cursor over a streaming (``IterableDataset``) split.

    Streaming datasets returned by
    :func:`plaid.storage.init_streaming_from_hub` do not support
    indexing or ``len``. This cursor consumes the underlying iterable
    one sample at a time and caches the most recently produced raw
    record so repeated ``load_sample`` calls (e.g. when the UI loads
    summary then full sample) do not advance the stream.
    """

    iterator: Iterator[Any] | None = None
    position: int = -1  # -1 means "no sample fetched yet".
    current_record: Any | None = None
    exhausted: bool = False
    extras: dict = field(default_factory=dict)


def _safe_list_dir(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir())


def _array_preview(value, *, max_items: int = 6) -> str | None:
    """Return a short string preview of a numpy-like array value."""
    if value is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:  # pragma: no cover - numpy is a transitive dep
        return None
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return "[]"
    flat = arr.ravel()
    if flat.size <= max_items:
        return np.array2string(arr, separator=", ", threshold=max_items + 1)
    head = np.array2string(flat[:max_items], separator=", ")
    return f"{head[:-1]}, ...] (total {flat.size} values)"


def _collect_data_arrays(cgns_node) -> list[dict[str, object]]:
    """Recursively collect ``DataArray_t`` descriptors under ``cgns_node``.

    Each entry contains the array name, its shape as a list, dtype as a
    string, and a short string preview of the values.
    """
    try:
        from CGNS.PAT import cgnskeywords as CK  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return []

    entries: list[dict[str, object]] = []

    def _walk(node) -> None:
        name, value, children, label = node
        if label == CK.DataArray_ts:
            shape = list(getattr(value, "shape", ())) if value is not None else []
            dtype = str(getattr(value, "dtype", ""))
            entries.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "preview": _array_preview(value),
                }
            )
            return
        for child in children or []:
            _walk(child)

    for child in cgns_node[2] or []:
        _walk(child)
    return entries


class PlaidDatasetService:
    """High-level access to PLAID datasets stored under a root directory.

    A dataset is a subdirectory of ``config.datasets_root`` that contains a
    ``data/`` directory readable by :func:`plaid.storage.init_from_disk`.
    The function returns a ``dataset_dict`` and a ``converter_dict`` keyed
    by split name; the viewer iterates splits and addresses samples by
    integer index in ``range(len(dataset_dict[split]))``.
    """

    def __init__(self, config: ViewerConfig) -> None:
        self._config = config
        # Datasets root is kept on the service (not on the frozen config)
        # so it can be changed at runtime through ``set_datasets_root``.
        # ``None`` means no root has been selected yet: discovery methods
        # return empty lists and the UI is expected to prompt the user.
        self._datasets_root: Path | None = (
            Path(config.datasets_root) if config.datasets_root is not None else None
        )
        # Sandbox for interactive root selection. Defaults to the user's
        # home directory when no explicit ``browse_roots`` is configured.
        # The configured ``datasets_root`` is always implicitly allowed so
        # ``list_subdirs`` can start from there.
        browse_roots: list[Path] = [Path(p).expanduser() for p in config.browse_roots]
        if not browse_roots:
            browse_roots = [Path.home()]
        if self._datasets_root is not None:
            # Make sure the startup root is always reachable even if
            # ``browse_roots`` is more restrictive.
            browse_roots.append(self._datasets_root)
        self._browse_roots: tuple[Path, ...] = tuple(
            dict.fromkeys(p.resolve() for p in browse_roots)
        )
        # Cache of (dataset_dict, converter_dict) keyed by dataset_id to
        # avoid re-parsing large arrow/zarr datasets on every call.
        self._store_cache: dict[str, tuple[dict, dict]] = {}
        # Registered Hugging Face Hub repositories that should be exposed
        # as datasets through :func:`plaid.storage.init_streaming_from_hub`.
        # The ``dataset_id`` used throughout the viewer is the raw
        # ``repo_id`` string (e.g. ``"PLAID-lib/VKI-LS59"``), which never
        # collides with a local directory name (it always contains a
        # forward slash).
        self._hub_repos: list[str] = []
        # Per-(dataset_id, split) streaming cursors. Streaming datasets
        # are ``datasets.IterableDataset`` instances without ``__len__``
        # so we cannot index them. We maintain a forward-only cursor
        # instead: ``_cursors[(dataset_id, split)] = (iterator, position,
        # cached_sample)``. ``Next`` consumes the iterator and advances
        # ``position``; ``Reset`` discards the iterator so a fresh one is
        # built on the next access.
        self._cursors: dict[tuple[str, str], _StreamCursor] = {}
        # User-selected feature filter per dataset. ``None`` means "no
        # filter" (load every feature, current default behaviour). An
        # empty list means "all features unselected".
        self._features: dict[str, list[str] | None] = {}
        # Memoised ``(constant_feature_keys, variable_feature_keys)`` per
        # dataset, retrieved through ``load_metadata_from_disk`` or
        # ``load_metadata_from_hub``. Used to (a) populate the UI
        # checkbox list through :meth:`list_available_features` and (b)
        # expand user-selected feature paths with
        # :func:`plaid.utils.cgns_helper.update_features_for_CGNS_compatibility`
        # before handing them to ``init_streaming_from_hub`` (which, unlike
        # :meth:`Converter.to_plaid`, does not expand features by itself).
        self._feature_metadata: dict[str, tuple[list[str], list[str]]] = {}
        # Memoised per-split feature catalogue for a dataset. Unlike
        # ``_feature_metadata`` (which aggregates constants across
        # splits so the UI can offer a union of fields), this mapping
        # preserves the split boundary so :meth:`load_sample` can
        # filter the user's selection down to what a specific split
        # actually carries. ``PlaidSampleConverter.to_plaid`` otherwise
        # raises ``KeyError('Missing features in dataset/converter:
        # ...')`` whenever the request names a path that the split in
        # hand does not know about.
        self._split_feature_metadata: dict[str, dict[str, set[str]]] = {}

    # ----------------------------------------------------------- Discovery

    @property
    def datasets_root(self) -> Path | None:
        """Return the currently active datasets root, or ``None``."""
        return self._datasets_root

    @property
    def browse_roots(self) -> tuple[Path, ...]:
        """Return the sandbox directories for interactive path selection."""
        return self._browse_roots

    def set_datasets_root(self, path: Path | str | None) -> Path | None:
        """Change the active datasets root at runtime.

        The new path (when not ``None``) must exist, be a directory, and be
        located under one of ``browse_roots``. All per-dataset caches are
        invalidated so the next discovery call reflects the new root.

        Args:
            path: The new datasets root. ``None`` clears the current root.

        Returns:
            The resolved new datasets root, or ``None`` if cleared.

        Raises:
            ValueError: If the path does not exist, is not a directory, or
                escapes ``browse_roots``.
        """
        # Deferred import so the service module stays importable without
        # write access to the user config directory (e.g. in read-only
        # CI sandboxes that don't touch ``set_datasets_root`` anyway).
        from plaid.viewer.preferences import (  # noqa: PLC0415
            set_last_datasets_root,
        )

        if path is None:
            self._datasets_root = None
            self._store_cache.clear()
            set_last_datasets_root(None)
            return None
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Not a directory: {resolved}")
        self._ensure_within_browse_roots(resolved)
        self._datasets_root = resolved
        self._store_cache.clear()
        # Persist the new root so the next launch of the viewer picks it
        # up automatically when ``--datasets-root`` is not provided.
        set_last_datasets_root(resolved)
        return resolved

    def list_subdirs(self, path: Path | str | None = None) -> dict[str, object]:
        """Return immediate subdirectories of ``path`` for the file browser.

        Each entry is tagged with ``is_plaid_candidate`` (``True`` when it
        looks like a PLAID dataset, i.e. contains a ``data/`` subdirectory)
        so the UI can highlight it. The returned ``path`` is always an
        absolute resolved path inside ``browse_roots``.

        Args:
            path: Directory to list. When ``None`` the first browse root is
                used (typically ``$HOME``).

        Returns:
            A dict ``{"path": str, "parent": str | None,
            "entries": [{"name": str, "path": str,
            "is_plaid_candidate": bool}, ...]}``.

        Raises:
            ValueError: If ``path`` is not a directory or escapes the
                sandbox.
        """
        if path is None:
            target = self._browse_roots[0]
        else:
            target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError(f"Not a directory: {target}")
        self._ensure_within_browse_roots(target)
        entries: list[dict[str, object]] = []
        for entry in sorted(target.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            entries.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "is_plaid_candidate": (entry / "data").is_dir(),
                }
            )
        # Rank PLAID candidates first, then alphabetical (stable).
        entries.sort(key=lambda e: (not e["is_plaid_candidate"], e["name"].lower()))
        parent: str | None = None
        if any(
            target != root and root in target.parents for root in self._browse_roots
        ):
            parent = str(target.parent)
        elif (
            target.parent != target
            and any(  # pragma: no cover - alternate browse-root ancestry guard
                target.parent == root or root in target.parent.parents
                for root in self._browse_roots
            )
        ):
            parent = str(target.parent)
        return {
            "path": str(target),
            "parent": parent,
            "entries": entries,
        }

    def _ensure_within_browse_roots(self, path: Path) -> None:
        for root in self._browse_roots:
            try:
                path.relative_to(root)
            except ValueError:
                continue
            return
        roots = ", ".join(str(r) for r in self._browse_roots)
        raise ValueError(f"Path {path} is outside the allowed browse roots ({roots}).")

    def list_datasets(self) -> list[DatasetInfo]:
        """Return a summary of every dataset available to the viewer.

        Local datasets (subdirectories of ``datasets_root``) and registered
        Hugging Face Hub repositories (added via :meth:`add_hub_dataset`)
        are both included, in that order.
        """
        infos: list[DatasetInfo] = []
        root = self._datasets_root
        if root is not None:
            for entry in _safe_list_dir(root):
                if not entry.is_dir():
                    continue
                if not (entry / "data").is_dir():
                    continue
                infos.append(
                    DatasetInfo(
                        dataset_id=entry.name,
                        backend_id="disk",
                        path=str(entry),
                        has_infos=(entry / "infos.yaml").exists()
                        or (entry / "infos.json").exists(),
                        has_problem_definitions=(
                            entry / "problem_definitions"
                        ).is_dir(),
                    )
                )
        for repo_id in self._hub_repos:
            infos.append(
                DatasetInfo(
                    dataset_id=repo_id,
                    backend_id="hub",
                    path=f"hf://{repo_id}",
                    has_infos=False,
                    has_problem_definitions=False,
                )
            )

        return infos

    @property
    def hub_repos(self) -> tuple[str, ...]:
        """Return the list of registered Hugging Face Hub repositories."""
        return tuple(self._hub_repos)

    def add_hub_dataset(self, repo_id: str) -> str:
        """Register a Hugging Face Hub dataset to stream from.

        The dataset is exposed through :func:`plaid.storage.init_streaming_from_hub`
        and appears in :meth:`list_datasets` with ``dataset_id == repo_id``.

        Args:
            repo_id: Hugging Face repository identifier, e.g.
                ``"PLAID-lib/VKI-LS59"``. Must contain a ``/`` separator.

        Returns:
            The normalised ``repo_id``.

        Raises:
            ValueError: If ``repo_id`` is empty or does not look like a
                ``namespace/name`` pair.
        """
        normalised = (repo_id or "").strip()
        if not normalised:
            raise ValueError("repo_id must be a non-empty string.")
        if "/" not in normalised:
            raise ValueError(
                f"repo_id {normalised!r} must be of the form 'namespace/name'."
            )
        if normalised in self._hub_repos:
            return normalised
        self._hub_repos.append(normalised)
        return normalised

    def remove_hub_dataset(self, repo_id: str) -> None:
        """Unregister a previously added Hugging Face Hub dataset."""
        if repo_id in self._hub_repos:
            self._hub_repos.remove(repo_id)
            self._store_cache.pop(repo_id, None)
            self._features.pop(repo_id, None)
            self._feature_metadata.pop(repo_id, None)
            # Drop any streaming cursors owned by the removed dataset.
            self._cursors = {
                key: cur for key, cur in self._cursors.items() if key[0] != repo_id
            }

    # ------------------------------------------------------- Feature filter

    def _load_feature_metadata(self, dataset_id: str) -> tuple[list[str], list[str]]:
        """Return ``(constant_feature_keys, variable_feature_keys)`` for a dataset.

        Uses :func:`plaid.storage.common.reader.load_metadata_from_disk` for
        local datasets and :func:`plaid.storage.common.reader.load_metadata_from_hub`
        for registered Hugging Face Hub repositories. The result is
        memoised on the service instance.

        Constant features are aggregated across splits (constant schemas
        in PLAID are split-specific), variable features are global.
        """
        if dataset_id in self._feature_metadata:
            return self._feature_metadata[dataset_id]
        # Deferred imports so the module stays importable without PLAID.
        from plaid.storage.common.reader import (  # noqa: PLC0415
            load_metadata_from_disk,
            load_metadata_from_hub,
        )

        if self._is_hub_dataset(dataset_id):
            _flat_cst, variable_schema, constant_schema, _cgns_types = (
                load_metadata_from_hub(dataset_id)
            )
        else:
            base = self._dataset_dir(dataset_id)
            _flat_cst, variable_schema, constant_schema, _cgns_types = (
                load_metadata_from_disk(str(base))
            )
        constant_keys: set[str] = set()
        for split_const in (constant_schema or {}).values():
            constant_keys.update(split_const.keys())
        variable_keys = list((variable_schema or {}).keys())
        metadata = (sorted(constant_keys), sorted(variable_keys))
        self._feature_metadata[dataset_id] = metadata
        # Build the per-split catalogue in one pass: variable features
        # are global so every split shares them, but constant features
        # are keyed by split.
        per_split: dict[str, set[str]] = {
            split: set(variable_keys) | set(split_const.keys())
            for split, split_const in (constant_schema or {}).items()
        }
        self._split_feature_metadata[dataset_id] = per_split
        return metadata

    def _split_feature_keys(self, dataset_id: str, split_key: str) -> set[str]:
        """Return the feature catalogue of a single split.

        Ensures the per-split mapping is populated (it is filled as a
        side effect of :meth:`_load_feature_metadata`). Falls back to
        the dataset-wide union when the split name is not recorded
        (typical for streaming datasets that expose a single
        ``__default__`` split).
        """
        if dataset_id not in self._split_feature_metadata:
            self._load_feature_metadata(dataset_id)
        per_split = self._split_feature_metadata.get(dataset_id, {})
        if split_key in per_split:
            return per_split[split_key]
        constant_keys, variable_keys = self._load_feature_metadata(dataset_id)
        return set(constant_keys) | set(variable_keys)

    def list_available_features(self, dataset_id: str) -> list[str]:
        """Return the feature paths offered to the user for filtering.

        The viewer only exposes paths that are CGNS *fields* (i.e. what
        :func:`plaid.containers.utils.get_feature_details_from_path`
        classifies as ``type == "field"``). Globals, coordinates,
        element connectivities, boundary conditions, etc. are hidden
        because they are not what the user means when they want to
        "filter the displayed features" in a 3D viewer.

        Paths ending in ``_times`` (time-series bookkeeping duplicates
        of a field, e.g. ``Base_.../FlowSolution/Pressure_times``) are
        also filtered out: they are artefacts of the temporal storage
        layout, not distinct physical quantities the user would want to
        toggle.
        """
        # Deferred import - the helper lives in PLAID's containers module.
        from plaid.containers.utils import (  # noqa: PLC0415
            get_feature_details_from_path,
        )

        constant_keys, variable_keys = self._load_feature_metadata(dataset_id)
        candidates = set(constant_keys) | set(variable_keys)
        fields: list[str] = []
        for path in candidates:
            if path.endswith("_times"):
                continue
            try:
                details = get_feature_details_from_path(path)
            except Exception:  # noqa: BLE001 - malformed path, skip
                continue
            # Only expose "genuine" field paths - i.e. those that carry
            # a ``name`` entry in ``details``. Some variants returned by
            # :func:`get_feature_details_from_path` are typed as
            # ``"field"`` but describe a container (e.g. a
            # ``FlowSolution_t`` node) rather than a specific data array,
            # and therefore have no ``name``. Filtering on ``name``
            # removes those from the UI while keeping every real scalar
            # / vector field the user can actually plot.
            # ``GridLocation`` nodes are CGNS metadata (they describe
            # *where* a field lives, e.g. ``Vertex`` vs ``CellCenter``)
            # rather than a plottable field, so they must not appear in
            # the viewer's feature selection.
            name = details.get("name")
            if details.get("type") == "field" and name and name != "GridLocation":
                fields.append(path)
        return sorted(fields)

    def get_features(self, dataset_id: str) -> list[str] | None:
        """Return the active feature filter for ``dataset_id``.

        ``None`` means "no filter": every feature is loaded (default
        behaviour). An explicit empty list means "no feature selected".
        """
        return self._features.get(dataset_id)

    def set_features(
        self, dataset_id: str, features: list[str] | None
    ) -> list[str] | None:
        """Set (or clear) the active feature filter for ``dataset_id``.

        Only the *user-visible* field paths (those returned by
        :meth:`list_available_features`) are stored. Geometric supports
        (coordinates, element connectivities, boundary conditions,
        ``GridLocation`` metadata, ``_times`` bookkeeping paths, ...)
        required to render the selected fields are handled transparently
        by :meth:`Converter.to_plaid`, which runs
        :func:`~plaid.utils.cgns_helper.update_features_for_CGNS_compatibility`
        internally against its *own* per-split
        ``constant_features`` / ``variable_features`` catalogues. We
        therefore never pre-expand the selection here - doing so would
        use the dataset-wide (union) catalogue and, on splits whose
        data does not contain the selected fields, would hand PLAID a
        list of coordinates *without the fields that justify them* and
        trigger ``Missing features in dataset/converter`` in the CGNS
        expander.

        For disk-backed datasets the filter is applied on every call to
        :meth:`Converter.to_plaid` during :meth:`load_sample`. For
        streaming (Hugging Face Hub) datasets it is injected into
        :func:`plaid.storage.init_streaming_from_hub` *before* any
        sample is consumed; we therefore invalidate the cached
        ``(datasetdict, converterdict)`` and any open streaming cursors
        so the next :meth:`_open` call rebuilds them with the new
        feature list.

        Args:
            dataset_id: Target dataset identifier.
            features: Field paths to keep (subset of
                :meth:`list_available_features`), or ``None`` to clear
                the filter and load every feature.

        Returns:
            The normalised, deduplicated feature list (``None`` when no
            filter is active).

        Raises:
            ValueError: If ``features`` contains paths not declared in
                the dataset metadata.
        """
        if features is None:
            normalised: list[str] | None = None
        else:
            normalised = sorted(dict.fromkeys(str(f) for f in features))
            all_keys = set(self._load_feature_metadata(dataset_id)[0]) | set(
                self._load_feature_metadata(dataset_id)[1]
            )
            unknown = [f for f in normalised if f not in all_keys]
            if unknown:
                raise ValueError(
                    f"Unknown features for dataset {dataset_id!r}: {unknown}"
                )
        self._features[dataset_id] = normalised
        # Invalidate store cache so streaming datasets rebuild their
        # IterableDataset with the new feature list. For disk datasets
        # this is not strictly required (features are applied on each
        # ``to_plaid`` call) but keeping a single invalidation policy is
        # simpler and does not hurt performance measurably.
        self._store_cache.pop(dataset_id, None)
        self._cursors = {
            key: cur for key, cur in self._cursors.items() if key[0] != dataset_id
        }
        return normalised

    def is_streaming(self, dataset_id: str) -> bool:
        """Return ``True`` when ``dataset_id`` is a Hugging Face Hub stream.

        Streaming datasets have no ``__len__`` on their splits and must be
        navigated forward-only through :meth:`advance_stream_cursor` /
        :meth:`reset_stream_cursor` rather than indexed.
        """
        if not self._is_hub_dataset(dataset_id):
            return False
        try:
            datasetdict, _ = self._open(dataset_id)
        except Exception:  # noqa: BLE001
            return True
        return not all(hasattr(ds, "__len__") for ds in datasetdict.values())

    def get_dataset(self, dataset_id: str) -> DatasetDetail:
        """Return detailed information about a single dataset."""
        if self._is_hub_dataset(dataset_id):
            splits = self._splits_with_counts(dataset_id)
            return DatasetDetail(
                dataset_id=dataset_id,
                backend_id="hub",
                path=f"hf://{dataset_id}",
                has_infos=False,
                has_problem_definitions=False,
                splits=splits,
                infos=None,
                problem_definitions=[],
            )
        base = self._dataset_dir(dataset_id)
        splits = self._splits_with_counts(dataset_id)
        pb_defs_dir = base / "problem_definitions"
        pb_defs = (
            [
                p.stem
                for p in _safe_list_dir(pb_defs_dir)
                if p.suffix in {".yaml", ".yml"}
            ]
            if pb_defs_dir.is_dir()
            else []
        )
        return DatasetDetail(
            dataset_id=dataset_id,
            backend_id="disk",
            path=str(base),
            has_infos=(base / "infos.yaml").exists() or (base / "infos.json").exists(),
            has_problem_definitions=bool(pb_defs),
            splits=splits,
            infos=self._load_infos(base),
            problem_definitions=pb_defs,
        )

    def list_samples(self, dataset_id: str) -> list[SampleRefDTO]:
        """Return every sample reference available in a dataset.

        For disk-backed datasets, sample ids are the zero-based integer
        indices used with ``converter.to_plaid(dataset, index)``. For
        streaming datasets (Hugging Face Hub), each split contributes a
        single reference whose ``sample_id`` is the
        :data:`STREAM_CURSOR_ID` sentinel; the actual sample is obtained
        by advancing the per-split cursor with
        :meth:`advance_stream_cursor`.
        """
        datasetdict, _ = self._open(dataset_id)
        streaming = self.is_streaming(dataset_id)
        backend_id = "hub" if self._is_hub_dataset(dataset_id) else "disk"

        refs: list[SampleRef] = []
        for split, ds in datasetdict.items():
            split_key = None if split == "__default__" else split
            if streaming:
                refs.append(
                    SampleRef(
                        backend_id=backend_id,
                        dataset_id=dataset_id,
                        split=split_key,
                        sample_id=STREAM_CURSOR_ID,
                    )
                )
                continue
            for index in range(len(ds)):
                refs.append(
                    SampleRef(
                        backend_id=backend_id,
                        dataset_id=dataset_id,
                        split=split_key,
                        sample_id=str(index),
                    )
                )
        return [SampleRefDTO.from_ref(ref) for ref in refs]

    # --------------------------------------------------- Streaming cursors

    def stream_cursor_position(self, dataset_id: str, split: str | None) -> int:
        """Return the current forward position of a streaming cursor.

        Returns ``-1`` before the first call to :meth:`advance_stream_cursor`.
        """
        cursor = self._cursors.get(self._cursor_key(dataset_id, split))
        return cursor.position if cursor is not None else -1

    def advance_stream_cursor(self, dataset_id: str, split: str | None) -> SampleRef:
        """Consume the next record from the stream and return its ref.

        The returned :class:`SampleRef` always carries the
        :data:`STREAM_CURSOR_ID` sentinel in its ``sample_id``; the
        underlying record is cached on the service so a subsequent
        :meth:`load_sample` call returns the freshly fetched sample.

        Raises:
            StopIteration: If the underlying stream is exhausted.
        """
        key = self._cursor_key(dataset_id, split)
        cursor = self._cursors.get(key)
        if cursor is None or cursor.iterator is None:
            cursor = self._build_cursor(dataset_id, split)
            self._cursors[key] = cursor
        try:
            record = next(cursor.iterator)
        except StopIteration:
            cursor.exhausted = True
            raise
        cursor.current_record = record
        cursor.position += 1
        return SampleRef(
            backend_id="hub",
            dataset_id=dataset_id,
            split=split,
            sample_id=STREAM_CURSOR_ID,
        )

    def reset_stream_cursor(self, dataset_id: str, split: str | None) -> None:
        """Rebuild a fresh iterator for ``(dataset_id, split)``.

        The cached record is discarded and the position reset to ``-1``
        so the next :meth:`advance_stream_cursor` call yields the first
        sample again.
        """
        key = self._cursor_key(dataset_id, split)
        self._cursors[key] = self._build_cursor(dataset_id, split)

    @staticmethod
    def _cursor_key(dataset_id: str, split: str | None) -> tuple[str, str]:
        return dataset_id, split if split is not None else "__default__"

    def _build_cursor(self, dataset_id: str, split: str | None) -> _StreamCursor:
        datasetdict, _ = self._open(dataset_id)
        split_key = split if split is not None else "__default__"
        if split_key not in datasetdict and len(datasetdict) == 1:
            split_key = next(iter(datasetdict))
        if split_key not in datasetdict:
            raise KeyError(
                f"Split {split!r} not found in dataset {dataset_id!r}; "
                f"available splits: {sorted(datasetdict.keys())}"
            )
        return _StreamCursor(iterator=iter(datasetdict[split_key]))

    # -------------------------------------------------------------- Samples

    def load_sample(self, ref: SampleRef):
        """Return a PLAID :class:`plaid.Sample` for the given reference.

        Uses ``converter.to_plaid(dataset, index)`` to rebuild the sample
        from whatever backend store (hf_datasets, cgns, zarr) is in use.
        """
        datasetdict, converterdict = self._open(ref.dataset_id)
        split_key = ref.split if ref.split is not None else "__default__"
        if split_key not in datasetdict:
            # Fallback: some converters return a single unnamed split.
            if len(datasetdict) == 1:
                split_key = next(iter(datasetdict))
            else:
                raise KeyError(
                    f"Split {ref.split!r} not found in dataset {ref.dataset_id!r}; "
                    f"available splits: {sorted(datasetdict.keys())}"
                )
        dataset = datasetdict[split_key]
        converter = converterdict[split_key]
        # Streaming datasets expose a forward-only cursor rather than
        # random access. The viewer drives the cursor explicitly via
        # ``advance_stream_cursor`` and then calls ``load_sample`` with
        # ``sample_id == STREAM_CURSOR_ID`` to materialise the PLAID
        # sample from the most recently consumed raw record.
        if ref.sample_id == STREAM_CURSOR_ID:
            cursor = self._cursors.get(self._cursor_key(ref.dataset_id, ref.split))
            if cursor is None or cursor.current_record is None:
                # Auto-advance once so a fresh selection behaves like
                # "show me the first sample".
                self.advance_stream_cursor(ref.dataset_id, ref.split)
                cursor = self._cursors[self._cursor_key(ref.dataset_id, ref.split)]
            # Streaming converters use ``sample_to_plaid`` (single record)
            # rather than ``to_plaid(dataset, index)`` (random access).
            return converter.sample_to_plaid(cursor.current_record)

        try:
            index = int(ref.sample_id)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sample id {ref.sample_id!r}; expected an integer index."
            ) from exc
        features = self._features.get(ref.dataset_id)
        if features is None:
            # No filter active: load every feature.
            return converter.to_plaid(dataset, index)
        # ``features`` is a (possibly empty) list: the filter IS active.
        # We must not fall through to the unfiltered branch, otherwise
        # an empty selection would load every feature instead of none.
        #
        # Feature schemas are split-specific in PLAID: the UI dropdown
        # aggregates every split's catalogue, so a user-selected field
        # may be absent from the current split. ``Converter.to_plaid``
        # runs :func:`~plaid.utils.cgns_helper.update_features_for_CGNS_compatibility`
        # internally against its own per-split ``constant_features`` /
        # ``variable_features`` and raises
        # ``KeyError('Missing features in dataset/converter: ...')``
        # for any unknown path. We therefore intersect the user's
        # field selection with the split's catalogue first. Geometric
        # supports required to render the kept fields are added by the
        # converter itself on the ``to_plaid`` call.
        split_constant = set(getattr(converter, "constant_features", set()))
        split_variable = set(getattr(converter, "variable_features", set()))
        split_keys = split_constant | split_variable
        selected = [f for f in features if f in split_keys]
        # The split's feature catalogue contains more than the fields
        # the user can toggle in the UI: it also carries CGNS
        # bookkeeping paths (coordinates, element connectivities,
        # ``GridLocation`` metadata, ``_times`` series, ...) and the
        # paths backing the sample's globals / scalars. Those entries
        # must always be loaded, otherwise the rendered sample would
        # lose its mesh and the "Globals" panel would be empty.
        #
        # We therefore compute the set of "user-controllable" field
        # paths (the same set the UI exposes through
        # :meth:`list_available_features`) and re-inject *only* the
        # remaining split paths. Filtering by
        # ``set(user_visible) - set(selected)`` is not enough: we have
        # to build the complement inside the current split so that
        # constant fields the user deselected are genuinely dropped.
        user_visible = set(self.list_available_features(ref.dataset_id))
        # ``_times`` bookkeeping paths are hidden from the UI but
        # semantically follow their companion field: toggling ``sdf`` on
        # or off must also toggle ``sdf_times``. Treat them as linked
        # to their base path so deselecting a field genuinely drops
        # both entries (and re-selecting a field adds both back).
        user_visible_linked = user_visible | {f"{path}_times" for path in user_visible}
        selected_linked = set(selected) | {
            f"{path}_times" for path in selected if f"{path}_times" in split_keys
        }
        always_keep = split_keys - user_visible_linked
        augmented = sorted(selected_linked | always_keep)
        if not augmented:
            # Split has no bookkeeping paths AND user-selected fields
            # were all absent from this split: nothing sensible to
            # filter with. Fall back to the unfiltered load so the user
            # still sees *something* (the raw sample).
            return converter.to_plaid(dataset, index)
        try:
            return converter.to_plaid(dataset, index, features=augmented)
        except KeyError:
            # ``augmented`` can itself contain paths that the CGNS
            # expander or the HF bridge reject (bookkeeping entries not
            # materialised as columns in the backend store). A
            # ``KeyError("Missing features in …")`` from that code path
            # should not be user-facing: degrade gracefully to an
            # unfiltered load.
            return converter.to_plaid(dataset, index)

    def get_sample_summary(self, ref: SampleRef) -> SampleSummary:
        """Return a minimal summary of the PLAID sample."""
        sample = self.load_sample(ref)
        times = self._time_keys(sample)
        bases, zones_by_base, fields_by_base = self._describe_tree(sample, times)
        globals_dict = {
            name: str(sample.get_scalar(name)) for name in sample.get_scalar_names()
        }
        return SampleSummary(
            ref=SampleRefDTO.from_ref(ref),
            n_times=len(times),
            time_values=list(times),
            bases=bases,
            zones_by_base=zones_by_base,
            fields_by_base=fields_by_base,
            globals=globals_dict,
        )

    def list_time_values(self, ref: SampleRef) -> list[float]:
        """Return the sorted list of time values available for a sample.

        Thin wrapper around :meth:`plaid.Sample.features.get_all_time_values`
        that always returns a ``list[float]`` (it may be empty for static
        samples).
        """
        sample = self.load_sample(ref)
        try:
            times = sample.features.get_all_time_values()
        except Exception:  # noqa: BLE001 - defensive, PLAID shouldn't raise
            return []
        return sorted(float(t) for t in times)

    def describe_globals(
        self, ref: SampleRef, *, time: float | None = None
    ) -> list[dict[str, object]]:
        """Return PLAID global scalars/tensors reported by the sample.

        Uses :meth:`plaid.Sample.get_global_names` to enumerate globals
        and :meth:`plaid.Sample.get_global` to fetch each value, so only
        the "real" globals exposed by PLAID's API are reported. The CGNS
        bookkeeping arrays ``IterationValues`` and ``TimeValues`` (which
        describe time steps, not physical scalars) are filtered out.

        Args:
            ref: The sample to inspect.
            time: Optional time value; when ``None`` the sample's first
                available time (or the static value) is used.

        Returns:
            A list of ``{"name": str, "shape": list[int], "dtype": str,
            "preview": str | None}`` descriptors, one per global.
        """
        sample = self.load_sample(ref)
        kwargs = {"time": time} if time is not None else {}
        try:
            names = sample.get_global_names(**kwargs)
        except TypeError:
            names = sample.get_global_names()
        entries: list[dict[str, object]] = []
        for name in names:
            if name in {"IterationValues", "TimeValues"}:
                continue
            try:
                value = sample.get_global(name, **kwargs)
            except TypeError:
                value = sample.get_global(name)
            except Exception:  # noqa: BLE001 - skip unreadable globals
                continue
            shape = list(getattr(value, "shape", ())) if value is not None else []
            dtype = str(getattr(value, "dtype", type(value).__name__))
            entries.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "preview": _array_preview(value),
                }
            )
        return entries

    def describe_non_visual_bases(
        self, ref: SampleRef
    ) -> dict[str, list[dict[str, object]]]:
        """Return data arrays of CGNS bases that carry no zones.

        Some datasets store auxiliary tensors (constants, global reference
        values, look-up tables, ...) inside a CGNS base that has no
        ``Zone_t`` children, so VTK cannot render them as geometry. This
        method returns, for each zone-less base, a list of descriptors
        ``{"name": str, "shape": list[int], "dtype": str,
        "preview": str | None}`` suitable for display in the viewer.

        Args:
            ref: The sample to inspect.

        Returns:
            A mapping from base name to a list of data-array descriptors.
            Bases that do contain zones are omitted.
        """
        sample = self.load_sample(ref)
        times = self._time_keys(sample)
        if not times:
            return {}
        try:
            from CGNS.PAT import cgnskeywords as CK  # noqa: PLC0415
            from CGNS.PAT import cgnsutils as CU  # noqa: PLC0415
        except ImportError:  # pragma: no cover - defensive
            return {}
        tree = sample.features.data[times[0]]
        summary: dict[str, list[dict[str, object]]] = {}
        for base_node in (
            CU.hasChildType(tree, CK.CGNSBase_ts) or []
        ):  # pragma: no cover - CGNS tree introspection
            if CU.hasChildType(base_node, CK.Zone_ts):
                continue
            summary[base_node[0]] = _collect_data_arrays(base_node)
        return summary

    def get_sample_validation(self, ref: SampleRef) -> ValidationResult:
        """Check basic sample completeness using PLAID's built-in validator."""
        warnings: list[str] = []
        errors: list[str] = []
        try:
            sample = self.load_sample(ref)
        except Exception as exc:  # noqa: BLE001 - surface error to API caller
            return ValidationResult(
                ref=SampleRefDTO.from_ref(ref),
                ok=False,
                errors=[f"Failed to load sample: {exc}"],
            )
        try:
            report = sample.check_completeness()
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(
                ref=SampleRefDTO.from_ref(ref),
                ok=False,
                errors=[f"Completeness check failed: {exc}"],
            )
        ok = isinstance(report, str) and "error" not in report.lower()
        if report and not ok:
            errors.append(report)
        elif report:
            warnings.append(report)
        return ValidationResult(
            ref=SampleRefDTO.from_ref(ref),
            ok=ok,
            warnings=warnings,
            errors=errors,
        )

    # -------------------------------------------------------------- Helpers

    def _dataset_dir(self, dataset_id: str) -> Path:
        if self._datasets_root is None:
            raise FileNotFoundError(
                "No datasets root selected; call set_datasets_root first."
            )
        base = self._datasets_root / dataset_id
        if not base.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset_id}")
        return base

    def _is_hub_dataset(self, dataset_id: str) -> bool:
        """Return ``True`` when ``dataset_id`` refers to a registered HF repo."""
        return dataset_id in self._hub_repos

    def _open(self, dataset_id: str) -> tuple[dict, dict]:
        """Load (and cache) ``(dataset_dict, converter_dict)`` for a dataset.

        Dispatches between :func:`plaid.storage.init_from_disk` for local
        datasets and :func:`plaid.storage.init_streaming_from_hub` for
        registered Hugging Face Hub repositories.
        """
        if dataset_id in self._store_cache:
            return self._store_cache[dataset_id]
        if self._is_hub_dataset(dataset_id):
            # Deferred import so the module can be loaded without PLAID present.
            from plaid.storage import init_streaming_from_hub  # noqa: PLC0415
            from plaid.utils.cgns_helper import (  # noqa: PLC0415
                update_features_for_CGNS_compatibility,
            )

            features = self._features.get(dataset_id)
            # ``features is None`` means "no filter active" - let PLAID
            # materialise every feature, as before. An *empty* list is
            # a deliberate user choice ("show me only the geometry"):
            # we hand PLAID the union of every constant feature path
            # (so ``init_streaming_from_hub`` keeps the mesh and zone
            # metadata) and nothing else. Passing ``features=[]``
            # directly is not an option because PLAID's ``if features``
            # gate treats empty lists as "unfiltered".
            if features is None:
                datasetdict, converterdict = init_streaming_from_hub(dataset_id)
            else:
                constant_keys, variable_keys = self._load_feature_metadata(dataset_id)
                base_features = list(features) if features else list(constant_keys)
                expanded_features = update_features_for_CGNS_compatibility(
                    base_features, constant_keys, variable_keys
                )
                try:
                    datasetdict, converterdict = init_streaming_from_hub(
                        dataset_id, features=expanded_features
                    )
                except KeyError:
                    # ``expanded_features`` is derived from the
                    # dataset-wide metadata union and can therefore name
                    # paths that are not materialised as columns in a
                    # given split's HF table. The HF bridge then raises
                    # ``KeyError("Missing features in hf_dataset: …")``.
                    # Degrade gracefully to an unfiltered stream so the
                    # user still sees the geometry instead of a hard
                    # failure.
                    datasetdict, converterdict = init_streaming_from_hub(dataset_id)
        else:
            # Deferred import so the module can be loaded without PLAID present.
            from plaid.storage import init_from_disk  # noqa: PLC0415

            base = self._dataset_dir(dataset_id)
            datasetdict, converterdict = init_from_disk(str(base))
        # Normalise split-less case to a stable "__default__" key.
        if not datasetdict:
            raise RuntimeError(f"Dataset {dataset_id!r} is empty.")
        self._store_cache[dataset_id] = (datasetdict, converterdict)
        return datasetdict, converterdict

    def _splits_with_counts(self, dataset_id: str) -> dict[str, int | None]:
        """Return ``{split: len(ds)}``; ``None`` for streaming splits."""
        datasetdict, _ = self._open(dataset_id)
        counts: dict[str, int | None] = {}
        for split, ds in datasetdict.items():
            try:
                counts[split] = len(ds)
            except TypeError:
                counts[split] = None
        return counts

    @staticmethod
    def _load_infos(base: Path) -> dict | None:
        for candidate in (base / "infos.json", base / "infos.yaml", base / "infos.yml"):
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text()
            except OSError:
                return None
            if candidate.suffix == ".json":
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return None
            try:
                import yaml  # type: ignore  # noqa: PLC0415
            except ImportError:  # pragma: no cover - pyyaml is transitive
                return None
            try:
                return yaml.safe_load(text)
            except yaml.YAMLError:
                return None
        return None

    @staticmethod
    def _time_keys(sample) -> list[float]:
        data = getattr(sample.features, "data", None)
        if not data:
            return []
        return sorted(float(t) for t in data.keys())

    @staticmethod
    def _describe_tree(sample, times: list[float]):
        """Walk the CGNS tree of the first timestep and return bases, zones, fields."""
        bases: list[str] = []
        zones_by_base: dict[str, list[str]] = {}
        fields_by_base: dict[str, list[str]] = {}
        if not times:
            return bases, zones_by_base, fields_by_base
        tree = sample.features.data[times[0]]
        # Deferred import - CGNS helpers live inside pyCGNS.
        try:
            from CGNS.PAT import cgnskeywords as CK  # noqa: PLC0415
            from CGNS.PAT import cgnsutils as CU  # noqa: PLC0415
        except ImportError:  # pragma: no cover - defensive
            return bases, zones_by_base, fields_by_base
        for base_node in CU.hasChildType(tree, CK.CGNSBase_ts) or []:
            base_name = base_node[0]
            bases.append(base_name)
            zones_by_base[base_name] = []
            field_names: set[str] = set()
            for zone_node in CU.hasChildType(base_node, CK.Zone_ts) or []:
                zones_by_base[base_name].append(zone_node[0])
                for sol_node in CU.hasChildType(zone_node, CK.FlowSolution_ts) or []:
                    for da in CU.hasChildType(sol_node, CK.DataArray_ts) or []:
                        field_names.add(da[0])
            fields_by_base[base_name] = sorted(field_names)
        return bases, zones_by_base, fields_by_base


@lru_cache(maxsize=8)
def _cached_service(root: str, backend_id: str) -> PlaidDatasetService:
    return PlaidDatasetService(
        ViewerConfig(datasets_root=Path(root), backend_id=backend_id)
    )
