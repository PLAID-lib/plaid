import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

from packaging.version import Version

import plaid
from plaid import ProblemDefinition, Sample

from plaid.storage.common.preprocessor import preprocess
from plaid.storage.common.reader import (
    load_infos_from_disk,
    load_metadata_from_disk,
    load_problem_definitions_from_disk,
)
from plaid.storage.common.writer import (
    _check_folder,
    push_infos_to_hub,
    push_metadata_to_hub,
    push_problem_definitions_to_hub,
    save_infos_to_disk,
    save_metadata_to_disk,
    save_problem_definitions_to_disk,
)

from plaid.storage.hf_datasets.bridge import (
    generator_to_datasetdict as generator_to_hf_datasetdict,
)
from plaid.storage.hf_datasets.reader import (
    init_datasetdict_from_disk as init_hf_datasetdict_from_disk,
)
from plaid.storage.hf_datasets.writer import (
    generate_datasetdict_to_disk as generate_hf_datasetdict_to_disk,
    push_local_datasetdict_to_hub as push_local_hf_datasetdict_to_hub,
    configure_dataset_card as configure_hf_dataset_card,
)

from plaid.storage.zarr.writer import (
    generate_datasetdict_to_disk as generate_zarr_datasetdict_to_disk,
    push_local_datasetdict_to_hub as push_local_zarr_datasetdict_to_hub,
    configure_dataset_card as configure_zarr_dataset_card,
)

from plaid.storage.cgns.writer import (
    generate_datasetdict_to_disk as generate_cgns_datasetdict_to_disk,
    push_local_datasetdict_to_hub as push_local_cgns_datasetdict_to_hub,
    configure_dataset_card as configure_cgns_dataset_card,
)

logger = logging.getLogger(__name__)

AVAILABLE_BACKENDS = ["cgns", "hf_datasets", "zarr"]

generate_datasetdict_to_disk = {
    "hf_datasets": generate_hf_datasetdict_to_disk,
    "zarr": generate_zarr_datasetdict_to_disk,
    "cgns": generate_cgns_datasetdict_to_disk,
}

push_local_datasetdict_to_hub = {
    "hf_datasets": push_local_hf_datasetdict_to_hub,
    "zarr": push_local_zarr_datasetdict_to_hub,
    "cgns": push_local_cgns_datasetdict_to_hub,
}

configure_dataset_card = {
    "hf_datasets": configure_hf_dataset_card,
    "zarr": configure_zarr_dataset_card,
    "cgns": configure_cgns_dataset_card,
}

def save_to_disk(
    output_folder: Union[str, Path],
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    backend: str,
    infos: Optional[dict[str, Any]] = None,
    pb_defs: Optional[Union[ProblemDefinition, Iterable[ProblemDefinition]]] = None,
    gen_kwargs: Optional[dict[str, dict[str, Any]]] = None,
    num_proc: int = 1,
    verbose: bool = False,
    overwrite: bool = False,
) -> None:
    assert backend in AVAILABLE_BACKENDS, (
        f"backend {backend} not among available ones: {AVAILABLE_BACKENDS}"
    )

    output_folder = Path(output_folder)

    _check_folder(output_folder, overwrite)

    flat_cst, variable_schema, constant_schema, split_n_samples, cgns_types = preprocess(
        generators, gen_kwargs=gen_kwargs, num_proc=num_proc, verbose=verbose
    )

    generate_datasetdict_to_disk[backend](
            output_folder,
            generators,
            variable_schema,
            gen_kwargs=gen_kwargs,
            num_proc=num_proc,
            verbose=verbose,
    )

    save_metadata_to_disk(output_folder, flat_cst, variable_schema, constant_schema, cgns_types)

    infos = infos.copy() if infos else {}
    infos.setdefault("num_samples", split_n_samples)
    infos.setdefault("storage_backend", backend)
    infos.setdefault("plaid", {"version": str(Version(plaid.__version__))})

    save_infos_to_disk(output_folder, infos)

    if pb_defs is not None:
        save_problem_definitions_to_disk(output_folder, pb_defs)


def push_to_hub(repo_id: str, local_dir: Union[str, Path], num_workers: int = 1, viewer: bool = False) -> None:
    pb_defs = load_problem_definitions_from_disk(local_dir)
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_disk(local_dir)
    infos = load_infos_from_disk(local_dir)

    backend = infos["storage_backend"]

    push_local_datasetdict_to_hub[backend](repo_id, local_dir, num_workers=num_workers)
    configure_dataset_card[backend](repo_id, infos, local_dir, variable_schema, viewer)

    push_metadata_to_hub(repo_id, flat_cst, variable_schema, constant_schema, cgns_types)
    push_infos_to_hub(repo_id, infos)
    if pb_defs is not None:
        push_problem_definitions_to_hub(repo_id, pb_defs)
