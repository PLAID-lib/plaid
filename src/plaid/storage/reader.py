from pathlib import Path
from typing import Optional, Union

from plaid.storage.cgns.reader import (
    download_datasetdict_from_hub as download_cgns_datasetdict_from_hub,
)
from plaid.storage.cgns.reader import (
    init_datasetdict_from_disk as init_cgns_datasetdict_from_disk,
)
from plaid.storage.cgns.reader import (
    init_datasetdict_streaming_from_hub as init_cgns_datasetdict_streaming_from_hub,
)
from plaid.storage.common.bridge import (
    plaid_to_sample_dict,
    to_plaid_sample,
    to_sample_dict,
)
from plaid.storage.common.reader import (
    load_infos_from_disk,
    load_infos_from_hub,
    load_metadata_from_disk,
    load_metadata_from_hub,
    load_problem_definitions_from_hub,
)
from plaid.storage.common.writer import (
    save_infos_to_disk,
    save_metadata_to_disk,
    save_problem_definitions_to_disk,
)
from plaid.storage.hf_datasets.bridge import (
    sample_to_var_sample_dict as hf_sample_to_var_sample_dict,
)
from plaid.storage.hf_datasets.bridge import (
    to_var_sample_dict as hf_to_var_sample_dict,
)
from plaid.storage.hf_datasets.reader import (
    download_datasetdict_from_hub as download_hf_datasetdict_from_hub,
)
from plaid.storage.hf_datasets.reader import (
    init_datasetdict_from_disk as init_hf_datasetdict_from_disk,
)
from plaid.storage.hf_datasets.reader import (
    init_datasetdict_streaming_from_hub as init_hf_datasetdict_streaming_from_hub,
)
from plaid.storage.zarr.bridge import (
    sample_to_var_sample_dict as zarr_sample_to_var_sample_dict,
)
from plaid.storage.zarr.bridge import (
    to_var_sample_dict as zarr_to_var_sample_dict,
)
from plaid.storage.zarr.reader import (
    download_datasetdict_from_hub as download_zarr_datasetdict_from_hub,
)
from plaid.storage.zarr.reader import (
    init_datasetdict_from_disk as init_zarr_datasetdict_from_disk,
)
from plaid.storage.zarr.reader import (
    init_datasetdict_streaming_from_hub as init_zarr_datasetdict_streaming_from_hub,
)

init_datasetdict_from_disk = {
    "hf_datasets": init_hf_datasetdict_from_disk,
    "cgns": init_cgns_datasetdict_from_disk,
    "zarr": init_zarr_datasetdict_from_disk,
}

download_datasetdict_from_hub = {
    "hf_datasets": download_hf_datasetdict_from_hub,
    "cgns": download_cgns_datasetdict_from_hub,
    "zarr": download_zarr_datasetdict_from_hub,
}

init_datasetdict_streaming_from_hub = {
    "hf_datasets": init_hf_datasetdict_streaming_from_hub,
    "cgns": init_cgns_datasetdict_streaming_from_hub,
    "zarr": init_zarr_datasetdict_streaming_from_hub,
}

to_var_sample_dict = {
    "hf_datasets": hf_to_var_sample_dict,
    "zarr": zarr_to_var_sample_dict,
}

sample_to_var_sample_dict = {
    "hf_datasets": hf_sample_to_var_sample_dict,
    "zarr": zarr_sample_to_var_sample_dict,
}


class Converter:
    def __init__(self, backend, flat_cst, cgns_types, variable_schema, constant_schema):
        self.backend = backend
        self.flat_cst = flat_cst
        self.cgns_types = cgns_types
        self.variable_schema = variable_schema
        self.constant_schema = constant_schema

    def to_dict(self, dataset, idx):
        if self.backend == "cgns":
            raise ValueError("Converter.to_dict not available for cgns backend")
        var_sample_dict = to_var_sample_dict[self.backend](dataset, idx)
        return to_sample_dict(var_sample_dict, self.flat_cst, self.cgns_types)

    def to_plaid(self, dataset, idx):
        if self.backend != "cgns":
            sample_dict = self.to_dict(dataset, idx)
            return to_plaid_sample(sample_dict, self.cgns_types)
        else:
            return dataset[idx]

    def sample_to_dict(self, sample):
        if self.backend == "cgns":
            raise ValueError("Converter.sample_to_dict not available for cgns backend")
        var_sample_dict = sample_to_var_sample_dict[self.backend](sample)
        return to_sample_dict(var_sample_dict, self.flat_cst, self.cgns_types)

    def sample_to_plaid(self, sample):
        if self.backend != "cgns":
            sample_dict = self.sample_to_dict(sample)
            return to_plaid_sample(sample_dict, self.cgns_types)
        else:
            return sample

    def plaid_to_dict(self, plaid_sample):
        return plaid_to_sample_dict(
            plaid_sample, self.variable_schema, self.constant_schema
        )

    def __repr__(self) -> str:
        return f"Converter(backend={self.backend})"


def init_from_disk(local_dir: Union[Path, str]):
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_disk(
        local_dir
    )
    infos = load_infos_from_disk(local_dir)

    backend = infos["storage_backend"]

    datasetdict = init_datasetdict_from_disk[backend](local_dir)

    converterdict = {}
    for split in datasetdict.keys():
        converterdict[split] = Converter(
            backend, flat_cst[str(split)], cgns_types, variable_schema, constant_schema
        )
    return datasetdict, converterdict


def download_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
    overwrite: bool = False,
):  # pragma: no cover
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_hub(
        repo_id
    )
    infos = load_infos_from_hub(repo_id)
    pb_defs = load_problem_definitions_from_hub(repo_id)

    backend = infos["storage_backend"]

    download_datasetdict_from_hub[backend](
        repo_id, local_dir, split_ids, features, overwrite
    )

    save_metadata_to_disk(
        local_dir, flat_cst, variable_schema, constant_schema, cgns_types
    )
    save_infos_to_disk(local_dir, infos)
    if pb_defs is not None:
        save_problem_definitions_to_disk(local_dir, pb_defs)


def init_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
):  # pragma: no cover
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_hub(
        repo_id
    )
    infos = load_infos_from_hub(repo_id)

    backend = infos["storage_backend"]

    datasetdict = init_datasetdict_streaming_from_hub[backend](
        repo_id, split_ids, features
    )

    converterdict = {}
    for split in datasetdict.keys():
        converterdict[split] = Converter(
            backend, flat_cst[str(split)], cgns_types, variable_schema, constant_schema
        )

    return datasetdict, converterdict
