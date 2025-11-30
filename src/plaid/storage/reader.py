from pathlib import Path
from typing import Any, Optional, Union

from plaid import Sample
from plaid.storage.cgns.reader import (
    download_datasetdict_from_hub as download_cgns_datasetdict_from_hub,
)

# CGNS
from plaid.storage.cgns.reader import (
    init_datasetdict_from_disk as init_cgns_datasetdict_from_disk,
)
from plaid.storage.cgns.reader import (
    init_datasetdict_streaming_from_hub as init_cgns_datasetdict_streaming_from_hub,
)
from plaid.storage.common.bridge import to_sample_dict, to_plaid_sample

# COMMON
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

# HF_DATASETS
from plaid.storage.hf_datasets.bridge import to_var_sample_dict as to_hf_var_sample_dict
from plaid.storage.hf_datasets.bridge import (
    to_var_sample_dict_streamed as to_hf_var_sample_dict_streamed,
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

# ZARR
from plaid.storage.zarr.bridge import to_var_sample_dict as to_zarr_var_sample_dict
from plaid.storage.zarr.bridge import (
    to_var_sample_dict_streamed as to_zarr_var_sample_dict_streamed,
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


def _initialize_cgns_types(
        variable_schema: dict[str, Any],
        constant_schema: dict[str, Any])->dict[str, Any]:

    cgns_types = {}
    for path in variable_schema.keys():
        cgns_types[path] = variable_schema[path]["cgns_type"]
    for path in constant_schema.keys():
        cgns_types[path] = constant_schema[path]["cgns_type"]
    cgns_types[path]


class _SampleCreator:
    def __init__(
        self,
        backend: str,
        dataset,
        flat_cst: dict[str, Any],
        cgns_types: dict[str, Any],
        features: Optional[list[str]] = None,
    ):
        self.backend = backend
        self.dataset = dataset
        self.flat_cst = flat_cst
        self.cgns_types = cgns_types
        self.features = features

    def __getitem__(self, index: int) -> Sample:
        if self.backend == "hf_datasets":
            var_sample_dict = to_hf_var_sample_dict(self.dataset, index)
            sample_dict = to_sample_dict(
                var_sample_dict, self.flat_cst, self.cgns_types, self.features
            )
            sample = to_plaid_sample(sample_dict, self.cgns_types)
        elif self.backend == "zarr":
            var_sample_dict = to_zarr_var_sample_dict(self.dataset, index)
            sample_dict = to_sample_dict(
                var_sample_dict, self.flat_cst, self.cgns_types, self.features
            )
            sample = to_plaid_sample(sample_dict, self.cgns_types)
        elif self.backend == "cgns":
            sample = self.dataset[index]()
        else:
            raise ValueError(f"backend {self.backend} not implemented")
        return sample

    def __len__(self)->int:
        return len(self.dataset)



class _SampleDictCreator:
    def __init__(
        self,
        backend: str,
        dataset,
        flat_cst: dict[str, Any],
        cgns_types: dict[str, Any],
        features: Optional[list[str]] = None,
    ):
        self.backend = backend
        self.dataset = dataset
        self.flat_cst = flat_cst
        self.cgns_types = cgns_types
        self.features = features

    def __getitem__(self, index: int) -> dict[float, dict[str, Any]]:
        if self.backend == "hf_datasets":
            var_sample_dict = to_hf_var_sample_dict(self.dataset, index)
            sample_dict = to_sample_dict(
                var_sample_dict, self.flat_cst, self.cgns_types, self.features
            )
        elif self.backend == "zarr":
            var_sample_dict = to_zarr_var_sample_dict(self.dataset, index)
            sample_dict = to_sample_dict(
                var_sample_dict, self.flat_cst, self.cgns_types, self.features
            )
        else:
            raise ValueError(f"backend {self.backend} not implemented")
        return sample_dict

    def __len__(self)->int:
        return len(self.dataset)


def init_from_disk(local_dir: Union[Path, str], sample_type:str = "plaid", features: Optional[list[str]] = None)->Union[dict[str, _SampleCreator], dict[str, _SampleDictCreator]]:

    assert sample_type in ["plaid", "dict"], f"sample type {sample_type} not available"

    flat_cst, _, _, cgns_types = load_metadata_from_disk(local_dir)
    infos = load_infos_from_disk(local_dir)

    backend = infos["storage_backend"]

    if backend == "hf_datasets":
        datasetdict = init_hf_datasetdict_from_disk(local_dir)
    elif backend == "zarr":
        datasetdict = init_zarr_datasetdict_from_disk(local_dir)
    elif backend == "cgns":
        datasetdict = init_cgns_datasetdict_from_disk(local_dir)
    else:
        raise ValueError(f"backend {backend} not implemented")

    if sample_type == "plaid":
        sample_creator = {
            split: _SampleCreator(
                backend,
                datasetdict[str(split)],
                flat_cst[str(split)],
                cgns_types,
                features,
            )
            for split in datasetdict.keys()
        }
    elif sample_type == "dict":
        sample_creator = {
            split: _SampleDictCreator(
                backend,
                datasetdict[str(split)],
                flat_cst[str(split)],
                cgns_types,
                features,
            )
            for split in datasetdict.keys()
        }
    else:
        raise ValueError(f"sample type {sample_type} not implemented")
    return sample_creator


def download_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
    overwrite: bool = False,
):
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_hub(repo_id)
    infos = load_infos_from_hub(repo_id)
    pb_defs = load_problem_definitions_from_hub(repo_id)

    backend = infos["storage_backend"]

    if backend == "hf_datasets":
        download_hf_datasetdict_from_hub(repo_id, local_dir, overwrite)
    elif backend == "zarr":
        download_zarr_datasetdict_from_hub(
            repo_id, local_dir, split_ids, features, overwrite
        )
    elif backend == "cgns":
        download_cgns_datasetdict_from_hub(repo_id, local_dir, split_ids, overwrite)
    else:
        raise ValueError(f"backend {backend} not implemented")

    save_metadata_to_disk(local_dir, flat_cst, variable_schema, constant_schema, cgns_types)
    save_infos_to_disk(local_dir, infos)
    if pb_defs is not None:
        save_problem_definitions_to_disk(local_dir, pb_defs)


class _StreamedSampleCreator:
    def __init__(
        self,
        backend: str,
        dataset,
        flat_cst: dict[str, Any],
        cgns_types: dict[str, Any],
        features: Optional[list[str]] = None,
    ):
        self.backend = backend
        self.dataset = dataset
        self.flat_cst = flat_cst
        self.cgns_types = cgns_types
        self.features = features

    def __getitem__(self, index: int) -> Sample:
        if self.backend == "zarr":
            var_sample_dict = to_zarr_var_sample_dict_streamed(self.dataset, index)
            sample_dict = to_sample_dict(
                var_sample_dict, self.flat_cst, self.cgns_types, self.features
            )
            sample = to_plaid_sample(sample_dict, self.cgns_types)
        elif self.backend == "cgns":
            sample = self.dataset[index]()
        else:
            raise ValueError(f"backend {self.backend} not implemented")
        return sample


class _StreamedIterableSampleCreator:
    def __init__(
        self,
        backend: str,
        dataset,
        flat_cst: dict[str, Any],
        cgns_types: dict[str, Any],
        features: Optional[list[str]] = None,
    ):
        assert backend == "hf_datasets", (
            f"only compatible with backend = hf_datasets (called with backend = {backend})"
        )
        self.backend = backend
        self.dataset = iter(dataset)
        self.flat_cst = flat_cst
        self.cgns_types = cgns_types
        self.features = features

    def __call__(self) -> Sample:
        var_sample_dict = to_hf_var_sample_dict_streamed(next(self.dataset))
        sample_dict = to_sample_dict(
            var_sample_dict, self.flat_cst, self.cgns_types, self.features
        )
        sample = to_plaid_sample(
            sample_dict, self.cgns_types
        )
        return sample


def init_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
):
    flat_cst, _, _, cgns_types = load_metadata_from_hub(repo_id)
    infos = load_infos_from_hub(repo_id)

    backend = infos["storage_backend"]

    if backend == "hf_datasets":
        datasetdict = init_hf_datasetdict_streaming_from_hub(repo_id, features)
        sample_creator = {
            split: _StreamedIterableSampleCreator(
                backend,
                datasetdict[str(split)],
                flat_cst[str(split)],
                cgns_types,
                features,
            )
            for split in datasetdict.keys()
        }
    elif backend == "zarr":
        datasetdict = init_zarr_datasetdict_streaming_from_hub(
            repo_id, split_ids, features
        )
        sample_creator = {
            split: _StreamedSampleCreator(
                backend,
                datasetdict[str(split)],
                flat_cst[str(split)],
                cgns_types,
                features,
            )
            for split in datasetdict.keys()
        }
    elif backend == "cgns":
        datasetdict = init_cgns_datasetdict_streaming_from_hub(repo_id, split_ids)
        sample_creator = {
            split: _StreamedSampleCreator(
                backend,
                datasetdict[str(split)],
                flat_cst[str(split)],
                cgns_types,
                features,
            )
            for split in datasetdict.keys()
        }
    else:
        raise ValueError(f"backend {backend} not implemented")

    return sample_creator
