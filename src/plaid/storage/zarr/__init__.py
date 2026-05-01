"""Package for Zarr storage."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
from typing import Any, Union
from pathlib import Path

from .bridge import (
    sample_to_var_sample_dict,
    to_var_sample_dict,
)
from .reader import (
    download_datasetdict_from_hub,
    init_datasetdict_from_disk,
    init_datasetdict_streaming_from_hub,
)
from .writer import (
    configure_dataset_card,
    generate_datasetdict_to_disk,
    push_local_datasetdict_to_hub,
)

class ZarrBackend():
    name = "zarr"

    @staticmethod
    def init_from_disk( path: Union[str, Path]) -> Any: 
        return init_datasetdict_from_disk(path=path)
    
    def download_from_hub (self, repo_id: str, local_dir: Union[str, Path]) -> str: 
        return  download_datasetdict_from_hub(repo_id=repo_id, local_dir=local_dir)
    
    def init_datasetdict_streaming_from_hub(self, repo_id: str) -> dict[str, Any]:
        return init_datasetdict_streaming_from_hub(repo_id=repo_id)
    
    @staticmethod 
    def generate_to_disk(output_folder: Union[str, Path], generators, variable_schema,gen_kwargs,num_proc,verbose) -> None:
        return generate_datasetdict_to_disk(output_folder = output_folder,
                                            generators=generators,
                                            variable_schema=variable_schema,
                                            gen_kwargs=gen_kwargs,
                                            num_proc=num_proc,
                                            verbose=verbose
                                            )
    
    def push_local_datasetdict_to_hub(self, repo_id: str, local_dir: Union[str, Path]) -> None:
        return push_local_datasetdict_to_hub(repo_id=repo_id, local_dir=local_dir)
    
    def configure_dataset_card(self, repo_id: str, infos: dict) -> None:
        return configure_dataset_card(repo_id=repo_id, infos=infos)

    @staticmethod
    def to_var_sample_dict( dataset, idx, features):
        return to_var_sample_dict(zarr_dataset = dataset, idx=idx, features=features)

    @staticmethod
    def sample_to_var_sample_dict(sample):
        return sample_to_var_sample_dict(zarr_sample=sample) 
        

__all__ = [
    "configure_dataset_card",
    "download_datasetdict_from_hub",
    "generate_datasetdict_to_disk",
    "init_datasetdict_from_disk",
    "init_datasetdict_streaming_from_hub",
    "push_local_datasetdict_to_hub",
    "sample_to_var_sample_dict",
    "to_var_sample_dict",
]
