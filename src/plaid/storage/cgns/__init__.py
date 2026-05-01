"""Package for CGNS storage."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

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

from typing import Any, Union
from pathlib import Path


class CgnsBackend():
    name = "cgns"

    def init_from_disk(self, path: Union[str, Path]) -> Any: 
        return init_datasetdict_from_disk( path= path)
    
    def download_from_hub (self, repo_id: str, local_dir: Union[str, Path]) -> str: 
        return download_datasetdict_from_hub(repo_id, local_dir)
    
    def init_streaming_from_hub(self, repo_id: str) -> dict[str, Any]:
        return init_datasetdict_streaming_from_hub(self, output_folder)
    
    def generate_to_disk(self, output_folder: Union[str, Path]) -> None:
        return generate_datasetdict_to_disk(self, output_folder)
    
    def push_local_to_hub(self, repo_id: str, local_dir: Union[str, Path]) -> None:
        return push_local_datasetdict_to_hub(self, repo_id, local_dir= local_dir)
    
    def get_configure_dataset_card(self) -> None:
        return configure_dataset_card
    
    def to_var_sample_dict(self):
        pass
    
    def sample_to_var_sample_dict(self):
        pass 

__all__ = [
    "configure_dataset_card",
    "download_datasetdict_from_hub",
    "generate_datasetdict_to_disk",
    "init_datasetdict_from_disk",
    "init_datasetdict_streaming_from_hub",
    "push_local_datasetdict_to_hub",
]
