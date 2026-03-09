"""
Storage Tools for PLAID MCP Server

Provides tools for loading/saving PLAID datasets and accessing storage operations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from plaid.storage import init_from_disk, download_from_hub


class StorageTools:
    """Tools for PLAID storage operations."""
    
    def __init__(self, session_manager):
        self.session = session_manager
        
    async def init_from_disk(
        self,
        local_dir: str,
        splits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load a PLAID dataset from local disk."""
        try:
            local_path = Path(local_dir)
            if not local_path.exists():
                return {"error": f"Directory not found: {local_dir}"}
            
            # Load dataset and converters
            datasetdict, converterdict = init_from_disk(local_dir, splits=splits)
            
            # Store in session
            dataset_id = self.session.add_dataset(
                local_dir=local_dir,
                datasetdict=datasetdict,
                converterdict=converterdict,
                metadata={
                    "splits": list(datasetdict.keys()),
                    "backend": converterdict[list(converterdict.keys())[0]].backend if converterdict else "unknown"
                }
            )
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "local_dir": local_dir,
                "splits": list(datasetdict.keys()),
                "num_samples_per_split": {
                    split: len(dataset) for split, dataset in datasetdict.items()
                },
                "message": f"Dataset loaded successfully with ID: {dataset_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to load dataset from {local_dir}"
            }
    
    async def download_from_hub(
        self,
        repo_id: str,
        local_dir: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Download a PLAID dataset from HuggingFace Hub."""
        try:
            download_from_hub(
                repo_id=repo_id,
                local_dir=local_dir,
                overwrite=overwrite
            )
            
            return {
                "success": True,
                "repo_id": repo_id,
                "local_dir": local_dir,
                "message": f"Dataset downloaded from {repo_id} to {local_dir}. Use init_from_disk to load it."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to download dataset from {repo_id}"
            }
    
    async def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get information about a loaded dataset."""
        dataset = self.session.get_dataset(dataset_id)
        if not dataset:
            return {"error": f"Dataset not found: {dataset_id}"}
        
        info = {
            "dataset_id": dataset_id,
            "local_dir": dataset.local_dir,
            "splits": list(dataset.datasetdict.keys()),
            "backend": dataset.metadata.get("backend", "unknown"),
            "splits_info": {}
        }
        
        # Get info for each split
        for split_name, split_dataset in dataset.datasetdict.items():
            converter = dataset.converterdict.get(split_name)
            info["splits_info"][split_name] = {
                "num_samples": len(split_dataset),
                "variable_features": list(converter.variable_features) if converter else [],
                "constant_features": list(converter.constant_features) if converter else []
            }
        
        return info
    
    async def list_loaded_datasets(self) -> Dict[str, Any]:
        """List all currently loaded datasets."""
        datasets_info = self.session.list_datasets()
        return {
            "num_datasets": len(datasets_info),
            "datasets": datasets_info
        }
