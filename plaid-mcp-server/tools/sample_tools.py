"""
Sample Tools for PLAID MCP Server

Provides tools for accessing and manipulating PLAID samples.
"""

from typing import Any, Dict, List, Optional


class SampleTools:
    """Tools for PLAID sample operations."""

    def __init__(self, session_manager):
        self.session = session_manager

    async def get_sample(
        self,
        dataset_id: str,
        split_name: str,
        sample_idx: int,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve a sample from a loaded dataset."""
        try:
            dataset = self.session.get_dataset(dataset_id)
            if not dataset:
                return {"error": f"Dataset not found: {dataset_id}"}

            if split_name not in dataset.datasetdict:
                return {"error": f"Split not found: {split_name}"}

            split_dataset = dataset.datasetdict[split_name]
            converter = dataset.converterdict[split_name]

            if sample_idx < 0 or sample_idx >= len(split_dataset):
                return {
                    "error": f"Sample index {sample_idx} out of range [0, {len(split_dataset)}]"
                }

            # Get the sample as a PLAID Sample object
            sample = converter.to_plaid(split_dataset, sample_idx, features=features)

            # Return sample summary (can't easily serialize full Sample)
            return {
                "success": True,
                "dataset_id": dataset_id,
                "split_name": split_name,
                "sample_idx": sample_idx,
                "sample_summary": sample.summarize(),
                "scalar_names": sample.get_scalar_names(),
                "field_names": sample.get_field_names(),
                "message": "Sample retrieved successfully. Use sample_summary for details.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to retrieve sample",
            }

    async def get_sample_info(
        self, dataset_id: str, split_name: str, sample_idx: int
    ) -> Dict[str, Any]:
        """Get metadata about a sample without loading all data."""
        try:
            dataset = self.session.get_dataset(dataset_id)
            if not dataset:
                return {"error": f"Dataset not found: {dataset_id}"}

            if split_name not in dataset.datasetdict:
                return {"error": f"Split not found: {split_name}"}

            split_dataset = dataset.datasetdict[split_name]
            converter = dataset.converterdict[split_name]

            if sample_idx < 0 or sample_idx >= len(split_dataset):
                return {"error": f"Sample index {sample_idx} out of range"}

            # Get sample and basic info
            sample = converter.to_plaid(split_dataset, sample_idx)

            return {
                "success": True,
                "dataset_id": dataset_id,
                "split_name": split_name,
                "sample_idx": sample_idx,
                "scalars": sample.get_scalar_names(),
                "fields": sample.get_field_names(),
                "feature_identifiers": [
                    str(fi) for fi in sample.get_all_features_identifiers()
                ],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
