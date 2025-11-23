import zarr

def unflatten_zarr_key(key: str) -> str:
    return key.replace("__", "/")

def to_var_sample_dict(
    zarr_dataset: zarr.core.group.Group,
    i: int
) -> dict:
    """Convert a Hugging Face dataset row to a PLAID Sample object.

    Extracts a single row from a Hugging Face dataset and converts it
    into a PLAID Sample by unflattening the CGNS tree structure. Constant features
    from flat_cst are merged with the variable features from the row.

    Args:
        ds (datasets.Dataset): The Hugging Face dataset containing the sample data.
        i (int): The index of the row to convert.
        flat_cst (dict[str, Any]): Dictionary of constant features to add to each sample.
        cgns_types (dict[str, str]): Dictionary mapping paths to CGNS types for reconstruction.
        enforce_shapes (bool, optional): If True, ensures consistent array shapes during conversion. Defaults to True.

    Returns:
        Sample: A validated PLAID Sample object reconstructed from the Hugging Face dataset row.

    Note:
        - Uses the dataset's pyarrow table data for efficient access.
        - Handles array shapes and types according to enforce_shapes.
        - Constant features from flat_cst are merged with the variable features from the row.
    """
    zarr_sample = zarr_dataset[f"sample_{i:09d}"]
    return {unflatten_zarr_key(path):zarr_sample[path] for path in zarr_sample.array_keys()}