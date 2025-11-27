import zarr

def unflatten_zarr_key(key: str) -> str:
    return key.replace("__", "/")

def to_var_sample_dict(
    zarr_dataset: zarr.core.group.Group,
    i: int
) -> dict:
    zarr_sample = zarr_dataset[f"sample_{i:09d}"]
    return {unflatten_zarr_key(path):zarr_sample[path] for path in zarr_sample.array_keys()}