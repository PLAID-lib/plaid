import zarr


def unflatten_zarr_key(key: str) -> str:
    return key.replace("__", "/")


def to_var_sample_dict(zarr_dataset: zarr.core.group.Group, idx: int) -> dict:
    return zarr_dataset[idx]


def sample_to_var_sample_dict(zarr_sample: dict) -> dict:
    return zarr_sample
