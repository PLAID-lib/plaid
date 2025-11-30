import zarr

from plaid.storage.zarr.reader import _LazyZarrArray


def unflatten_zarr_key(key: str) -> str:
    return key.replace("__", "/")


def to_var_sample_dict(zarr_dataset: zarr.core.group.Group, i: int) -> dict:
    zarr_sample = zarr_dataset[f"sample_{i:09d}"]
    return {
        unflatten_zarr_key(path): zarr_sample[path] for path in zarr_sample.array_keys()
    }


def to_var_sample_dict_streamed(
    zarr_dataset: dict[int, dict[str, _LazyZarrArray]], i: int
) -> dict:
    return {unflatten_zarr_key(path): value for path, value in zarr_dataset[i].items()}
