from functools import partial
from typing import Callable, Generator, Optional

import datasets
import numpy as np
import pyarrow as pa
from datasets import Features, Sequence, Value

from plaid import Dataset, Sample
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.storage.common.bridge import to_sample_dict, to_plaid_sample
from plaid.types import IndexType



# class DictView:
#     def __init__(self, dataset, flat_cst, cgns_types):
#         self.dataset = dataset
#         self.flat_cst = flat_cst
#         self.cgns_types = cgns_types

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         var_sample_dict = to_var_sample_dict(self.dataset, idx)
#         return to_sample_dict(
#             var_sample_dict,
#             self.flat_cst,
#             self.cgns_types,
#         )

# class PLAIDView:
#     def __init__(self, dataset, flat_cst, cgns_types):
#         self.dataset = dataset
#         self.flat_cst = flat_cst
#         self.cgns_types = cgns_types

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         var_sample_dict = to_var_sample_dict(self.dataset, idx)
#         sample_dict = to_sample_dict(
#             var_sample_dict,
#             self.flat_cst,
#             self.cgns_types,
#         )
#         return to_plaid_sample(sample_dict, self.cgns_types)



def convert_dtype_to_hf_feature(feature_type):
    """Convert a dict {'dtype': ..., 'ndim': ...} into HF Feature/Sequence."""
    base_dtype = feature_type["dtype"]
    ndim = feature_type["ndim"]

    # Map numpy dtype to HF Value type
    if base_dtype.startswith("float") or base_dtype.startswith("int"):
        value_type = Value(base_dtype)
    elif base_dtype.startswith("<U") or base_dtype.startswith("|S"):
        value_type = Value("string")
    else:
        raise ValueError(f"Unsupported dtype: {base_dtype}")

    # Wrap in Sequence according to ndim
    feature = value_type
    for _ in range(ndim):
        feature = Sequence(feature)
    return feature


def convert_to_hf_feature(variable_schema):
    return Features(
        {k: convert_dtype_to_hf_feature(v) for k, v in variable_schema.items()}
    )


def plaid_dataset_to_datasetdict(
    dataset: Dataset,
    main_splits: dict[str, IndexType],
    var_features_types: dict[str, dict],
    processes_number: int = 1,
    writer_batch_size: int = 1,
) -> datasets.DatasetDict:
    """Convert a PLAID dataset into a Hugging Face `datasets.DatasetDict`.

    This is a thin wrapper that creates per-split generators from a PLAID dataset
    and delegates the actual dataset construction to
    `plaid_generator_to_datasetdict`.

    Args:
        dataset (plaid.Dataset):
            The PLAID dataset to be converted. Must support indexing with
            a list of IDs (from `main_splits`).
        main_splits (dict[str, IndexType]):
            Mapping from split names (e.g. "train", "test") to the subset of
            sample indices belonging to that split.
        processes_number (int, optional, default=1):
            Number of parallel processes to use when writing the Hugging Face dataset.
        writer_batch_size (int, optional, default=1):
            Batch size used when writing samples to disk in Hugging Face format.
        verbose (bool, optional, default=False):
            If True, print progress and debug information.

    Returns:
        datasets.DatasetDict:
            A Hugging Face `DatasetDict` containing one dataset per split.

    Example:
        >>> ds_dict = plaid_dataset_to_huggingface_datasetdict(
        ...     dataset=my_plaid_dataset,
        ...     main_splits={"train": [0, 1, 2], "test": [3]},
        ...     processes_number=4,
        ...     writer_batch_size=3
        ... )
        >>> print(ds_dict)
        DatasetDict({
            train: Dataset({
                features: ...
            }),
            test: Dataset({
                features: ...
            })
        })
    """

    def generator(dataset):
        for sample in dataset:
            yield sample

    generators = {
        split_name: partial(generator, dataset[ids])
        for split_name, ids in main_splits.items()
    }

    return generator_to_datasetdict(
        generators,
        var_features_types,
        processes_number=processes_number,
        writer_batch_size=writer_batch_size,
    )


def generator_to_datasetdict(
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    variable_schema: dict,
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    processes_number: int = 1,
    writer_batch_size: int = 1,
) -> datasets.DatasetDict:
    """Convert PLAID dataset generators into a Hugging Face `datasets.DatasetDict`.

    This function inspects samples produced by the given generators, flattens their
    CGNS tree structure, infers Hugging Face feature types, and builds one
    `datasets.Dataset` per split. Constant features (identical across all samples)
    are separated out from variable features.

    Args:
        generators (dict[str, Callable]):
            Mapping from split names (e.g., "train", "test") to generator functions.
            Each generator function must return an iterable of PLAID samples, where
            each sample provides `sample.features.data[0.0]` for flattening.
        processes_number (int, optional, default=1):
            Number of processes used internally by Hugging Face when materializing
            the dataset from the generators.
        writer_batch_size (int, optional, default=1):
            Batch size used when writing samples to disk in Hugging Face format.
        gen_kwargs (dict, optional, default=None):
            Optional mapping from split names to dictionaries of keyword arguments
            to be passed to each generator function, used for parallelization.
        verbose (bool, optional, default=False):
            If True, displays progress bars and diagnostic messages.

    Returns:
        tuple:
            - **DatasetDict** (`datasets.DatasetDict`):
              A Hugging Face dataset dictionary with one dataset per split.
            - **flat_cst** (`dict[str, Any]`):
              Dictionary of constant features detected across all splits.
            - **key_mappings** (`dict[str, Any]`):
              Metadata dictionary containing:
                - `"variable_features"`: list of paths for non-constant features.
                - `"constant_features"`: list of paths for constant features.
                - `"cgns_types"`: inferred CGNS types for all features.

    Example:
        >>> ds_dict, flat_cst, key_mappings = plaid_generator_to_huggingface_datasetdict(
        ...     {"train": lambda: iter(train_samples),
        ...      "test": lambda: iter(test_samples)},
        ...     processes_number=4,
        ...     writer_batch_size=2,
        ...     verbose=True
        ... )
        >>> print(ds_dict)
        DatasetDict({
            train: Dataset({
                features: ...
            }),
            test: Dataset({
                features: ...
            })
        })
        >>> print(flat_cst)
        {'Zone1/GridCoordinates': array([0., 0.1, 0.2])}
        >>> print(key_mappings["variable_features"][:3])
        ['Zone1/FlowSolution/VelocityX', 'Zone1/FlowSolution/VelocityY', ...]
    """
    hf_features = convert_to_hf_feature(variable_schema)

    all_features_keys = list(variable_schema.keys())

    def generator_fn(gen_func, all_features_keys, **kwargs):
        for sample in gen_func(**kwargs):
            hf_sample, _, _ = build_sample_dict(sample)
            yield {path: hf_sample.get(path, None) for path in all_features_keys}

    _dict = {}
    for split_name, gen_func in generators.items():
        gen = partial(generator_fn, all_features_keys=all_features_keys)
        gen_kwargs_ = gen_kwargs or {split_name: {} for split_name in generators.keys()}
        _dict[split_name] = datasets.Dataset.from_generator(
            generator=gen,
            gen_kwargs={"gen_func": gen_func, **gen_kwargs_[split_name]},
            features=hf_features,
            num_proc=processes_number,
            writer_batch_size=writer_batch_size,
            split=datasets.splits.NamedSplit(split_name),
        )

    return datasets.DatasetDict(_dict)


def to_var_sample_dict(
    ds: datasets.Dataset,
    i: int,
    enforce_shapes: bool = True,
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
    table = ds.data
    var_sample_dict = {}
    if not enforce_shapes:
        for name in table.column_names:
            value = table[name][i].values
            if value is None:
                var_sample_dict[name] = None  # pragma: no cover
            else:
                var_sample_dict[name] = value.to_numpy(zero_copy_only=False)
    else:
        for name in table.column_names:
            if isinstance(table[name][i], pa.NullScalar):
                var_sample_dict[name] = None  # pragma: no cover
            else:
                value = table[name][i].values
                if value is None:
                    var_sample_dict[name] = None  # pragma: no cover
                else:
                    if isinstance(value, pa.ListArray):
                        var_sample_dict[name] = np.stack(
                            value.to_numpy(zero_copy_only=False)
                        )
                    elif isinstance(value, pa.StringArray):  # pragma: no cover
                        var_sample_dict[name] = value.to_numpy(zero_copy_only=False)
                    else:
                        var_sample_dict[name] = value.to_numpy(zero_copy_only=True)

    return var_sample_dict


def sample_to_var_sample_dict(
    hf_sample: dict,
) -> dict:
    var_sample_dict = {}
    for name, value in hf_sample.items():
        if value is None:
            var_sample_dict[name] = None
        else:
            var_sample_dict[name] = np.array(value)
    return var_sample_dict
