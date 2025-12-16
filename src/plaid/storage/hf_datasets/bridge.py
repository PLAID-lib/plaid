# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#


"""HF Datasets bridge utilities.

This module provides bridge functions for converting between PLAID datasets/samples
and Hugging Face Datasets format. It includes utilities for feature type conversion,
dataset generation from PLAID objects, and sample reconstruction.
"""

from functools import partial
from typing import Any, Callable, Generator, Optional

import datasets
import numpy as np
import pyarrow as pa
from datasets import Features, Sequence, Value

from plaid import Dataset, Sample
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.types import IndexType


def convert_dtype_to_hf_feature(feature_type: dict[str, Any]):
    """Convert a PLAID feature type dict to Hugging Face Feature.

    Args:
        feature_type (dict): Dictionary with 'dtype' and 'ndim' keys.

    Returns:
        Features or Sequence: The corresponding HF feature type.
    """
    base_dtype = feature_type["dtype"]
    ndim = feature_type["ndim"]

    feature = Value(base_dtype)
    for _ in range(ndim):
        feature = Sequence(feature)
    return feature


def convert_to_hf_feature(variable_schema: dict[str, dict]):
    """Convert a PLAID variable schema to Hugging Face Features.

    Args:
        variable_schema (dict[str, dict]): Mapping of variable names to type dicts.

    Returns:
        Features: The HF Features object.
    """
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
        var_features_types (dict[str, dict]):
            Dictionary mapping feature names to their type information.
        processes_number (int, optional, default=1):
            Number of parallel processes to use when writing the Hugging Face dataset.
        writer_batch_size (int, optional, default=1):
            Batch size used when writing samples to disk in Hugging Face format.

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
        variable_schema (dict):
            Dictionary defining the schema of variables/features in the dataset.
        processes_number (int, optional, default=1):
            Number of processes used internally by Hugging Face when materializing
            the dataset from the generators.
        writer_batch_size (int, optional, default=1):
            Batch size used when writing samples to disk in Hugging Face format.
        gen_kwargs (dict, optional, default=None):
            Optional mapping from split names to dictionaries of keyword arguments
            to be passed to each generator function, used for parallelization.

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
) -> dict[str, Any]:
    """Convert a Hugging Face dataset row to a variable sample dict containing the features that vary in the dataset.

    Args:
        ds (datasets.Dataset): The Hugging Face dataset.
        i (int): The row index.
        enforce_shapes (bool): Whether to enforce consistent shapes.

    Returns:
        dict: The variable sample dictionary.
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
    hf_sample: dict[str, Any],
) -> dict[str, Any]:
    """Convert a Hugging Face sample dict to variable sample dict.

    Args:
        hf_sample (dict): The HF sample dictionary.

    Returns:
        dict: The processed variable sample dictionary.
    """
    var_sample_dict = {}
    for name, value in hf_sample.items():
        if value is None:
            var_sample_dict[name] = None
        else:
            var_sample_dict[name] = np.array(value)
    return var_sample_dict
