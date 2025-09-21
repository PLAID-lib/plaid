"""Utility function for splitting a Dataset."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from plaid import Dataset

logger = logging.getLogger(__name__)


# %% Functions


def split_dataset(dset: Dataset, options: dict[str, Any]) -> dict[str, int]:
    """Splits a Dataset in several sub Datasets.

    Args:
        dset(Dataset): dataset to be splited.
        options([str,Any]): may have keys 'shuffle', 'split_sizes', 'split_ratios' or 'split_ids':
            - 'split_sizes' is supposed to be a dict[str,int]: split name -> size of splited dataset
            - 'split_ratios' is supposed to be a dict[str,float]: split name -> size ratios of splited dataset
            - 'split_ids' is supposed to be a dict[str,np.ndarray(int)]: split name -> ids of samples in splited dataset
            - if 'shuffle' is not set, it is supposed to be False
            - if 'split_ids' is present, other keys will be ignored
    Returns:
        Dataset: the dataset with splits.

    Raises:
        ValueError: If a split is named 'other' (not authorized).
        ValueError: If there are some ids out of bounds.
        ValueError: If some split names are in 'split_ratios' and 'split_sizes'.

    Example:
        .. code-block:: python

            # Given a dataset of 2 samples
            print(dataset)
            >>> Dataset(2 samples, 2 scalars, 2 fields)

            options = {
                'shuffle':False,
                'split_sizes': {
                    'train':1,
                    'val':1,
                    },
            }
            split = split_dataset(dataset, options)
            print(split)
            >>> {'train': [0], 'val': [1]}

    """
    _splits = {}
    all_ids = dset.get_sample_ids()
    total_size = len(dset)

    # Verify that split option validity
    def check_options_validity(split_option: dict):
        assert isinstance(split_option, dict), "split option must be a dictionary"
        if "other" in split_option:
            raise ValueError("name 'other' is not authorized for a split")

    # Check that the keys in options are among authorized keys
    authorized_task = ["split_ids", "split_ratios", "split_sizes", "shuffle"]
    for task in options:
        if task in authorized_task:
            continue
        logger.warning(f"option {task} is not authorized. {task} key will be ignored")

    f_case = len(set(["split_ids"]).intersection(set(options.keys())))
    s_case = len(set(["split_ratios", "split_sizes"]).intersection(set(options.keys())))
    assert f_case == 0 or s_case == 0, (
        "split by id cannot exist with split by ratios or sizes"
    )

    # First case
    if "split_ids" in options:
        check_options_validity(options["split_ids"])

        if len(options) > 1:
            logger.warning(
                "options has key 'split_ids' and 'shuffle' -> 'shuffle' key will be ignored"
            )

        # all_ids = np.arange(total_size)
        used_ids = np.unique(
            np.concatenate([ids for ids in options["split_ids"].values()])
        )

        if np.min(used_ids) < 0 or np.max(used_ids) >= total_size:
            raise ValueError(
                "there are some ids out of bounds -> min/max:{}/{} | dataset len:{}".format(
                    np.min(used_ids), np.max(used_ids), total_size
                )
            )

        other_ids = np.setdiff1d(all_ids, used_ids)
        if len(other_ids) > 0:
            options["split_ids"]["other"] = other_ids

        if len(used_ids) < np.sum([len(ids) for ids in options["split_ids"].values()]):
            logger.warning("there are some ids present in several splits")

        for name in options["split_ids"]:
            _splits[name] = options["split_ids"][name]
            # split_samples = []
            # for id in options['split_ids'][name]:
            #     split_samples.append(dset[id])
            # dset._splits[name] = Dataset()
            # dset._splits[name].add_samples(split_samples)
        return _splits

    if "shuffle" in options:
        shuffle = options["shuffle"]
    else:
        shuffle = False

    split_sizes = [0]
    split_names = []
    # Second case
    if "split_ratios" in options:
        check_options_validity(options["split_ratios"])

        for key, value in options["split_ratios"].items():
            assert isinstance(value, float)
            split_names.append(key)
            split_sizes.append(int(total_size * value))

    if "split_sizes" in options:
        check_options_validity(options["split_sizes"])

        for key, value in options["split_sizes"].items():
            assert "split_ratios" not in options or key not in options["split_ratios"]
            assert isinstance(value, int)
            split_names.append(key)
            split_sizes.append(value)

    assert np.sum(split_sizes) <= total_size
    if np.sum(split_sizes) < total_size:
        split_names.append("other")
        split_sizes.append(total_size - np.sum(split_sizes))
    slices = np.cumsum(split_sizes)

    # all_ids = np.arange(total_size)
    if shuffle:
        all_ids = np.random.permutation(all_ids)

    for i_split in range(len(split_names)):
        _splits[split_names[i_split]] = all_ids[slices[i_split] : slices[i_split + 1]]
        # split_samples = []
        # for id in all_ids[slices[i_split]:slices[i_split+1]]:
        #     split_samples.append(dset[id])
        # dset._splits[split_names[i_split]] = Dataset()
        # dset._splits[split_names[i_split]].add_samples(split_samples)

    return _splits


def mmd_subsample_fn(
    X: NDArray[np.float64],
    size: int,
    initial_ids: Optional[list[int]] = None,
    memory_safe: bool = False,
) -> NDArray[np.int64]:
    """Selects samples in the input table by greedily minimizing the maximum mena discrepancy (MMD).

    Args:
        X(np.ndarray): input table of shape n_samples x n_features
        size(int): number of samples to select
        initial_ids(list[int]): a list of ids of points to initialize the gready algorithm. Defaults to None.
        memory_safe(bool): if True, avoids a memory expensive computation. Useful for large tables. Defaults to False.

    Returns:
        np.ndarray: array of selected samples
    Example:
        .. code-block:: python

            # Let X be drawn from a standard 10-dimensional Gaussian distribution
            np.random.seed(0)
            X = np.random.randn(1000,10)
            # Select 100 particles
            idx = mmd_subsample_fn(X, size=100)
            print(idx)
            >>> [765 113 171 727 796 855 715 207 458 603  23 384 860   3 459 708 794 138
                 221 639   8 816 619 806 398 236  36 404 167  87 201 676 961 624 556 840
                 485 975 283 150 554 409  69 769 332 357 388 216 900 134  15 730  80 694
                 251 714  11 817 525 382 328  67 356 514 597 668 959 260 968  26 209 789
                 305 122 989 571 801 322  14 160 908  12   1 980 582 440  42 452 666 526
                 290 231 712  21 606 575 656 950 879 948]
            # In this simple Gaussian example, the means and standard deviations of the
            # selected subsample should be close to the ones of the original sample
            print(np.abs(np.mean(x[idx], axis=0) - np.mean(x, axis=0)))
            >>> [0.00280955 0.00220179 0.01359079 0.00461107 0.0011997  0.01106616
            0.01157571 0.0061314  0.00813494 0.0026543]
            print(np.abs(np.std(x[idx], axis=0) - np.std(x, axis=0)))
            >>> [0.0067711  0.00316008 0.00860733 0.07130127 0.02858514 0.0173707
            0.00739646 0.03526784 0.0054039  0.00351996]
    """
    n = X.shape[0]
    assert size <= n
    # Precompute norms and distance matrix
    norms = np.linalg.norm(X, axis=1)
    if memory_safe:
        k0_mean = np.zeros(n)
        for i in range(n):
            kxy = norms[i : i + 1, None] + norms[None, :] - cdist(X[i : i + 1], X)
            k0_mean[i] = np.mean(kxy)
    else:
        dist_matrix = cdist(X, X)
        gram_matrix = norms[:, None] + norms[None, :] - dist_matrix
        k0_mean = np.mean(gram_matrix, axis=1)

    idx = np.zeros(size, dtype=np.int64)
    if initial_ids is None or len(initial_ids) == 0:
        k0 = np.zeros((n, size))
        k0[:, 0] = 2.0 * norms

        idx[0] = np.argmin(k0[:, 0] - 2.0 * k0_mean)
        for i in range(1, size):
            x_ = X[idx[i - 1]]
            dist = np.linalg.norm(X - x_, axis=1)
            k0[:, i] = -dist + norms[idx[i - 1]] + norms

            idx[i] = np.argmin(
                k0[:, 0]
                + 2.0 * np.sum(k0[:, 1 : (i + 1)], axis=1)
                - 2.0 * (i + 1) * k0_mean
            )
    else:
        assert len(initial_ids) < size
        idx[: len(initial_ids)] = initial_ids
        k0 = np.zeros((n, size))

        k0[:, 0] = 2.0 * norms
        for i in range(1, size):
            x_ = X[idx[i - 1]]
            dist = np.linalg.norm(X - x_, axis=1)
            k0[:, i] = -dist + norms[idx[i - 1]] + norms

            if i >= len(initial_ids):
                idx[i] = np.argmin(
                    k0[:, 0]
                    + 2.0 * np.sum(k0[:, 1 : (i + 1)], axis=1)
                    - 2.0 * (i + 1) * k0_mean
                )
    return idx
