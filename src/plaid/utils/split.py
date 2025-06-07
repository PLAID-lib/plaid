"""Utility function for splitting a Dataset."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import logging
from typing import Any

import numpy as np

from plaid.containers.dataset import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)

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


# %% Classes
