# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np

from plaid.utils.init_with_tabular import initialize_dataset_with_tabular_data
from plaid.utils.split import split_dataset

# %% Functions


def split_examples():
    """
    This function shows the usage of dataset splitting functions.

    Example Usage:

    1. Initializing a Dataset:
    - Create a dataset with random tabular data for testing purposes.

    2. Splitting a Dataset with ratios:
    - Split the dataset into training, validation, and test sets using specified ratios.
    - Shuffle the dataset if desired.

    3. Splitting a Dataset with fixed sizes:
    - Split the dataset into training, validation, and test sets with fixed sample counts for each set.
    - Shuffle the dataset if desired.

    4. Splitting a Dataset with custom split IDs:
    - Split the dataset based on custom sample IDs for each set.
    - Specify the sample IDs for training, validation, and prediction sets.

    This function provides examples of using dataset splitting functions to divide a dataset into training,
    validation, and test sets, offering flexibility in the splitting process. It is intended for documentation
    purposes and familiarization with the PLAID library.
    """
    # %% Initialize Dataset
    print("# Initialize Dataset")
    nb_scalars = 7
    nb_samples = 700
    dset = initialize_dataset_with_tabular_data(
        {f'scalar_{j}': np.random.randn(nb_samples) for j in range(nb_scalars)})
    print(f"{dset=}")

    # %% Test split

    # 1. Split dataset
    print()
    print("# First split")
    options = {
        'shuffle': True,
        'split_ratios': {
            'train': 0.8,
            'val': 0.1,
        },
    }
    split = split_dataset(dset, options)
    print(f"{split=}")

    # 2. Split dataset
    print()
    print("# Second split")
    options = {
        'shuffle': True,
        'split_sizes': {
            'train': 140,
            'val': 80,
            'test': 50,
        },
    }
    split = split_dataset(dset, options)
    print(f"{dset=}")

    # 3. Split dataset
    print()
    print("# Third split")
    options = {
        'split_ids': {
            'train': np.arange(200),
            'val': np.arange(300, 600),
            'predict': np.arange(250, 350),
        },
    }
    split = split_dataset(dset, options)
    print(f"{split=}")


# %% Main Script
if __name__ == '__main__':
    split_examples()
