# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np

from plaid.utils.init import initialize_dataset_with_tabular_data
from plaid.utils.split import split_dataset

# %% Functions


def main():

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
    main()
