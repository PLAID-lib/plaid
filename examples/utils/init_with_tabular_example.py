# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np

from plaid.utils.init_with_tabular import initialize_dataset_with_tabular_data

# %% Functions


def init_with_tabular_example():
    """
    This function shows the initialization and basic operations of a dataset with tabular data.

    Example Usage:

    1. Initializing a Dataset with Tabular Data:
    - Generate random tabular data for multiple scalars.
    - Initialize a dataset with the tabular data.

    2. Accessing and Manipulating Data in the Dataset:
    - Retrieve and print the dataset and specific samples.
    - Access and display the value of a particular scalar within a sample.
    - Retrieve tabular data from the dataset based on scalar names.

    This function provides a simple example of initializing a dataset with tabular data and performing
    basic operations on the dataset. It is intended for documentation purposes and familiarization with
    the PLAID library.
    """

    # %% Initialize Dataset
    nb_scalars = 7
    nb_samples = 10

    names = [f"scalar_{j}" for j in range(nb_scalars)]

    tabular_data = {}
    for name in names:
        tabular_data[name] = np.random.randn(nb_samples)

    dataset = initialize_dataset_with_tabular_data(tabular_data)
    print(dataset)
    print(dataset[1])
    print(dataset[1].get_scalar("scalar_0"))
    print(dataset.get_scalars_to_tabular(names))

    print("done: initialize_dataset_with_tabular_data")


# %% Main Script
if __name__ == '__main__':
    init_with_tabular_example()
