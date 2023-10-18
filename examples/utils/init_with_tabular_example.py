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


def main():

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
    main()
