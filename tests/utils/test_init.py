# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np
import pytest

from plaid.utils.init import initialize_dataset_with_tabular_data

# %% Fixtures


@pytest.fixture()
def nb_samples():
    return 400


@pytest.fixture()
def scalar_tabular_data(nb_samples):
    return {
        'scalar_name_1': np.random.randn(nb_samples),
        'scalar_name_2': np.random.randn(nb_samples),
    }


@pytest.fixture()
def quantity_tabular_data(nb_samples):
    nx = 11
    ny = 7
    nz = 5
    return {
        'test_scalar': np.random.randn(nb_samples),
        'test_1D_field': np.random.randn(nb_samples, nx),
        'test_2D_field': np.random.randn(nb_samples, nx, ny),
        'test_3D_field': np.random.randn(nb_samples, nx, ny, nz),
    }

# %% Tests


class Test_initialize_dataset_with_tabular_data():

    def test_initialize_dataset_with_tabular_data(
            self, scalar_tabular_data, nb_samples):
        dset = initialize_dataset_with_tabular_data(scalar_tabular_data)
        assert (len(dset) == nb_samples)
