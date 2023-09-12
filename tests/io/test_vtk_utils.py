# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np
import pytest
from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.io.vtk_utils import (load_temporal_vtk, read_vtk, save_temporal_vtk,
                                write_vtk)

# %% Fixtures

# @pytest.fixture()
# def samples():
#     nb_samples = 400
#     samples = [Sample() for _ in range(nb_samples)]
#     return samples

# @pytest.fixture()
# def dset(samples):
#     dset = Dataset()
#     dset.add_samples(samples)
#     return dset

# %% Tests


class test_read_vtk():
    pass


class test_write_vtk():
    pass


class test_load_temporal_vtk():
    pass


class test_save_temporal_vtk():
    pass
