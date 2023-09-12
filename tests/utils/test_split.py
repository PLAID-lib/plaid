# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np
import pytest

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.utils.split import split_dataset

from ..containers.test_dataset import dataset, dataset_with_samples, samples

# %% Fixtures


@pytest.fixture()
def nb_samples():
    return 400

# %% Tests


class Test_split_dataset():
    pass

    # TODO: update from new plaid with Splits separated from Dataset ->
    # SplitStrategy

    # def test_shuffle(self, dataset_with_samples):
    #     split_dataset(dataset_with_samples, {'shuffle': True})

    # def test_split_ratios(self, dataset_with_samples):
    #     split_dataset(dataset_with_samples, {'shuffle': True, 'split_ratios': {'train': 0.5, 'val': 0.3}})

    # def test_split_sizes(self, dataset_with_samples):
    #     split_dataset(dataset_with_samples, {'shuffle': True, 'split_sizes': {'train': 140, 'val': 80, 'test': 50}})

    # def test_split_ids(self, dataset_with_samples):
    #     split_dataset(dataset_with_samples, {'shuffle': True, 'split_ids': {'train': np.arange(200), 'predict': np.arange(300, 400), 'test': np.arange(250,350)}})

    # def test_split_ids_fail_other(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         split_dataset(dataset_with_samples, {'shuffle': True, 'split_ids': {'train': np.arange(200), 'other': np.arange(300, 400), 'test': np.arange(250,350)}})

    # def test_split_ids_out_of_bounds(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         split_dataset(dataset_with_samples, {'shuffle': True, 'split_ids': {'train': np.arange(-1, 1001)}})

    # def test_split_ratios_sizes_fail_other(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         split_dataset(dataset_with_samples, {'shuffle': True, 'split_ratios': {'train': 0.8, 'other': 0.05, 'test': 0.1}, 'split_sizes': {'val': 80}})
