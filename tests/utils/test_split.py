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
from plaid.utils.init_with_tabular import initialize_dataset_with_tabular_data
from plaid.utils.split import split_dataset


# %% Fixtures

@pytest.fixture()
def nb_scalars():
    return 7

@pytest.fixture()
def nb_samples():
    return 70

@pytest.fixture()
def dataset(nb_scalars, nb_samples):
    tabular_data = {f'scalar_{j}': np.random.randn(nb_samples) for j in range(nb_scalars)}
    return initialize_dataset_with_tabular_data(tabular_data)
# %% Tests


class Test_split_dataset:

    def test_ratios(self, dataset):
        options = {
            'shuffle': True,
            'split_ratios': {
                'train': 0.8,
                'val': 0.1,
                'test': 0.1,
            },
            'unknown': { # it will be ignored
                'train': 0.8,
                'val': 0.1,
                'test': 0.1,
            },
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 56
        assert len(split['val']) == 7
        assert len(split['test']) == 7

        result = np.concatenate((split['train'], split['val'], split['test']), axis=0)
        assert len(set(result.tolist())) == len(result)


    def test_ratios_other(self, dataset):
        options = {
            'shuffle': True,
            'split_ratios': {
                'train': 0.8,
                'val': 0.1,
            },
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 56
        assert len(split['val']) == 7
        assert len(split['other']) == 7

        result = np.concatenate((split['train'], split['val'], split['other']), axis=0)
        assert len(set(result.tolist())) == len(result)

    def test_split_size(self, dataset):
        options = {
            'shuffle': True,
            'split_sizes': {
                'train': 40,
                'val': 20,
                'test': 10,
            },
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 40
        assert len(split['val']) == 20
        assert len(split['test']) == 10

        result = np.concatenate((split['train'], split['val'], split['test']), axis=0)
        assert len(set(result.tolist())) == len(result)

    def test_split_size_other(self, dataset):
        options = {
            'shuffle': True,
            'split_sizes': {
                'train': 40,
                'val': 10,
            },
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 40
        assert len(split['val']) == 10
        assert len(split['other']) == 20

        result = np.concatenate((split['train'], split['val'], split['other']), axis=0)
        assert len(set(result.tolist())) == len(result)


    def test_split_ids_unique_use(self, dataset):
        options = {
            'split_ids': {
                'train': np.arange(30),
                'val': np.arange(30, 60),
                'predict': np.arange(60, 70)
            }
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 30
        assert len(split['val']) == 30
        assert len(split['predict']) == 10

        result = np.concatenate((split['train'], split['val'], split['predict']), axis=0)
        assert len(set(result.tolist())) == len(result)

    def test_split_ids(self, dataset):
        options = {
            'shuffle': True,
            'split_ids': {
                'train': np.arange(30),
                'val': np.arange(30, 70),
                'predict': np.arange(25, 35)
            }
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 30
        assert len(split['val']) == 40
        assert len(split['predict']) == 10

    def test_split_ids_other(self, dataset):
        options = {
            'split_ids': {
                'train': np.arange(20),
                'val': np.arange(30, 60),
                'predict': np.arange(25, 35)
            }
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 20
        assert len(split['val']) == 30
        assert len(split['predict']) == 10
        assert len(split['other']) == 15

    def test_split_ratios_and_sizes(self, dataset):
        options = {
            'shuffle': True,
            'split_ratios': {
                'train': 0.8,
                'test': 0.1,
            },
            'split_sizes': {
                'val': 7
            }
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 56
        assert len(split['test']) == 7
        assert len(split['val']) == 7

        result = np.concatenate((split['train'], split['val'], split['test']), axis=0)
        assert len(set(result.tolist())) == len(result)

    def test_split_ratios_and_sizes_other(self, dataset):
        options = {
            'shuffle': True,
            'split_ratios': {
                'train': 0.7,
                'test': 0.1,
            },
            'split_sizes': {
                'val': 7
            }
        }
        split = split_dataset(dataset, options)
        assert len(split['train']) == 49
        assert len(split['test']) == 7
        assert len(split['val']) == 7
        assert len(split['other']) == 7

        result = np.concatenate((split['train'], split['val'], split['test'], split['other']), axis=0)
        assert len(set(result.tolist())) == len(result)


    def test_split_ids_out_of_bounds(self, dataset):
        with pytest.raises(ValueError):
            split_dataset(dataset, {'shuffle': True, 'split_ids': {'train': np.arange(-1, 69)}})
        with pytest.raises(ValueError):
            split_dataset(dataset, {'shuffle': True, 'split_ids': {'train': np.arange(0, 80)}})

    def test_split_ratios_sizes_out_of_bounds(self, dataset):
        with pytest.raises(AssertionError):
            split_dataset(dataset, {'shuffle': True, 'split_ratios': {'train': 0.8, 'predict': 0.05, 'test': 0.1}, 'split_sizes': {'val': 80}})

    def test_ratios_error(self, dataset):
        options = {
            'shuffle': True,
            'split_ratios': {
                'train': 0.8,
                'val': 0.8,
            }
        }
        with pytest.raises(AssertionError):
            split_dataset(dataset, options)

    def test_fail_other(self, dataset):
        # 'other' key name is not authorized
        with pytest.raises(ValueError):
            split_dataset(dataset, {'split_ratios': {'other': 0.8}})
        with pytest.raises(ValueError):
            split_dataset(dataset, {'split_sizes': {'other': 1}})
        with pytest.raises(ValueError):
            split_dataset(dataset, {'split_ids': {'other': 1}})

    def test_various_assertion_cases(self, dataset):
        # Incompatible strategies
        with pytest.raises(AssertionError):
            split_dataset(dataset, {'split_ids': {'a': 1}, 'split_ratios': {'b': 0.2}})
        with pytest.raises(AssertionError):
            split_dataset(dataset, {'split_ids': {'a': 1}, 'split_sizes': {'b': 1}})
        # Same key name 'a'
        with pytest.raises(AssertionError):
            split_dataset(dataset, {'split_ratios': {'a': 0.2}, 'split_sizes': {'a': 1}})
        # Bad type for ratios (must be float)
        with pytest.raises(AssertionError):
            split_dataset(dataset, {'split_ratios': {'a': 1}})
        # Bad type for ratios (must be int)
        with pytest.raises(AssertionError):
            split_dataset(dataset, {'split_sizes': {'a': 0.1}})

    def test_shuffle(self, dataset):
        split1 = split_dataset(dataset, {'shuffle': True})
        split2 = split_dataset(dataset, {'shuffle': True})
        assert not np.array_equal(split1['other'], split2['other']), "shuffle didn't work"
