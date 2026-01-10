# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import copy

import numpy as np
import pytest

from plaid.utils import cgns_helper


# %% Tests
class Test_cgns_helper:
    def test_get_base_names(self, sample_with_tree):
        tree = sample_with_tree.get_tree()
        # Test with full_path=False and unique=False
        base_names = cgns_helper.get_base_names(tree, full_path=False, unique=False)
        assert base_names == ["Base_2_2"]

        # Test with full_path=True and unique=False
        base_names_full = cgns_helper.get_base_names(tree, full_path=True, unique=False)
        print(base_names_full)
        assert base_names_full == ["/Base_2_2"]

        # Test with full_path=False and unique=True
        base_names_unique = cgns_helper.get_base_names(
            tree, full_path=False, unique=True
        )
        print(base_names_unique)
        assert base_names_unique == ["Base_2_2"]

    def test_get_time_values(self, samples):
        tree = samples[0].get_tree()
        time_value = cgns_helper.get_time_values(tree)
        assert time_value == 0.0

        empty_tree = []
        with pytest.raises(IndexError):
            cgns_helper.get_time_values(empty_tree)

    def test_show_cgns_tree(self, tree):
        cgns_helper.show_cgns_tree(tree)

    def test_show_cgns_tree_not_a_list(self):
        with pytest.raises(TypeError):
            cgns_helper.show_cgns_tree({1: 2})

    def test_fix_cgns_tree_types(self, tree):
        cgns_helper.fix_cgns_tree_types(tree)

    def test_compare_cgns_trees(self, tree, samples):
        assert cgns_helper.compare_cgns_trees(tree, tree)
        assert not cgns_helper.compare_cgns_trees(tree, samples[0].get_tree())

        tree2 = copy.deepcopy(tree)
        tree2[0] = "A"
        assert not cgns_helper.compare_cgns_trees(tree, tree2)

        tree2[0] = tree[0]
        tree2[1] = np.array([0], dtype=np.float32)
        tree[1] = np.array([0], dtype=np.float64)
        assert not cgns_helper.compare_cgns_trees(tree, tree2)

        tree[1] = np.array([1], dtype=np.float32)
        assert not cgns_helper.compare_cgns_trees(tree, tree2)

        tree[1] = "A"
        assert not cgns_helper.compare_cgns_trees(tree, tree2)

        tree[1] = tree2[1]
        tree[3] = "A_t"
        assert not cgns_helper.compare_cgns_trees(tree, tree2)

        tree[3] = tree2[3]
        tree[2][0][3] = "A_t"
        assert not cgns_helper.compare_cgns_trees(tree, tree2)

    def test_compare_cgns_trees_no_types(self, tree, samples):
        assert cgns_helper.compare_cgns_trees_no_types(tree, tree)
        assert not cgns_helper.compare_cgns_trees_no_types(tree, samples[0].get_tree())

        tree2 = copy.deepcopy(tree)
        tree2[0] = "A"
        assert not cgns_helper.compare_cgns_trees_no_types(tree, tree2)

        tree2[0] = tree[0]
        tree[2][0][1] = 1.0
        assert not cgns_helper.compare_cgns_trees_no_types(tree, tree2)

        tree[2][0][1] = tree2[2][0][1]
        tree[3] = "A_t"
        assert not cgns_helper.compare_cgns_trees_no_types(tree, tree2)

    def test_summarize_cgns_tree(self, tree):
        cgns_helper.summarize_cgns_tree(tree, verbose=False)

    def test_summarize_cgns_tree_verbose(self, tree):
        cgns_helper.summarize_cgns_tree(tree, verbose=True)
