# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import pytest

from plaid.utils.cgns_helper import get_base_names, get_time_values, show_cgns_tree


# %% Tests
class Test_cgns_helper:
    def test_get_base_names(self, sample_with_tree):
        tree = sample_with_tree.get_mesh()
        # Test with full_path=False and unique=False
        base_names = get_base_names(tree, full_path=False, unique=False)
        assert base_names == ["Base_2_2"]

        # Test with full_path=True and unique=False
        base_names_full = get_base_names(tree, full_path=True, unique=False)
        print(base_names_full)
        assert base_names_full == ["/Base_2_2"]

        # Test with full_path=False and unique=True
        base_names_unique = get_base_names(tree, full_path=False, unique=True)
        print(base_names_unique)
        assert base_names_unique == ["Base_2_2"]

    def test_get_time_values(self, sample_with_tree):
        tree = sample_with_tree.get_mesh()
        time_value = get_time_values(tree)
        assert time_value == 0.0

        empty_tree = []
        with pytest.raises(IndexError):
            get_time_values(empty_tree)

    def test_show_cgns_tree(self, tree):
        show_cgns_tree(tree)

    def test_show_cgns_tree_not_a_list(self):
        with pytest.raises(TypeError):
            show_cgns_tree({1: 2})
