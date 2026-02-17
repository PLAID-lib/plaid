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

    def test_update_features_for_CGNS_compatibility(self):
        """Test the update_features_for_CGNS_compatibility function."""

        context_constant_features = {
            "Base_1_2",
            "Base_1_2/Blade",
            "Base_1_2/Blade_times",
            "Base_1_2/Zone",
            "Base_1_2/Zone/CellData",
            "Base_1_2/Zone/CellData/GridLocation",
            "Base_1_2/Zone/CellData/GridLocation_times",
            "Base_1_2/Zone/CellData_times",
            "Base_1_2/Zone/Elements_BAR_2",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange_times",
            "Base_1_2/Zone/Elements_BAR_2_times",
            "Base_1_2/Zone/FamilyName",
            "Base_1_2/Zone/FamilyName_times",
            "Base_1_2/Zone/GridCoordinates",
            "Base_1_2/Zone/GridCoordinates/CoordinateX_times",
            "Base_1_2/Zone/GridCoordinates/CoordinateY_times",
            "Base_1_2/Zone/GridCoordinates_times",
            "Base_1_2/Zone/PointData",
            "Base_1_2/Zone/PointData/GridLocation",
            "Base_1_2/Zone/PointData/GridLocation_times",
            "Base_1_2/Zone/PointData/M_iso_times",
            "Base_1_2/Zone/PointData_times",
            "Base_1_2/Zone/SurfaceData",
            "Base_1_2/Zone/SurfaceData/GridLocation",
            "Base_1_2/Zone/SurfaceData/GridLocation_times",
            "Base_1_2/Zone/SurfaceData_times",
            "Base_1_2/Zone/ZoneBC",
            "Base_1_2/Zone/ZoneBC/1D",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation_times",
            "Base_1_2/Zone/ZoneBC/1D/PointList",
            "Base_1_2/Zone/ZoneBC/1D/PointList_times",
            "Base_1_2/Zone/ZoneBC/1D_times",
            "Base_1_2/Zone/ZoneBC_times",
            "Base_1_2/Zone/ZoneType",
            "Base_1_2/Zone/ZoneType_times",
            "Base_1_2/Zone_times",
            "Base_1_2_times",
            "Base_2_2",
            "Base_2_2/Blade",
            "Base_2_2/Blade_times",
            "Base_2_2/Zone",
            "Base_2_2/Zone/CellData",
            "Base_2_2/Zone/CellData/GridLocation",
            "Base_2_2/Zone/CellData/GridLocation_times",
            "Base_2_2/Zone/CellData_times",
            "Base_2_2/Zone/Elements_QUAD_4",
            "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
            "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity_times",
            "Base_2_2/Zone/Elements_QUAD_4/ElementRange",
            "Base_2_2/Zone/Elements_QUAD_4/ElementRange_times",
            "Base_2_2/Zone/Elements_QUAD_4_times",
            "Base_2_2/Zone/FamilyName",
            "Base_2_2/Zone/FamilyName_times",
            "Base_2_2/Zone/GridCoordinates",
            "Base_2_2/Zone/GridCoordinates/CoordinateX_times",
            "Base_2_2/Zone/GridCoordinates/CoordinateY_times",
            "Base_2_2/Zone/GridCoordinates_times",
            "Base_2_2/Zone/PointData",
            "Base_2_2/Zone/PointData/GridLocation",
            "Base_2_2/Zone/PointData/GridLocation_times",
            "Base_2_2/Zone/PointData/mach_times",
            "Base_2_2/Zone/PointData/nut_times",
            "Base_2_2/Zone/PointData/ro_times",
            "Base_2_2/Zone/PointData/roe_times",
            "Base_2_2/Zone/PointData/rou_times",
            "Base_2_2/Zone/PointData/rov_times",
            "Base_2_2/Zone/PointData/sdf_times",
            "Base_2_2/Zone/PointData_times",
            "Base_2_2/Zone/SurfaceData",
            "Base_2_2/Zone/SurfaceData/GridLocation",
            "Base_2_2/Zone/SurfaceData/GridLocation_times",
            "Base_2_2/Zone/SurfaceData_times",
            "Base_2_2/Zone/ZoneBC",
            "Base_2_2/Zone/ZoneBC/Extrado",
            "Base_2_2/Zone/ZoneBC/Extrado/GridLocation",
            "Base_2_2/Zone/ZoneBC/Extrado/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Extrado/PointList",
            "Base_2_2/Zone/ZoneBC/Extrado/PointList_times",
            "Base_2_2/Zone/ZoneBC/Extrado_times",
            "Base_2_2/Zone/ZoneBC/Inflow",
            "Base_2_2/Zone/ZoneBC/Inflow/GridLocation",
            "Base_2_2/Zone/ZoneBC/Inflow/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Inflow/PointList",
            "Base_2_2/Zone/ZoneBC/Inflow/PointList_times",
            "Base_2_2/Zone/ZoneBC/Inflow_times",
            "Base_2_2/Zone/ZoneBC/Intrado",
            "Base_2_2/Zone/ZoneBC/Intrado/GridLocation",
            "Base_2_2/Zone/ZoneBC/Intrado/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Intrado/PointList",
            "Base_2_2/Zone/ZoneBC/Intrado/PointList_times",
            "Base_2_2/Zone/ZoneBC/Intrado_times",
            "Base_2_2/Zone/ZoneBC/Outflow",
            "Base_2_2/Zone/ZoneBC/Outflow/GridLocation",
            "Base_2_2/Zone/ZoneBC/Outflow/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Outflow/PointList",
            "Base_2_2/Zone/ZoneBC/Outflow/PointList_times",
            "Base_2_2/Zone/ZoneBC/Outflow_times",
            "Base_2_2/Zone/ZoneBC/Periodic_1",
            "Base_2_2/Zone/ZoneBC/Periodic_1/GridLocation",
            "Base_2_2/Zone/ZoneBC/Periodic_1/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Periodic_1/PointList",
            "Base_2_2/Zone/ZoneBC/Periodic_1/PointList_times",
            "Base_2_2/Zone/ZoneBC/Periodic_1_times",
            "Base_2_2/Zone/ZoneBC/Periodic_2",
            "Base_2_2/Zone/ZoneBC/Periodic_2/GridLocation",
            "Base_2_2/Zone/ZoneBC/Periodic_2/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Periodic_2/PointList",
            "Base_2_2/Zone/ZoneBC/Periodic_2/PointList_times",
            "Base_2_2/Zone/ZoneBC/Periodic_2_times",
            "Base_2_2/Zone/ZoneBC_times",
            "Base_2_2/Zone/ZoneType",
            "Base_2_2/Zone/ZoneType_times",
            "Base_2_2/Zone_times",
            "Base_2_2_times",
            "Global",
            "Global/Pr_times",
            "Global/Q_times",
            "Global/Tr_times",
            "Global/angle_in_times",
            "Global/angle_out_times",
            "Global/eth_is_times",
            "Global/mach_out_times",
            "Global/power_times",
            "Global_times",
        }

        # Setup: Create context features that would exist in a dataset
        context_variable_features = {
            "Base_1_2/Zone/GridCoordinates/CoordinateX",
            "Base_1_2/Zone/GridCoordinates/CoordinateY",
            "Base_1_2/Zone/PointData/M_iso",
            "Base_2_2/Zone/GridCoordinates/CoordinateX",
            "Base_2_2/Zone/GridCoordinates/CoordinateY",
            "Base_2_2/Zone/PointData/mach",
            "Base_2_2/Zone/PointData/nut",
            "Base_2_2/Zone/PointData/ro",
            "Base_2_2/Zone/PointData/roe",
            "Base_2_2/Zone/PointData/rou",
            "Base_2_2/Zone/PointData/rov",
            "Base_2_2/Zone/PointData/sdf",
            "Global/Pr",
            "Global/Q",
            "Global/Tr",
            "Global/angle_in",
            "Global/angle_out",
            "Global/eth_is",
            "Global/mach_out",
            "Global/power",
        }

        # Test 1: Basic field feature expansion
        print("Test 1")
        features = ["Base_1_2/Zone/GridCoordinates/CoordinateX"]
        result = cgns_helper.update_features_for_CGNS_compatibility(
            features, context_constant_features, context_variable_features
        )
        ref = [
            "Base_1_2/Zone/GridCoordinates/CoordinateY_times",
            "Base_1_2",
            "Base_1_2/Zone",
            "Base_1_2/Zone/Elements_BAR_2",
            "Base_1_2/Zone/ZoneType",
            "Base_1_2/Zone/GridCoordinates_times",
            "Base_1_2/Zone/ZoneBC/1D",
            "Base_1_2/Zone/FamilyName_times",
            "Base_1_2/Zone/GridCoordinates/CoordinateX_times",
            "Base_1_2/Zone/ZoneBC_times",
            "Base_1_2/Zone/ZoneBC",
            "Base_1_2/Zone/Elements_BAR_2_times",
            "Base_1_2/Blade_times",
            "Base_1_2/Zone/GridCoordinates/CoordinateX",
            "Base_1_2/Zone/FamilyName",
            "Base_1_2_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange_times",
            "Base_1_2/Blade",
            "Base_1_2/Zone/ZoneBC/1D/PointList_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation",
            "Base_1_2/Zone/GridCoordinates/CoordinateY",
            "Base_1_2/Zone/ZoneType_times",
            "Base_1_2/Zone/ZoneBC/1D_times",
            "Base_1_2/Zone_times",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation_times",
            "Base_1_2/Zone/ZoneBC/1D/PointList",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity",
            "Base_1_2/Zone/GridCoordinates",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity_times",
        ]
        assert set(result) == set(ref)
        print("----------------")

        print("Test 2")
        features = ["Base_1_2/Zone/Elements_BAR_2/ElementConnectivity"]
        result = cgns_helper.update_features_for_CGNS_compatibility(
            features, context_constant_features, context_variable_features
        )
        ref = [
            "Base_1_2/Blade_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity",
            "Base_1_2/Zone/ZoneBC/1D/PointList_times",
            "Base_1_2/Zone/ZoneType",
            "Base_1_2/Zone_times",
            "Base_1_2/Zone/GridCoordinates",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity_times",
            "Base_1_2",
            "Base_1_2/Zone/GridCoordinates/CoordinateX",
            "Base_1_2/Zone/ZoneType_times",
            "Base_1_2/Zone/ZoneBC/1D",
            "Base_1_2/Zone/ZoneBC",
            "Base_1_2_times",
            "Base_1_2/Zone/ZoneBC/1D/PointList",
            "Base_1_2/Zone/Elements_BAR_2",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation",
            "Base_1_2/Zone/Elements_BAR_2_times",
            "Base_1_2/Zone/ZoneBC/1D_times",
            "Base_1_2/Zone/GridCoordinates/CoordinateY_times",
            "Base_1_2/Zone/FamilyName_times",
            "Base_1_2/Zone/GridCoordinates/CoordinateY",
            "Base_1_2/Zone",
            "Base_1_2/Zone/ZoneBC_times",
            "Base_1_2/Blade",
            "Base_1_2/Zone/GridCoordinates_times",
            "Base_1_2/Zone/FamilyName",
            "Base_1_2/Zone/GridCoordinates/CoordinateX_times",
        ]
        assert set(result) == set(ref)
        print("----------------")

        print("Test 3")
        features = ["Base_1_2/Zone/ZoneBC/1D/PointList"]
        result = cgns_helper.update_features_for_CGNS_compatibility(
            features, context_constant_features, context_variable_features
        )
        ref = [
            "Base_1_2/Zone/ZoneType_times",
            "Base_1_2/Zone",
            "Base_1_2/Zone/ZoneBC_times",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation_times",
            "Base_1_2/Zone/GridCoordinates_times",
            "Base_1_2/Zone_times",
            "Base_1_2/Zone/FamilyName_times",
            "Base_1_2/Zone/GridCoordinates",
            "Base_1_2/Zone/Elements_BAR_2_times",
            "Base_1_2/Zone/ZoneBC/1D/PointList_times",
            "Base_1_2/Zone/GridCoordinates/CoordinateY",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity_times",
            "Base_1_2/Blade",
            "Base_1_2/Zone/FamilyName",
            "Base_1_2/Zone/ZoneBC/1D/GridLocation",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementRange",
            "Base_1_2",
            "Base_1_2/Zone/ZoneBC/1D/PointList",
            "Base_1_2/Zone/Elements_BAR_2",
            "Base_1_2/Zone/GridCoordinates/CoordinateX",
            "Base_1_2/Blade_times",
            "Base_1_2/Zone/ZoneBC",
            "Base_1_2/Zone/ZoneBC/1D_times",
            "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity",
            "Base_1_2/Zone/GridCoordinates/CoordinateY_times",
            "Base_1_2/Zone/ZoneType",
            "Base_1_2/Zone/ZoneBC/1D",
            "Base_1_2/Zone/GridCoordinates/CoordinateX_times",
            "Base_1_2_times",
        ]
        assert set(result) == set(ref)
        print("----------------")

        print("Test 4")
        features = ["Base_2_2/Zone/PointData/rov"]
        result = cgns_helper.update_features_for_CGNS_compatibility(
            features, context_constant_features, context_variable_features
        )
        ref = [
            "Base_2_2/Zone/ZoneBC/Periodic_1/GridLocation_times",
            "Base_2_2/Zone/Elements_QUAD_4/ElementRange_times",
            "Base_2_2/Zone/ZoneBC/Periodic_1/GridLocation",
            "Base_2_2/Zone/PointData",
            "Base_2_2/Zone/ZoneBC/Extrado",
            "Base_2_2/Zone/ZoneBC/Inflow/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Periodic_2_times",
            "Base_2_2/Zone/PointData/GridLocation_times",
            "Base_2_2/Blade_times",
            "Base_2_2/Zone/ZoneBC/Extrado/PointList",
            "Base_2_2/Zone/GridCoordinates/CoordinateY_times",
            "Base_2_2/Zone/ZoneBC/Inflow/PointList_times",
            "Base_2_2/Zone/ZoneType_times",
            "Base_2_2/Blade",
            "Base_2_2/Zone/PointData_times",
            "Base_2_2/Zone/ZoneBC/Periodic_2/PointList_times",
            "Base_2_2/Zone/Elements_QUAD_4/ElementRange",
            "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
            "Base_2_2/Zone/ZoneBC_times",
            "Base_2_2/Zone/ZoneBC/Extrado/GridLocation",
            "Base_2_2/Zone/ZoneBC/Periodic_1_times",
            "Base_2_2/Zone/ZoneBC/Intrado",
            "Base_2_2/Zone/ZoneBC/Intrado/GridLocation",
            "Base_2_2/Zone/ZoneBC/Outflow/GridLocation",
            "Base_2_2/Zone/GridCoordinates_times",
            "Base_2_2/Zone/ZoneBC/Periodic_2",
            "Base_2_2/Zone/ZoneBC/Extrado/GridLocation_times",
            "Base_2_2/Zone/ZoneBC/Outflow/PointList",
            "Base_2_2/Zone/ZoneBC/Periodic_2/GridLocation",
            "Base_2_2/Zone/GridCoordinates/CoordinateX",
            "Base_2_2/Zone/ZoneType",
            "Base_2_2/Zone/PointData/GridLocation",
            "Base_2_2/Zone/ZoneBC/Outflow",
            "Base_2_2/Zone/ZoneBC/Outflow/GridLocation_times",
            "Base_2_2/Zone",
            "Base_2_2/Zone/ZoneBC/Inflow_times",
            "Base_2_2/Zone/ZoneBC/Periodic_1",
            "Base_2_2/Zone/ZoneBC/Extrado/PointList_times",
            "Base_2_2/Zone/PointData/rov",
            "Base_2_2/Zone/ZoneBC/Periodic_2/GridLocation_times",
            "Base_2_2/Zone_times",
            "Base_2_2/Zone/GridCoordinates",
            "Base_2_2/Zone/ZoneBC/Periodic_2/PointList",
            "Base_2_2_times",
            "Base_2_2/Zone/ZoneBC/Inflow/GridLocation",
            "Base_2_2/Zone/FamilyName",
            "Base_2_2/Zone/ZoneBC",
            "Base_2_2/Zone/GridCoordinates/CoordinateX_times",
            "Base_2_2/Zone/FamilyName_times",
            "Base_2_2/Zone/ZoneBC/Intrado/PointList",
            "Base_2_2/Zone/ZoneBC/Intrado/GridLocation_times",
            "Base_2_2/Zone/PointData/rov_times",
            "Base_2_2/Zone/ZoneBC/Intrado_times",
            "Base_2_2/Zone/ZoneBC/Inflow/PointList",
            "Base_2_2/Zone/ZoneBC/Outflow/PointList_times",
            "Base_2_2/Zone/GridCoordinates/CoordinateY",
            "Base_2_2/Zone/Elements_QUAD_4_times",
            "Base_2_2/Zone/ZoneBC/Inflow",
            "Base_2_2/Zone/ZoneBC/Outflow_times",
            "Base_2_2/Zone/ZoneBC/Extrado_times",
            "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity_times",
            "Base_2_2/Zone/ZoneBC/Periodic_1/PointList",
            "Base_2_2/Zone/ZoneBC/Periodic_1/PointList_times",
            "Base_2_2/Zone/ZoneBC/Intrado/PointList_times",
            "Base_2_2",
            "Base_2_2/Zone/Elements_QUAD_4",
        ]
        assert set(result) == set(ref)
        print("----------------")

        print("Test 5")
        features = ["Global/Pr", "Global/Q"]
        result = cgns_helper.update_features_for_CGNS_compatibility(
            features, context_constant_features, context_variable_features
        )
        ref = [
            "Global",
            "Global/Pr",
            "Global/Pr_times",
            "Global/Q",
            "Global/Q_times",
            "Global_times",
        ]
        assert set(result) == set(ref)
        print("----------------")
