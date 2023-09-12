# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import pytest

from plaid.problem_definition import ProblemDefinition

# %% Fixtures


@pytest.fixture()
def problem_definition():
    return ProblemDefinition()

# %% Tests


class Test_ProblemDefinition():
    def test__init__(self, problem_definition):
        pass

    # #---# Inputs
    # def test_get_inputs(self, dataset):
    #     assert(dataset.get_inputs()==[])

    # def test_add_inputs_fail(self, dataset):
    #     with pytest.raises(ValueError):
    #         dataset.add_inputs(['feature_name','feature_name'])
    # def test_add_inputs(self, dataset_with_samples):
    #     dataset_with_samples.add_inputs([('scalar', 'test_scalar')])

    # def test_add_input(self, dataset_with_samples):
    #     dataset_with_samples.add_input(feature_type='scalar', feature_name='test_scalar')
    # def test_add_input_no_type(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         dataset_with_samples.add_input(feature_type='spatial_support', feature_name='test_scalar1')
    # def test_add_input_no_name(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         dataset_with_samples.add_input(feature_type='scalar', feature_name='missing_scalar_name')
    # def test_add_input_already_present(self, dataset_with_samples):
    #     dataset_with_samples.add_input(feature_type='scalar', feature_name='test_scalar')
    #     with pytest.raises(ValueError):
    #         dataset_with_samples.add_input(feature_type='scalar', feature_name='test_scalar')

    # #---# Outputs
    # def test_get_outputs(self, dataset):
    #     assert(dataset.get_outputs()==[])

    # def test_add_outputs_fail(self, dataset):
    #     with pytest.raises(ValueError):
    #         dataset.add_outputs(['feature_name','feature_name'])
    # def test_add_outputs(self, dataset_with_samples):
    #     dataset_with_samples.add_outputs([('scalar', 'test_scalar')])

    # def test_add_output(self, dataset_with_samples):
    #     dataset_with_samples.add_output(feature_type='scalar', feature_name='test_scalar')
    # def test_add_output_no_type(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         dataset_with_samples.add_output(feature_type='spatial_support', feature_name='test_scalar')
    # def test_add_output_no_name(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         dataset_with_samples.add_output(feature_type='scalar', feature_name='missing_scalar_name')
    # def test_add_output_already_present(self, dataset_with_samples):
    #     dataset_with_samples.add_output(feature_type='scalar', feature_name='test_scalar')
    #     with pytest.raises(ValueError):
    #         dataset_with_samples.add_output(feature_type='scalar', feature_name='test_scalar')
