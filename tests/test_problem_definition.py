# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import pytest
import os

from plaid.problem_definition import ProblemDefinition
# %% Fixtures


@pytest.fixture()
def problem_definition():
    return ProblemDefinition()

@pytest.fixture()
def current_directory():
    return os.path.dirname(os.path.abspath(__file__))

# %% Tests

class Test_ProblemDefinition():
    def test__init__(self, problem_definition):
        assert problem_definition.get_task() is None
        print(problem_definition)

    def test_task(self, problem_definition):
        # Unauthorized task
        with pytest.raises(TypeError):
            problem_definition.set_task("ighyurgv")
        problem_definition.set_task("classification")
        with pytest.raises(ValueError):
            problem_definition.set_task("regression")
        assert problem_definition.get_task() == "classification"
        print(problem_definition)

    def test_get_inputs(self, problem_definition):
        assert(problem_definition.get_inputs()==[])

    def test_add_inputs_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_inputs(['feature_name','feature_name'])
        problem_definition.add_input('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_input('feature_name')

    def test_add_inputs(self, problem_definition):
        problem_definition.add_inputs(['scalar', 'test_scalar'])
        problem_definition.add_input('predict_scalar')
        inputs = problem_definition.get_inputs()
        assert len(inputs) == 3
        assert set(inputs) == set(['predict_scalar', 'scalar', 'test_scalar'])
        print(problem_definition)


    def test_get_outputs(self, problem_definition):
        assert(problem_definition.get_outputs()==[])

    def test_add_outputs_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_outputs(['feature_name','feature_name'])
        problem_definition.add_output('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_output('feature_name')

    def test_add_outputs(self, problem_definition):
        problem_definition.add_outputs(['scalar', 'test_scalar'])
        problem_definition.add_output('predict_scalar')
        outputs = problem_definition.get_outputs()
        assert len(outputs) == 3
        assert set(outputs) == set(['predict_scalar', 'scalar', 'test_scalar'])
        print(problem_definition)

    def test_set_split(self, problem_definition):
        new_split = {'train': [0, 1, 2], 'test': [3, 4]}
        problem_definition.set_split(new_split)
        assert problem_definition.get_split('train') == [0, 1, 2]
        assert problem_definition.get_split('test') == [3, 4]

        all_split = problem_definition.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]
        assert problem_definition.get_all_indices() == [0, 1, 2, 3, 4]
        print(problem_definition)

    def test_save(self, problem_definition, current_directory):
        problem_definition.set_task("regression")

        problem_definition.add_inputs(['scalar', 'test_scalar'])
        problem_definition.add_input('predict_scalar')

        problem_definition.add_outputs(['scalar', 'test_scalar'])
        problem_definition.add_output('predict_scalar')

        new_split = {'train': [0, 1, 2], 'test': [3, 4]}
        problem_definition.set_split(new_split)

        problem_definition._save_to_dir_(os.path.join(current_directory, "problem_definition"))

    def test_load(self, current_directory):
        d_path = os.path.join(current_directory, "problem_definition")
        problem = ProblemDefinition(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_inputs()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        assert set(problem.get_outputs()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        all_split = problem.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]

        problem = ProblemDefinition()
        problem._load_from_dir_(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_inputs()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        assert set(problem.get_outputs()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        all_split = problem.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]

        problem = ProblemDefinition.load(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_inputs()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        assert set(problem.get_outputs()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        all_split = problem.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]

    def test_filter(self, current_directory):
        d_path = os.path.join(current_directory, "problem_definition")
        problem = ProblemDefinition(d_path)
        filter_in = problem.filter_input_names(['predict_scalar', 'test_scalar'])
        filter_out = problem.filter_output_names(['predict_scalar', 'test_scalar'])
        assert len(filter_in) == 2 and filter_in == ['predict_scalar', 'test_scalar']
        assert filter_in != ['test_scalar', 'predict_scalar'], "common inputs not sorted"

        assert len(filter_out) == 2 and filter_out == ['predict_scalar', 'test_scalar']
        assert filter_out != ['test_scalar', 'predict_scalar'], "common outputs not sorted"

        fail_filter_in = problem.filter_input_names(['a_scalar'])
        fail_filter_out = problem.filter_output_names(['b_scalar'])

        assert fail_filter_in == []
        assert fail_filter_out == []