# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import pytest
import os
import subprocess
from plaid.problem_definition import ProblemDefinition


# %% Fixtures

@pytest.fixture()
def problem_definition() -> ProblemDefinition:
    return ProblemDefinition()

@pytest.fixture()
def current_directory() -> str:
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session", autouse=True)
def clean_tests():
    if os.name == 'nt':
        # Windows
        retcode = subprocess.call(['cmd', '/c', 'tests\\clean.bat'])
    else:
        # Unix
        retcode = subprocess.call(['sh', './tests/clean.sh'])


# %% Tests

class Test_ProblemDefinition:
    def test__init__(self, problem_definition):
        assert problem_definition.get_task() is None
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_task(self, problem_definition):
        # Unauthorized task
        with pytest.raises(TypeError):
            problem_definition.set_task("ighyurgv")
        problem_definition.set_task("classification")
        with pytest.raises(ValueError):
            problem_definition.set_task("regression")
        assert problem_definition.get_task() == "classification"
        print(problem_definition)

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #-------------------------------------------------------------------------#
    def test_get_input_scalars_names(self, problem_definition):
        assert(problem_definition.get_input_scalars_names()==[])

    def test_add_input_scalars_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_scalars_names(['feature_name','feature_name'])
        problem_definition.add_input_scalar_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_input_scalar_name('feature_name')

    def test_add_input_scalars_names(self, problem_definition):
        problem_definition.add_input_scalars_names(['scalar', 'test_scalar'])
        problem_definition.add_input_scalar_name('predict_scalar')
        inputs = problem_definition.get_input_scalars_names()
        assert len(inputs) == 3
        assert set(inputs) == set(['predict_scalar', 'scalar', 'test_scalar'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_get_output_scalars_names(self, problem_definition):
        assert(problem_definition.get_output_scalars_names()==[])

    def test_add_output_scalars_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_scalars_names(['feature_name','feature_name'])
        problem_definition.add_output_scalar_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_output_scalar_name('feature_name')

    def test_add_output_scalars_names(self, problem_definition):
        problem_definition.add_output_scalars_names(['scalar', 'test_scalar'])
        problem_definition.add_output_scalar_name('predict_scalar')
        outputs = problem_definition.get_output_scalars_names()
        assert len(outputs) == 3
        assert set(outputs) == set(['predict_scalar', 'scalar', 'test_scalar'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_filter_scalars_names(self, current_directory):
        d_path = os.path.join(current_directory, "problem_definition")
        problem = ProblemDefinition(d_path)
        filter_in = problem.filter_input_scalars_names(['predict_scalar', 'test_scalar'])
        filter_out = problem.filter_output_scalars_names(['predict_scalar', 'test_scalar'])
        assert len(filter_in) == 2 and filter_in == ['predict_scalar', 'test_scalar']
        assert filter_in != ['test_scalar', 'predict_scalar'], "common inputs not sorted"

        assert len(filter_out) == 2 and filter_out == ['predict_scalar', 'test_scalar']
        assert filter_out != ['test_scalar', 'predict_scalar'], "common outputs not sorted"

        fail_filter_in = problem.filter_input_scalars_names(['a_scalar'])
        fail_filter_out = problem.filter_output_scalars_names(['b_scalar'])

        assert fail_filter_in == []
        assert fail_filter_out == []

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #-------------------------------------------------------------------------#
    def test_get_input_fields_names(self, problem_definition):
        assert(problem_definition.get_input_fields_names()==[])

    def test_add_input_fields_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_fields_names(['feature_name','feature_name'])
        problem_definition.add_input_field_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_input_field_name('feature_name')

    def test_add_input_fields_names(self, problem_definition):
        problem_definition.add_input_fields_names(['field', 'test_field'])
        problem_definition.add_input_field_name('predict_field')
        inputs = problem_definition.get_input_fields_names()
        assert len(inputs) == 3
        assert set(inputs) == set(['predict_field', 'field', 'test_field'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_get_output_fields_names(self, problem_definition):
        assert(problem_definition.get_output_fields_names()==[])

    def test_add_output_fields_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_fields_names(['feature_name','feature_name'])
        problem_definition.add_output_field_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_output_field_name('feature_name')

    def test_add_output_fields_names(self, problem_definition):
        problem_definition.add_output_fields_names(['field', 'test_field'])
        problem_definition.add_output_field_name('predict_field')
        outputs = problem_definition.get_output_fields_names()
        assert len(outputs) == 3
        assert set(outputs) == set(['predict_field', 'field', 'test_field'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_filter_fields_names(self, current_directory):
        d_path = os.path.join(current_directory, "problem_definition")
        problem = ProblemDefinition(d_path)
        filter_in = problem.filter_input_fields_names(['predict_field', 'test_field'])
        filter_out = problem.filter_output_fields_names(['predict_field', 'test_field'])
        assert len(filter_in) == 2 and filter_in == ['predict_field', 'test_field']
        assert filter_in != ['test_field', 'predict_field'], "common inputs not sorted"

        assert len(filter_out) == 2 and filter_out == ['predict_field', 'test_field']
        assert filter_out != ['test_field', 'predict_field'], "common outputs not sorted"

        fail_filter_in = problem.filter_input_fields_names(['a_field'])
        fail_filter_out = problem.filter_output_fields_names(['b_field'])

        assert fail_filter_in == []
        assert fail_filter_out == []

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #-------------------------------------------------------------------------#
    def test_get_input_timeseries_names(self, problem_definition):
        assert(problem_definition.get_input_timeseries_names()==[])

    def test_add_input_timeseries_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_timeseries_names(['feature_name','feature_name'])
        problem_definition.add_input_timeseries_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_input_timeseries_name('feature_name')

    def test_add_input_timeseries_names(self, problem_definition):
        problem_definition.add_input_timeseries_names(['timeseries', 'test_timeseries'])
        problem_definition.add_input_timeseries_name('predict_timeseries')
        inputs = problem_definition.get_input_timeseries_names()
        assert len(inputs) == 3
        assert set(inputs) == set(['predict_timeseries', 'timeseries', 'test_timeseries'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_get_output_timeseries_names(self, problem_definition):
        assert(problem_definition.get_output_timeseries_names()==[])

    def test_add_output_timeseries_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_timeseries_names(['feature_name','feature_name'])
        problem_definition.add_output_timeseries_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_output_timeseries_name('feature_name')

    def test_add_output_timeseries_names(self, problem_definition):
        problem_definition.add_output_timeseries_names(['timeseries', 'test_timeseries'])
        problem_definition.add_output_timeseries_name('predict_timeseries')
        outputs = problem_definition.get_output_timeseries_names()
        assert len(outputs) == 3
        assert set(outputs) == set(['predict_timeseries', 'timeseries', 'test_timeseries'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_filter_timeseries_names(self, current_directory):
        d_path = os.path.join(current_directory, "problem_definition")
        problem = ProblemDefinition(d_path)
        filter_in = problem.filter_input_timeseries_names(['predict_timeseries', 'test_timeseries'])
        filter_out = problem.filter_output_timeseries_names(['predict_timeseries', 'test_timeseries'])
        assert len(filter_in) == 2 and filter_in == ['predict_timeseries', 'test_timeseries']
        assert filter_in != ['test_timeseries', 'predict_timeseries'], "common inputs not sorted"

        assert len(filter_out) == 2 and filter_out == ['predict_timeseries', 'test_timeseries']
        assert filter_out != ['test_timeseries', 'predict_timeseries'], "common outputs not sorted"

        fail_filter_in = problem.filter_input_timeseries_names(['a_timeseries'])
        fail_filter_out = problem.filter_output_timeseries_names(['b_timeseries'])

        assert fail_filter_in == []
        assert fail_filter_out == []

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    #-------------------------------------------------------------------------#
    def test_get_input_meshes_names(self, problem_definition):
        assert(problem_definition.get_input_meshes_names()==[])

    def test_add_input_meshes_names_fail_same_name(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_input_meshes_names(['feature_name','feature_name'])
        problem_definition.add_input_mesh_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_input_mesh_name('feature_name')

    def test_add_input_meshes_names(self, problem_definition):
        problem_definition.add_input_meshes_names(['mesh', 'test_mesh'])
        problem_definition.add_input_mesh_name('predict_mesh')
        inputs = problem_definition.get_input_meshes_names()
        assert len(inputs) == 3
        assert set(inputs) == set(['predict_mesh', 'mesh', 'test_mesh'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_get_output_meshes_names(self, problem_definition):
        assert(problem_definition.get_output_meshes_names()==[])

    def test_add_output_meshes_names_fail(self, problem_definition):
        with pytest.raises(ValueError):
            problem_definition.add_output_meshes_names(['feature_name','feature_name'])
        problem_definition.add_output_mesh_name('feature_name')
        with pytest.raises(ValueError):
            problem_definition.add_output_mesh_name('feature_name')

    def test_add_output_meshes_names(self, problem_definition):
        problem_definition.add_output_meshes_names(['mesh', 'test_mesh'])
        problem_definition.add_output_mesh_name('predict_mesh')
        outputs = problem_definition.get_output_meshes_names()
        assert len(outputs) == 3
        assert set(outputs) == set(['predict_mesh', 'mesh', 'test_mesh'])
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_filter_meshes_names(self, current_directory):
        d_path = os.path.join(current_directory, "problem_definition")
        problem = ProblemDefinition(d_path)
        print(f"{problem=}")
        print(f"{problem.get_input_meshes_names()=}")
        filter_in = problem.filter_input_meshes_names(['predict_mesh', 'test_mesh'])
        filter_out = problem.filter_output_meshes_names(['predict_mesh', 'test_mesh'])
        assert len(filter_in) == 2 and filter_in == ['predict_mesh', 'test_mesh']
        assert filter_in != ['test_mesh', 'predict_mesh'], "common inputs not sorted"

        assert len(filter_out) == 2 and filter_out == ['predict_mesh', 'test_mesh']
        assert filter_out != ['test_mesh', 'predict_mesh'], "common outputs not sorted"

        fail_filter_in = problem.filter_input_meshes_names(['a_mesh'])
        fail_filter_out = problem.filter_output_meshes_names(['b_mesh'])

        assert fail_filter_in == []
        assert fail_filter_out == []

    #-------------------------------------------------------------------------#
    def test_set_split(self, problem_definition):
        new_split = {'train': [0, 1, 2], 'test': [3, 4]}
        problem_definition.set_split(new_split)
        assert problem_definition.get_split('train') == [0, 1, 2]
        assert problem_definition.get_split('test') == [3, 4]

        all_split = problem_definition.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]
        assert problem_definition.get_all_indices() == [0, 1, 2, 3, 4]
        print(problem_definition)

    #-------------------------------------------------------------------------#
    def test_save(self, problem_definition, current_directory):
        problem_definition.set_task("regression")

        problem_definition.add_input_scalars_names(['scalar', 'test_scalar'])
        problem_definition.add_input_scalar_name('predict_scalar')
        problem_definition.add_output_scalars_names(['scalar', 'test_scalar'])
        problem_definition.add_output_scalar_name('predict_scalar')

        problem_definition.add_input_fields_names(['field', 'test_field'])
        problem_definition.add_input_field_name('predict_field')
        problem_definition.add_output_fields_names(['field', 'test_field'])
        problem_definition.add_output_field_name('predict_field')

        problem_definition.add_input_timeseries_names(['timeseries', 'test_timeseries'])
        problem_definition.add_input_timeseries_name('predict_timeseries')
        problem_definition.add_output_timeseries_names(['timeseries', 'test_timeseries'])
        problem_definition.add_output_timeseries_name('predict_timeseries')

        problem_definition.add_input_meshes_names(['mesh', 'test_mesh'])
        problem_definition.add_input_mesh_name('predict_mesh')
        problem_definition.add_output_meshes_names(['mesh', 'test_mesh'])
        problem_definition.add_output_mesh_name('predict_mesh')

        new_split = {'train': [0, 1, 2], 'test': [3, 4]}
        problem_definition.set_split(new_split)

        problem_definition._save_to_dir_(os.path.join(current_directory, "problem_definition"))

    def test_load_path_object(self, current_directory):
        from pathlib import Path
        my_dir = Path(current_directory)
        ProblemDefinition(my_dir / 'problem_definition')

    def test_load(self, current_directory):
        d_path = os.path.join(current_directory, "problem_definition")
        problem = ProblemDefinition(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_input_scalars_names()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        assert set(problem.get_output_scalars_names()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        all_split = problem.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]

        problem = ProblemDefinition()
        problem._load_from_dir_(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_input_scalars_names()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        assert set(problem.get_output_scalars_names()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        all_split = problem.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]

        problem = ProblemDefinition.load(d_path)
        assert problem.get_task() == "regression"
        assert set(problem.get_input_scalars_names()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        assert set(problem.get_output_scalars_names()) == set(['predict_scalar', 'scalar', 'test_scalar'])
        all_split = problem.get_split()
        assert all_split['train'] == [0, 1, 2] and all_split['test'] == [3, 4]
