# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

try: # pragma: no cover
    from typing import Self
except ImportError: # pragma: no cover
    from typing import Any as Self

import csv
import logging
import os
from typing import Union

import numpy as np
import yaml

# %% Globals

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s',
    level=logging.INFO)

# %% Functions

# %% Classes

authorized_tasks = ["regression", "classification"]
"""List containing authorized machine learning tasks.
"""
IndexType = Union[list[int], np.ndarray]
"""IndexType is an Union[list[int],np.ndarray]
"""


class ProblemDefinition(object):
    """Gathers all necessary informations to define a learning problem."""

    def __init__(self, directory_path: str = None) -> None:
        """Initialize an empty :class:`ProblemDefinition <plaid.problem_definition.ProblemDefinition>`

        Use :meth:`add_inputs <plaid.problem_definition.ProblemDefinition.add_inputs>` or :meth:`add_outputs <plaid.problem_definition.ProblemDefinition.add_outputs>` to feed the :class:`ProblemDefinition`

        Args:
            directory_path (str, optional): The path from which to load PLAID problem definition files.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition

                # 1. Create empty instance of ProblemDefinition
                problem_definition = ProblemDefinition()
                print(problem_definition)
                >>> ProblemDefinition()

                # 2. Load problem definition and create ProblemDefinition instance
                problem_definition = ProblemDefinition("path_to_plaid_prob_def")
                print(problem_definition)
                >>> ProblemDefinition(input_names=['mesh', 's_1'], output_names=['s_2'], task='regression')
        """
        self._task: str = None  # list[task name]
        self._inputs: list[str] = []   # list[input name]
        self._outputs: list[str] = []   # list[output name]
        self._split: dict[str, IndexType] = None

        self.in_scalars_names: list[str] = []
        self.out_scalars_names: list[str] = []
        self.in_fields_names: list[str] = []
        self.out_fields_names: list[str] = []

        if directory_path is not None:
            self._load_from_dir_(directory_path)

    # -------------------------------------------------------------------------#
    def get_task(self) -> str:
        """Get the authorized task. None if not defined.

        Returns:
            str: The authorized task, such as "regression" or "classification".
        """
        return self._task

    def set_task(self, task: str) -> None:
        """Set the authorized task.

        Args:
            The authorized task to be set, such as "regression" or "classification".
        """
        if self._task is not None:
            raise ValueError(
                f"A task is already in self._task: (`{self._task}`)")
        elif task in authorized_tasks:
            self._task = task
        else:
            raise TypeError(
                f"{task} not among authorized tasks. Maybe you want to try among: {authorized_tasks}")
    # -------------------------------------------------------------------------#

    def get_split(
            self, indices_name: str = None) -> Union[IndexType, dict[str, IndexType]]:
        """Get the split indices. This function returns the split indices, either for a specific split
            with the provided `indices_name` or all split indices if `indices_name` is not specified.

        Args:
            indices_name (str, optional): The name of the split for which indices are requested. Defaults to None.

        Raises:
            KeyError: If `indices_name` is specified but not found among split names.

        Returns:
            Union[IndexType,dict[str,IndexType]]: If `indices_name` is provided, it returns
            the indices for that split (IndexType). If `indices_name` is not provided, it
            returns a dictionary mapping split names (str) to their respective indices
            (IndexType).

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                split_indices = problem.get_split()
                print(split_indices)
                >>> {'train': [0, 1, 2, ...], 'test': [100, 101, ...]}

                test_indices = problem.get_split('test')
                print(test_indices)
                >>> [100, 101, ...]
        """
        if indices_name is None:
            return self._split
        else:
            assert indices_name in self._split, indices_name + " not among split indices names"
            return self._split[indices_name]

    def set_split(self, split: dict[str, IndexType]) -> None:
        """Set the split indices. This function allows you to set the split indices by providing a dictionary
        mapping split names (str) to their respective indices (IndexType).

        Args:
            split (dict[str,IndexType]):  A dictionary containing split names and their indices.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                new_split = {'train': [0, 1, 2], 'test': [3, 4]}
                problem.set_split(new_split)
        """
        if self._split is not None: # pragma: no cover
            logger.warning(f"split already exists -> data will be replaced")
        self._split = split
    # -------------------------------------------------------------------------#

    def get_inputs(self) -> list[str]:
        """Get the input names or identifiers of the problem.

        Returns:
            list[str]: A list of input feature names or identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_names = problem.get_inputs()
                print(input_names)
                >>> ['omega', 'pressure']
        """
        return self._inputs

    def add_inputs(self, inputs: list[str]) -> None:
        """Add input names or identifiers to the problem.

        Args:
            inputs (list[str]): A list of input feature names or identifiers to add.

        Raises:
            ValueError: If some :code:`inputs` are redondant.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                input_names = ['omega', 'pressure']
                problem.add_inputs(input_names)
        """
        if not (len(set(inputs)) == len(inputs)):
            raise ValueError('Some inputs have same names')
        for input in inputs:
            self.add_input(input)

    def add_input(self, input: str) -> None:
        """Add an input name or identifier to the problem.

        Args:
            input (str):  The name or identifier of the input feature to add.

        Raises:
            ValueError: If the specified input feature is already in the list of inputs.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                input_name = 'pressure'
                problem.add_input(input_name)
        """
        if input in self._inputs:
            raise ValueError(f"{input} is already in self._inputs")
        self._inputs.append(input)
        self._inputs.sort()

    # -------------------------------------------------------------------------#
    def get_outputs(self) -> list[str]:
        """Get the outputs names or identifiers of the problem.

        Returns:
            list[str]: A list of output feature names or identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                outputs_names = problem.get_outputs()
                print(outputs_names)
                >>> ['compression_rate', 'in_massflow', 'isentropic_efficiency']
        """
        return self._outputs

    def add_outputs(self, outputs: list[str]) -> None:
        """Add output names or identifiers to the problem.

        Args:
            outputs (list[str]): A list of output feature names or identifiers to add.

        Raises:
            ValueError: if some :code:`outputs` are redondant.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                output_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                problem.add_outputs(output_names)
        """
        if not (len(set(outputs)) == len(outputs)):
            raise ValueError('Some outputs have same names')
        for output in outputs:
            self.add_output(output)

    def add_output(self, output: str) -> None:
        """Add an output name or identifier to the problem.
        Args:
            output (str):  The name or identifier of the output feature to add.

        Raises:
            ValueError: If the specified output feature is already in the list of outputs.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                output_names = 'pressure'
                problem.add_output(output_names)
        """
        if output in self._outputs:
            raise ValueError(f"{output} is already in self._outputs")
        self._outputs.append(output)
        self._inputs.sort()

    def filter_input_names(self, names: list[str]) -> list[str]:
        """Filter and get input features corresponding to a sorted list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding input features.

        Returns:
            list[str]: A sorted list of input feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_names = ['omega', 'pressure', 'temperature']
                input_features = my_instance.get_inputs_from_names(input_names)
                print(input_features)
                >>> ['omega', 'pressure']
        """
        return sorted(set(names).intersection(self.get_inputs()))

    def filter_output_names(self, names: list[str]) -> list[str]:
        """Filter and get output features corresponding to a sorted list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding output features.

        Returns:
            list[str]: A sorted list of output feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                output_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                output_features = my_instance.get_outputs_from_names(output_names)
                print(output_features)
                >>> ['in_massflow']
        """
        return sorted(set(names).intersection(self.get_outputs()))

    def get_all_indices(self) -> list[int]:
        """Get all indices from splits.

        Returns:
            list[int]: list containing all unique indices.
        """
        all_indices = []
        for indices in self.get_split().values():
            all_indices += list(indices)
        return list(set(all_indices))

    # def get_input_scalars_to_tabular(self, sample_ids:list[int]=None, as_dataframe=True) -> dict[str, np.ndarray]:
    #     """Return a dict containing input scalar values as tabulars/arrays

    #     Returns:
    #         pandas.DataFrame: if as_dataframe is True
    #         dict[str,np.ndarray]: if as_dataframe is False, scalar’s ``feature_name`` -> tabular values
    #     """
    #     res = {}
    #     for _,feature_name in self.get_inputs(feature_type='scalar'):
    #         res.update(self.get_scalars_to_tabular(feature_name, sample_ids))

    #     if as_dataframe:
    #         res = pandas.DataFrame(res)

    #     return res

    # def get_output_scalars_to_tabular(self, sample_ids:list[int]=None, as_dataframe=True) -> dict[str, np.ndarray]:
    #     """Return a dict containing output scalar values as tabulars/arrays

    #     Returns:
    #         pandas.DataFrame: if as_dataframe is True
    #         dict[str,np.ndarray]: if as_dataframe is False, scalar’s ``feature_name`` -> tabular values
    #     """
    #     res = {}
    #     for _,feature_name in self.get_outputs(feature_type='scalar'):
    #         res.update(self.get_scalars_to_tabular(feature_name, sample_ids))

    #     if as_dataframe:
    #         res = pandas.DataFrame(res)

    #     return res

    # -------------------------------------------------------------------------#
    def _save_to_dir_(self, savedir: str) -> None:
        """Save problem information, inputs, outputs, and split to the specified directory in YAML and CSV formats.

        Args:
            savedir (str): The directory where the problem information will be saved.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                problem._save_to_dir_("/path/to/save_directory")
        """
        if not (os.path.isdir(savedir)): # pragma: no cover
            os.makedirs(savedir)

        data = {
            "task": self._task,
            "inputs": self._inputs,     # list[input name]
            "outputs": self._outputs
        }

        pbdef_fname = os.path.join(savedir, 'problem_infos.yaml')
        with open(pbdef_fname, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

        split_fname = os.path.join(savedir, 'split.csv')
        if self._split is not None:
            with open(split_fname, 'w') as file:
                write = csv.writer(file)
                for name, indices in self._split.items():
                    write.writerow([name] + list(indices))

    @classmethod
    def load(cls, save_dir: str) -> Self: # pragma: no cover
        """Load data from a specified directory.

        Args:
            save_dir (str): The path from which to load files.

        Returns:
            Self: The loaded dataset (Dataset).
        """
        instance = cls()
        instance._load_from_dir_(save_dir)
        return instance

    def _load_from_dir_(self, save_dir: str) -> None:
        """Load problem information, inputs, outputs, and split from the specified directory in YAML and CSV formats.

        Args:
            savedir (str): The directory from which to load the problem information.

        Raises:
            FileNotFoundError: Triggered if the provided directory does not exist.
            FileExistsError: Triggered if the provided path is a file instead of a directory.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                problem._load_from_dir_("/path/to/load_directory")
        """
        if not os.path.exists(save_dir): # pragma: no cover
            raise FileNotFoundError(
                f"Directory \"{save_dir}\" does not exist. Abort")

        if not os.path.isdir(save_dir): # pragma: no cover
            raise FileExistsError(f"\"{save_dir}\" is not a directory. Abort")

        pbdef_fname = os.path.join(save_dir, 'problem_infos.yaml')
        data = {}  # To avoid crash if pbdef_fname does not exist
        if os.path.isfile(pbdef_fname):
            with open(pbdef_fname, 'r') as file:
                data = yaml.safe_load(file)
        else: # pragma: no cover
            logger.warning(
                f"file with path `{pbdef_fname}` does not exist. Task, inputs, and outputs will not be set")

        self._task = data["task"]
        self._inputs = data["inputs"]
        self._outputs = data["outputs"]

        split_fname = os.path.join(save_dir, 'split.csv')
        split = {}
        if os.path.isfile(split_fname):
            with open(split_fname) as file:
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    split[row[0]] = [int(i) for i in row[1:]]
        else: # pragma: no cover
            logger.warning(
                f"file with path `{split_fname}` does not exist. Splits will not be set")
        self._split = split

    # -------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """Return a string representation of the problem.

        Returns:
            str: A string representation of the overview of problem content.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                print(problem)
                >>> ProblemDefinition(input_names=['mesh', 's_1'], output_names=['s_2'], task='regression', split_names=['train', 'val'])
        """
        str_repr = "ProblemDefinition("
        if len(self._inputs) > 0:
            input_names = self._inputs
            str_repr += f"{input_names=}, "
        if len(self._outputs) > 0:
            output_names = self._outputs
            str_repr += f"{output_names=}, "
        if self._task is not None:
            task = self._task
            str_repr += f"{task=}, "
        if self._split is not None:
            split_names = list(self._split.keys())
            str_repr += f"{split_names=}, "
        if str_repr[-2:] == ', ':
            str_repr = str_repr[:-2]
        str_repr += ")"
        return str_repr
