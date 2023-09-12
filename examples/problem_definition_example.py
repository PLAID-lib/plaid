# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import os

import numpy as np

from plaid.containers.dataset import Dataset, Sample
from plaid.problem_definition import ProblemDefinition
from plaid.utils.split import split_dataset

# %% Functions


def main():

    # %% Init

    print("#---# Empty ProblemDefinition")
    problem = ProblemDefinition()
    print(f"{problem=}")

    problem.add_input('in')
    problem.add_inputs(['in2', 'in3'])
    problem.add_output('out')
    problem.add_outputs(['out2'])

    print("problem.get_inputs() =", problem.get_inputs())
    print("problem.get_outputs() =", problem.get_outputs())

    problem.set_task('regression')
    print(problem.get_task())

    options = {
        'shuffle': False,
        'split_sizes': {
            'train': 2,
            'val': 1,
        },
    }
    dset = Dataset()
    dset.add_samples([Sample(), Sample(), Sample(), Sample()])
    split = split_dataset(dset, options)

    problem.set_split(split)
    print("problem.get_split() =", problem.get_split())

    test_pth = os.path.join(
        f'/tmp/test_safe_to_delete_{np.random.randint(1e10, 1e12)}', 'test')
    os.makedirs(test_pth)
    print()
    print("-" * 80)
    print(f"--- problem.save({test_pth})")
    problem._save_to_dir_(test_pth)

    print()
    print("-" * 80)
    print("--- new_sample.load(os.path.join(test_pth, 'test'))")
    problem = ProblemDefinition()
    problem._load_from_dir_(test_pth)
    print(problem)


# %% Main Script
if __name__ == '__main__':
    main()

    print()
    print("#==============#")
    print("#===# DONE #===#")
    print("#==============#")
