# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import os

import BasicTools.Containers.ElementNames as ElementNames
import numpy as np
from BasicTools.Bridges.CGNSBridge import MeshToCGNS
from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample

# %% Functions


def dataset_examples():
    """
    This function shows various use cases for the Dataset classe.

    Example Usage:

    1. Initializing an Empty Dataset and Adding Samples:
    - Initialize an empty Dataset and add Samples.

    2. Retrieving and Manipulating Samples:
    - Create and add Samples to the Dataset.
    - Add tree structures and scalars to Samples.
    - Retrieve and manipulate Sample data within the Dataset.

    3. Performing Operations on the Dataset:
    - Add Samples to the Dataset, add information, and access data.
    - Perform operations like merging datasets, adding tabular scalars, and setting information.

    4. Saving and Loading Datasets:
    - Save and load datasets from directories or files.

    This function provides detailed examples of using the Dataset class to manage data, samples,
    and information within the dataset. It is intended for documentation purposes and
    familiarization with the PLAID library.
    """
    # %% Init

    print("#---# Empty Dataset")
    dataset = Dataset()
    print(f"{dataset=}")

    # %% Feed Samples
    # Example of creating Samples.

    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.5],
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [2, 4, 3],
    ])
    bars = np.array([
        [0, 1],
        [0, 2]
    ])
    BTMesh = UMCT.CreateMeshOfTriangles(points, triangles)
    elbars = BTMesh.GetElementsOfType(ElementNames.Bar_2)
    elbars.AddNewElements(bars, [1, 2])
    cgns_mesh = MeshToCGNS(BTMesh)

    print("#---# Empty Sample")
    sample_01 = Sample()
    print(f"{sample_01=}")

    sample_01.add_tree(cgns_mesh)
    print(f"{sample_01=}")
    # temporal_support_ref = sample_01.add_feature("temporal_support", "t", np.random.randn(10))
    # print(f"{sample_01=}")
    # print(f"{temporal_support_ref=}")
    scalar_ref_01 = sample_01.add_scalar('rotation', np.random.randn())
    print(f"{sample_01=}")
    print(f"{scalar_ref_01=}")

    print("#---# Empty Sample")
    sample_02 = Sample()
    print(f"{sample_02=}")
    scalar_ref_02 = sample_02.add_scalar('rotation', np.random.randn())
    print(f"{sample_02=}")
    print(f"{scalar_ref_02=}")

    print("#---# Empty Sample")
    sample_03 = Sample()
    sample_03.add_scalar('speed', np.random.randn())
    sample_03.add_scalar('rotation', sample_01.get_scalar('rotation'))
    sample_03.add_tree(cgns_mesh)
    sample_03.show_tree()
    sample_03.add_field('temperature', np.random.rand(5), "Zone", "Base_2_2")
    sample_03.show_tree()
    print(f"{sample_03=}")
    print(f"{sample_03.get_scalar('speed')=}")
    print(f"{sample_03.get_scalar('rotation')=}")
    print(f"{sample_03.get_scalar_names()=}")

    # %% Feed Dataset
    # Example of adding Samples to the Dataset, adding information, and accessing data.

    dataset.set_sample(id=0, sample=sample_01)
    dataset.set_sample(id=1, sample=sample_02)
    dataset.add_sample(sample_03)

    dataset.add_info("legal", "owner", "Safran")
    infos = {"legal": {"owner": "Safran", "license": "CC0"}}
    dataset.set_infos(infos)
    print("dataset.get_infos() =", dataset.get_infos())

    print()
    print("===")

    print("length of dataset =", len(dataset))
    print("first sample =", dataset[0].get_scalar_names())
    print("second sample =", dataset[1].get_scalar_names())
    print("third sample =", dataset[2].get_scalar_names())
    print("get_samples_from_ids =", dataset.get_samples(ids=[0, 1]))

    print()
    print("===")

    print("get_sample_ids =", dataset.get_sample_ids())

    print()
    print("===")
    print(f"{dataset=}")

    print()
    print("===")

    dataset = Dataset()
    samples = [sample_01, sample_02, sample_03]
    dataset.add_samples(samples)
    print(f"{dataset=}")

    print(f"{dataset=}")
    print(f"{dataset[0]=}")
    print(f"{dataset[1]=}")
    print(f"{dataset[2]=}")
    print(dataset[0].get_scalar("rotation"))
    print(dataset[1].get_scalar("rotation"))
    print(dataset[2].get_scalar("rotation"))

    print()
    print("===")

    print("dataset.get_scalars_to_tabular(['rotation']):")
    print(dataset.get_scalars_to_tabular(['rotation']))

    print("dataset.get_scalars_to_tabular(['speed']):")
    print(dataset.get_scalars_to_tabular(['speed']))

    print("dataset.get_scalars_to_tabular(['speed', 'rotation']):")
    print(dataset.get_scalars_to_tabular(['speed', 'rotation']))

    print(dataset.get_scalar_names())

    print("dataset.get_scalars_to_tabular():")
    print(dataset.get_scalars_to_tabular())

    # %% Various operations on the Dataset
    # Example of operations like merging datasets, adding tabular scalars, and setting information.

    print()
    print("===")
    other_dataset = Dataset()
    nb_samples = 3
    samples = []
    for _ in range(nb_samples):
        sample = Sample()
        sample.add_scalar('rotation', np.random.rand() + 1.0)
        sample.add_scalar('random_name', np.random.rand() - 1.0)
        samples.append(sample)
    other_dataset.add_samples(samples)
    print("dataset.merge_dataset(other_dataset):")
    dataset.merge_dataset(other_dataset)
    print(dataset)
    print(dataset.get_scalars_to_tabular())

    print()
    print("===")
    new_scalars = np.random.rand(3, 2)
    dataset.add_tabular_scalars(new_scalars, names=['Tu', 'random_name'])
    print(dataset)
    print(dataset.get_scalars_to_tabular())

    print()
    print("===")
    infos = {
        "legal": {
            "owner": "Safran",
            "license": "CC0"},
        "data_production": {
            "type": "simulation",
            "simulator": "dummy"}
    }
    dataset.set_infos(infos)
    print(dataset)

    print()
    print("===")

    print()
    print("===")

    # %% Saving and Loading Dataset
    # Example of saving and loading a dataset from a directory or file.

    tmpdir = f'/tmp/test_safe_to_delete_{np.random.randint(1e10,1e12)}'
    print(f"Test save in: {tmpdir}")
    dataset._save_to_dir_(tmpdir)
    print(f"{os.listdir(tmpdir)=}")

    print()
    print("===")
    dataset2 = Dataset(tmpdir)
    print(f"{dataset2=}")

    print()
    print("===")
    dataset3 = Dataset.load_from_dir(tmpdir)
    print(f"{dataset3=}")

    print()
    print("===")
    dataset4 = Dataset()
    dataset4._load_from_dir_(tmpdir)
    print(f"{dataset4=}")

    print()
    print("===")
    tmpdir = f'/tmp/test_safe_to_delete_{np.random.randint(1e10,1e12)}'
    tmpfile = os.path.join(tmpdir, 'test_file.plaid')
    print(f"Test save in: {tmpfile}")
    dataset.save(tmpfile)

    # print(f"{os.listdir(tmpdir)=}")
    new_dataset = Dataset()
    new_dataset.load(tmpfile)
    print(f"{dataset=}")
    print(f"{new_dataset=}")


# %% Main Script
if __name__ == '__main__':
    dataset_examples()

    print()
    print("#==============#")
    print("#===# DONE #===#")
    print("#==============#")
