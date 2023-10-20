# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np
from BasicTools.Bridges.CGNSBridge import MeshToCGNS
from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT

from plaid.containers.sample import Sample, show_cgns_tree

# %% Functions


def sample_examples():
    """
    This function shows the usage of various operations and methods involving a sample data structure.

    Example Usage:

    1. Initializing an Empty Sample and Adding Data:
    - Initialize an empty Sample and add scalars and time series data.

    2. Accessing and Modifying Sample Data:
    - Add scalar and field data to the Sample.
    - Access and modify scalar and field data within the Sample.

    3. Creating a Sample Hierarchy:
    - Create a sample hierarchy with bases, zones, and associated data.

    4. Saving and Loading Samples:
    - Save and load Samples from files or directories.

    This function provides detailed examples of using the Sample class to manage and manipulate
    sample data structures. It is intended for documentation purposes and familiarization with
    the PLAID library.
    """

    # %% Input data
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
    BTMesh = UMCT.CreateMeshOfTriangles(points, triangles)
    BTMesh.nodeFields['test_node_field_1'] = np.random.randn(5)
    BTMesh.elemFields['test_elem_field_1'] = np.random.randn(3)
    tree = MeshToCGNS(BTMesh)

    show_cgns_tree(tree)

    def show_sample(sample):
        print(f"{sample=}")
        sample.show_tree()
        print(f"{sample.get_scalar_names()=}")
        print(f"{sample.get_field_names()=}")

    # ---------------------------#
    # %% Initialize Sample
    print()
    print("-" * 80)
    print("--- sample = Sample()")
    sample = Sample()
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.add_scalar('rotation', np.random.randn())")
    sample.add_scalar('rotation', np.random.randn())
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.add_scalar('speed', np.random.randn())")
    sample.add_scalar('speed', np.random.randn())
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.add_scalar('autre', np.random.randn())")
    sample.add_scalar('autre', np.random.randn())
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.add_time_series('truc', np.random.randn())")
    sample.add_time_series('truc', np.arange(10), np.random.randn(10))
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.add_time_series('much', np.random.randn())")
    sample.add_time_series('much', np.arange(2, 6), np.random.randn(4))
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.init_base(2, 3, 'SurfaceMesh')")
    sample.init_base(2, 3, 'SurfaceMesh', time=0.)
    show_sample(sample)

    shape = np.array((len(points), len(triangles), 0))
    print()
    print("-" * 80)
    print("--- sample.init_zone('TestZoneName', shape, base_name='SurfaceMesh', time = 0.)")
    sample.init_zone('TestZoneName', shape, base_name='SurfaceMesh', time=0.)
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.set_nodes(points, base_name='SurfaceMesh', zone_name='TestZoneName', time = 0.)")
    sample.set_nodes(
        points,
        base_name='SurfaceMesh',
        zone_name='TestZoneName',
        time=0.)
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.add_field('Pressure', np.random.randn(len(points)), base_name='SurfaceMesh', zone_name='TestZoneName', time = 0.)")
    sample.add_field(
        'Pressure',
        np.random.randn(
            len(points)),
        base_name='SurfaceMesh',
        zone_name='TestZoneName',
        time=0.)
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- sample.add_field('Temperature', np.random.randn(len(points)), base_name='SurfaceMesh', zone_name='TestZoneName', time = 0.)")
    sample.add_field(
        'Temperature',
        np.random.randn(
            len(points)),
        base_name='SurfaceMesh',
        zone_name='TestZoneName',
        time=0.)
    show_sample(sample)

    print()
    print("-" * 80)
    print("--- Access Data")

    print()
    print(f"{sample.get_scalar_names()=}")

    print()
    print(f"{sample.get_scalar('omega')=}")

    print()
    print(f"{sample.get_scalar('rotation')=}")

    print()
    print(f"{sample.get_nodes()=}")
    print(f"{sample.get_points()=}")
    print(f"{sample.get_vertices()=}")

    print()
    print(f"{sample.get_field_names()=}")

    print()
    print(f"{sample.get_field('T')=}")

    print()
    print(f"{sample.get_field('Temperature')=}")

    print()
    bases_names = sample.get_base_names()
    full_bases_names = sample.get_base_names(full_path=True)
    print(f"{bases_names=}")
    print(f"{full_bases_names=}")
    for b_name in bases_names:
        zones_names = sample.get_zone_names(b_name)
        full_zones_names = sample.get_zone_names(b_name, full_path=True)
        print(f" - Base : {b_name}")
        for z_name, f_z_name in zip(zones_names, full_zones_names):
            print(
                f"    - {z_name} -> type: {sample.get_zone_type(z_name, b_name)} | full: {f_z_name}")

    test_pth = f'/tmp/test_safe_to_delete_{np.random.randint(1e10, 1e12)}'
    import os
    os.makedirs(test_pth)
    test_fname = os.path.join(test_pth, 'test')
    print()
    print("-" * 80)
    print(f"--- sample.save({test_fname})")
    sample.save(test_fname)

    print()
    print("-" * 80)
    print("--- new_sample.load(os.path.join(test_pth, 'test'))")
    new_sample = Sample(os.path.join(test_pth, 'test'))
    new_sample.show_tree()

    print()
    print("-" * 80)
    print("--- new_sample.load(os.path.join(test_pth, 'test'))")
    new_sample = Sample()
    new_sample.load(os.path.join(test_pth, 'test'))
    new_sample.show_tree()

    print()
    print("-" * 80)
    new_sample_2 = Sample.load_from_dir(os.path.join(test_pth, 'test'))
    new_sample_2.show_tree()


# %% Main Script
if __name__ == '__main__':
    sample_examples()

    print()
    print("-" * 80)
    print("#==============#")
    print("#===# DONE #===#")
    print("#==============#")
