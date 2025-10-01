# Import required libraries
import gc
import logging
import pickle
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Import necessary libraries and functions
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid.containers.sample import Sample

# Input data
tracemalloc.start()

nr_simulations = 100
nr_freq = 2
nr_sources = 724
nr_dof = 70000

logging.basicConfig(level=logging.INFO)
for i in range(nr_simulations):
    print(f"=== Simulation {i}")
    # --------------#
    # ---# Init #---#
    # For each simulation, load it corresponding mesh (pickle data)
    with open("square_2D.pickle", "rb") as f:
        points = pickle.load(f)
        triangles = pickle.load(f)
    # Create a Mesh object
    Mesh = MCT.CreateMeshOfTriangles(points.copy(), triangles.copy())
    # Add fields to the mesh (observables per frequency)
    t0 = time.perf_counter()
    for freq in range(nr_freq):
        Mesh.nodeFields[f"var_{freq}"] = np.random.randn(nr_dof, nr_sources)
        Mesh.nodeFields[f"mean_{freq}"] = np.random.randn(nr_dof, nr_sources)
        Mesh.elemFields[f"field_{freq}"] = np.random.randn(nr_dof, nr_sources)
    print(f"Allocate fields took: {time.perf_counter() - t0:.3f} s")
    tree = MeshToCGNS(Mesh)

    # --------------#
    # ---# Work #---#
    # Initialize an empty Sample
    sample = Sample()
    # Add the previously created CGNS tree to the sample
    sample.add_tree(tree)

    test_pth = Path(
        f"/tmp/test_safe_to_delete_{np.random.randint(1e10, 1e12, dtype=np.uint64)}"
    )
    test_pth.mkdir(parents=True, exist_ok=True)

    sample_save_fname = test_pth / "test"
    print(f"saving path: {sample_save_fname}")

    t0 = time.perf_counter()
    sample.save(sample_save_fname)
    print(f"Save Sample: {time.perf_counter() - t0:.3f} s")

    for k in list(Mesh.nodeFields.keys()):
        del Mesh.nodeFields[k]
    for k in list(Mesh.elemFields.keys()):
        del Mesh.elemFields[k]
    for k in list(sample._meshes.keys()):
        del sample._meshes[k]
    del Mesh
    del sample

    gc.collect()

    # --------------------#
    # ---# Monitoring #---#
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
