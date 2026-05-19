# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example of converting user data into PLAID
#
# This code provides an example for converting user data into the PLAID (Physics Informed AI Datamodel) format.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid import Sample
from plaid.storage import save_to_disk, push_to_hub

# %% [markdown]
# ## Construction stages

# %%
from IPython.display import Image
filename = list(x for x in [Path(locals().get("__file__","..")).parent.parent / "docs" / "source" / "images" / "to_plaid.png",
                        Path.cwd().parent / "docs" / "source" / "images" / "to_plaid.png"  ,
                        Path("..") / "images" / "to_plaid.png"
                      ] if x.exists())[0]
Image(filename=filename)

# %% [markdown]
# ## Define a 3D Mesh
#
# Define nodes and triangles to create a 3D mesh.

# %%
nodes_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.5, 1.0],
    ]
)

triangles = np.array(
    [
        [0, 1, 2],
        [0, 1, 4],
        [0, 2, 3],
        [0, 3, 4],
        [1, 2, 4],
        [2, 4, 3],
    ]
)

print(f"nb nodes: {len(nodes_3D)}")
print(f"nb triangles: {len(triangles)}")

# %% [markdown]
# ### Visualize the Mesh
#
# Create a 3D plot to visualize the mesh.

# %%
def in_notebook():
    try:
        from IPython import get_ipython
        return 'IPKernelApp' in get_ipython().config
    except Exception:
        return False

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(nodes_3D[:, 0], nodes_3D[:, 1], nodes_3D[:, 2], c="b", marker="o")

for triangle in triangles:
    triangle_nodes = nodes_3D[triangle]
    triangle_nodes = np.concatenate((triangle_nodes, [triangle_nodes[0]]))
    ax.plot(triangle_nodes[:, 0], triangle_nodes[:, 1], triangle_nodes[:, 2], c="g")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Show the plot
if in_notebook():
    plt.show()

# %% [markdown]
# ## Create Meshes Dataset from external data
#
# Generates a dataset (python list) of 3D meshes with random fields defined over nodes and elements


# %% [markdown]
# ## Sample contructor, a function to buid an instance for the sample i

def sample_constructor(i):
    #Creating mesh
    # this can be different for every sample
    """Create a Unstructured mesh using only points
    and the connectivity matrix for the triangles.
    Nodes id are given by there position in the list
    """
    mesh = MCT.CreateMeshOfTriangles(nodes_3D, triangles)


    """ Add field defined over the nodes (all the nodes).
        The keys are the names of the fields
        the values are the actual data of size (nb nodes, nb of components)"""
    mesh.nodeFields["node_field"] = np.random.randn(5)

    """ Add field defined over the elements (all the elements).
        The keys are the names of the fields
        the values are the actual data of size (nb elements, nb of components)"""
    mesh.elemFields["elem_field"] = np.random.randn(6)

    """ Convert the Muscat mesh to a cgns mesh """
    cgns_tree = MeshToCGNS(mesh)

    # operate directly on teh cgns mesh
    in_scalars_names = ["P", "p1", "p2", "p3", "p4", "p5"]
    out_scalars_names = ["max_von_mises", "max_q", "max_U2_top", "max_sig22_top"]
    out_fields_names = ["U1", "U2", "q", "sig11", "sig22", "sig12"]

    # Add CGNS Mesh to samples with specific time steps
    sample = Sample()

    sample.features.add_tree(cgns_tree)

    # Add random scalar values to the sample
    for sname in in_scalars_names:
        sample.add_global(sname, np.random.randn())

    for sname in out_scalars_names:
        sample.add_global(sname, np.random.randn())

    # Add random field values to the sample
    for j, sname in enumerate(out_fields_names):
        sample.add_field(sname, np.random.rand(len(nodes_3D)))

    return sample

# %% [markdown]
# ## Create PLAID Dataset on disk

# %%
infos: dict = {
    "legal": {"owner": "Bob", "license": "my_license"},
    "data_production": {"type": "simulation", "physics": "3D example"},
}

# %%
# save the dataset to disk using the cgns backend
import tempfile
temp_dir = tempfile.gettempdir() + "/my_new_cgns_dataset"

ids = {"train": list(range(5)),
       "test": list(range(10))}

save_to_disk(output_folder=temp_dir,
            sample_constructor=sample_constructor,
            ids=ids,
            backend="cgns",
            infos=infos,
            overwrite=True)


