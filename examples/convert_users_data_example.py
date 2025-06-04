# %% [markdown]
# # Example for converting user data into PLAID
#
# This code provides an example for converting user data into the PLAID (Physics Informed AI Datamodel) format.

# %%
import matplotlib.pyplot as plt
import numpy as np
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT
from tqdm import tqdm

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample

# %% [markdown]
# ## Construction stages
# ![Alt text](to_plaid.png)

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

# %% [markdown]
# ### Visualize the Mesh
#
# Create a 3D plot to visualize the mesh.

# %%
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
# plt.show()

# %% [markdown]
# ## Create Meshes Dataset from external data
#
# Generates a dataset (python list) of 3D meshes with random fields defined over nodes and elements

# %%

nb_meshes = 5000
meshes = []

print("Creating meshes dataset...")
for _ in tqdm(range(nb_meshes)):
    """Create a Unstructured mesh using only points
    and the connectivity matrix for the triangles.
    Nodes id are given by there position in the list
    """
    Mesh = MCT.CreateMeshOfTriangles(nodes_3D, triangles)

    """ Add field defined over the nodes (all the nodes).
        The keys are the names of the fields
        the values are the actual data of size (nb nodes, nb of components)"""
    Mesh.nodeFields["node_field"] = np.random.randn(5)

    """ Add field defined over the elements (all the elements).
        The keys are the names of the fields
        the values are the actual data of size (nb elements, nb of components)"""
    Mesh.elemFields["elem_field"] = np.random.randn(6)

    meshes.append(Mesh)

# %% [markdown]
# ## Convert to CGNS meshes

# %%
CGNS_meshes = []
for mesh in tqdm(meshes):
    # Converts a Mesh (muscat mesh following vtk conventions) to a CGNS Mesh
    CGNS_tree = MeshToCGNS(mesh)
    CGNS_meshes.append(CGNS_tree)

# %% [markdown]
# ## Create PLAID Samples from CGNS meshes

# %%
in_scalars_names = ["P", "p1", "p2", "p3", "p4", "p5"]
out_scalars_names = ["max_von_mises", "max_q", "max_U2_top", "max_sig22_top"]
out_fields_names = ["U1", "U2", "q", "sig11", "sig22", "sig12"]

samples = []
for cgns_tree in tqdm(CGNS_meshes):
    # Add CGNS Meshe to samples with specific time steps
    sample = Sample()

    sample.add_tree(cgns_tree)

    # Add random scalar values to the sample
    for sname in in_scalars_names:
        sample.add_scalar(sname, np.random.randn())

    for sname in out_scalars_names:
        sample.add_scalar(sname, np.random.randn())

    # Add random field values to the sample
    for j, sname in enumerate(out_fields_names):
        sample.add_field(sname, np.random.rand(1, len(nodes_3D)))

    samples.append(sample)

print(samples[0])

# %% [markdown]
# ## Create PLAID Dataset

# %%
infos: dict = {
    "legal": {"owner": "Bob", "license": "my_license"},
    "data_production": {"type": "simulation", "physics": "3D example"},
}


dataset = Dataset()

# Set information for the PLAID dataset
dataset.set_infos(infos)
dataset.print_infos()

# %%
# Add PLAID samples to the dataset
sample_ids = dataset.add_samples(samples)
print(sample_ids)
print(dataset)
