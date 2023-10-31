# %% [markdown]
# # Sample Examples
#
# This Jupyter Notebook demonstrates various operations and methods involving a sample data structure using the PLAID library. It includes examples of:
#
# 1. Initializing an Empty Sample and Adding Data
# 2. Accessing and Modifying Sample Data
# 3. Saving and Loading Samples
#
# This notebook provides detailed examples of using the Sample class to manage and manipulate sample data structures.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
import numpy as np
import os

# %%
# Import necessary libraries and functions
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import UnstructuredMeshCreationTools as UMCT

from plaid.containers.sample import Sample, show_cgns_tree

# %%
# Print Sample util
def show_sample(sample: Sample):
    print(f"{sample = }")
    sample.show_tree()
    print(f"{sample.get_scalar_names() = }")
    print(f"{sample.get_field_names() = }")

# %% [markdown]
# ## Section 1: Initializing an Empty Sample and Adding Data
#
# This section demonstrates how to initialize an empty Sample and add scalars, time series data, and meshes / CGNS trees.

# %% [markdown]
# ### Create and display CGNS tree from an unstructured mesh

# %%
# Input data
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

# Display CGNS Tree
show_cgns_tree(tree)

# %% [markdown]
# ### Initialize a new empty Sample and print it

# %%
# Initialize an empty Sample
print("#---# Empty Sample")
sample = Sample()

print(sample, end="\n\n")
show_sample(sample)

# %% [markdown]
# ### Add a scalars to a Sample

# %%
# Add a rotation scalar to this Sample
sample.add_scalar('rotation', np.random.randn())

show_sample(sample)

# %%
# Add a more scalars to this Sample
sample.add_scalar('speed', np.random.randn())
sample.add_scalar('other', np.random.randn())

show_sample(sample)

# %% [markdown]
# ### Add time series to a Sample

# %%
# Add a time series named 'stuff'
sample.add_time_series('stuff', np.arange(10), np.random.randn(10))

# Add a time series named 'bluff'
sample.add_time_series('bluff', np.arange(2, 6), np.random.randn(4))

# As you can see it is not displayed when printing
show_sample(sample)

# %% [markdown]
# ### Add a CGNS Tree to a Sample and display it

# %%
# Add the previously created CGNS tree to the sample
sample.add_tree(tree)

# Display the Sample CGNS tree
sample.show_tree()

# %% [markdown]
# ### Set all meshes with their corresponding time step

# %%
# Init an empty Sample
new_sample_mult_mesh = Sample()

# All meshes with their corresponding time step
meshes_dict = {
    0. : tree,
    0.5 : tree,
    1. : tree
    }

# Set meshes in the Sample
new_sample_mult_mesh.set_meshes(meshes_dict)

print(f"{new_sample_mult_mesh.get_all_mesh_times() = }")
# new_sample_mult_mesh.show_tree(1.)

# %% [markdown]
# ## Section 2: Accessing and Modifying Sample Data
#
# This section demonstrates how to access and modify base, zone, node, scalar, field and time series data within the Sample.

# %% [markdown]
# ### Initialize CGNS tree base

# %%
# Initialize an new empty Sample
print("#---# Empty Sample")
sample = Sample()
print(sample, end="\n\n")

# Init CGNS tree base at time 0.
sample.init_base(2, 3, 'SurfaceMesh', time=0.)

show_sample(sample)

# %% [markdown]
# ### Initialize CGNS tree zone

# %%
# Init CGNS tree zone to a base at time 0.
shape = np.array((len(points), len(triangles), 0))
sample.init_zone(
    'TestZoneName',
    shape,
    base_name='SurfaceMesh',
    time=0.)

show_sample(sample)

# %% [markdown]
# ### Set the coordinates of nodes for a specified base and zone

# %%
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.5, 1.5],
])

# Set the coordinates of nodes for a specified base and zone at a given time.
# set_points == set_nodes == set_vertices
sample.set_nodes(
    points,
    base_name='SurfaceMesh',
    zone_name='TestZoneName',
    time=0.)

show_sample(sample)

# %% [markdown]
# ### Add a field to a specified zone in the grid

# %%
# Add a field to a specified zone
sample.add_field(
    'Pressure',
    np.random.randn(
        len(points)),
    base_name='SurfaceMesh',
    zone_name='TestZoneName',
    time=0.)

show_sample(sample)

# %%
# Add another field
sample.add_field(
    'Temperature',
    np.random.randn(
        len(points)),
    base_name='SurfaceMesh',
    zone_name='TestZoneName',
    time=0.)

show_sample(sample)

# %% [markdown]
# ### Access scalars data in Sample

# %%
# It will look for a default base if no base and zone are given
print(f"{sample.get_scalar_names() = }")
print(f"{sample.get_scalar('omega') = }")
print(f"{sample.get_scalar('rotation') = }")

# %% [markdown]
# ### Access fields data in Sample

# %%
# It will look for a default base if no base and zone are given
print(f"{sample.get_field_names() = }")
print(f"{sample.get_field('T') = }")
print(f"{sample.get_field('Temperature') = }")

# %% [markdown]
# ### Access time series data in Sample

# %%
# It will look for a default base if no base and zone are given
sample.add_time_series('stuff', np.arange(10), np.random.randn(10))

print(f"{sample.get_time_series_names() = }")
print(f"{sample.get_time_series('S') = }")
print(f"{sample.get_time_series('stuff') = }")

# %% [markdown]
# ### Access to points coordinates

# %%
# It will look for a default base if no base and zone are given
print(f"{sample.get_nodes() = }")
print(f"{sample.get_points() = }") # same as get_nodes
print(f"{sample.get_vertices() = }") # same as get_nodes

# %% [markdown]
# ### Retrieve element connectivity data

# %%
# Create an empty Sample
tmp_sample = Sample()

# Add the previously created CGNS tree in the Sample
tmp_sample.add_tree(tree)

print("element connectivity = \n", f"{tmp_sample.get_elements()}")

# %% [markdown]
# ### Access the available base of the CGNS tree

# %%
# Get base names
bases_names = sample.get_base_names()
# Get full base path
full_bases_names = sample.get_base_names(full_path=True)

print(f"{bases_names=}")
print(f"{full_bases_names=}")

# %%
# Get the first base name
base_name = sample.get_base_names()[0]
# Get base node
base_node_content = sample.get_base(base_name)

print(f"{base_node_content = }")

# %% [markdown]
# ### Check if a base exists in a Sample

# %%
# Get the first base name
base_name = sample.get_base_names()[0]

print(f"{sample.has_base(base_name) = }")
print(f"{sample.has_base('unknown_base_name') = }")

# %% [markdown]
# ### Access the available zone from a CGNS tree base

# %%
# Get the first base name
base_name = sample.get_base_names()[0]

# Get zones associated with the first base
zones_names = sample.get_zone_names(base_name)
# Get full path of zones associated with the first base
full_zones_names = sample.get_zone_names(base_name, full_path=True)

print(f" - Base : {base_name}")
print(f"    - Zone(s): {zones_names}")
print(f"    - Zone(s) full path: {full_zones_names}")

# %%
# Get the first zone name from a base name
zone_name = zones_names[0]
# Get base node
zone_node_content = sample.get_zone(zone_name, base_name)

print(f"{zone_node_content = }")

# %% [markdown]
# ### Get the zone type

# %%
# Get the first zone name from a base name
zone_name = zones_names[0]
z_type = sample.get_zone_type(zone_name, base_name)

print(f"zone type = {z_type}")

# %% [markdown]
# ### Check if a zone exists in a Sample

# %%
# Get the first zone name from a base name
zone_name = zones_names[0]

print(f"{sample.has_zone(zone_name, base_name) = }")
print(f"{sample.has_zone('unknown_zone_name', base_name) = }")

# %% [markdown]
# ### Get mesh from sample

# %%
sample_mesh = sample.get_mesh()
print(sample_mesh)

# %% [markdown]
# ### Get all mesh time available in Sample

# %%
# Before adding new tree
print(f"{sample.get_all_mesh_times() = }")

# Add one CGNS tree at time 1.
sample.add_tree(tree, 1.)

# After adding new tree
print(f"{sample.get_all_mesh_times() = }")

# %% [markdown]
# ### Creating a Sample Hierarchy with bases, zones, and associated data.

# %%
bases_names = sample.get_base_names()
full_bases_names = sample.get_base_names(full_path=True)
print(f"{bases_names = }")
print(f"{full_bases_names = }", end="\n\n")

for b_name in bases_names:
    zones_names = sample.get_zone_names(b_name)
    full_zones_names = sample.get_zone_names(b_name, full_path=True)
    print(f" - Base : {b_name}")
    for z_name, f_z_name in zip(zones_names, full_zones_names):
        print(
            f"    - {z_name} -> type: {sample.get_zone_type(z_name, b_name)} | full: {f_z_name}")

# %% [markdown]
# ## Section 3: Saving and Loading Sample
#
# This section demonstrates how to save and load a Sample from a directory.

# %% [markdown]
# ### Save Sample to as a file tree

# %%
test_pth = f'/tmp/test_safe_to_delete_{np.random.randint(1e10, 1e12)}'
os.makedirs(test_pth)

sample_save_fname = os.path.join(test_pth, 'test')
print(f"saving path: {sample_save_fname}")

sample.save(sample_save_fname)

# %% [markdown]
# ### Load a Sample from a directory via initialization

# %%
new_sample = Sample(sample_save_fname)

show_sample(new_sample)

# %% [markdown]
# ### Load a Sample from a directory via the Sample class

# %%
new_sample_2 = Sample.load_from_dir(os.path.join(test_pth, 'test'))

show_sample(new_sample)

# %% [markdown]
# ### Load the Sample from a directory via a Sample instance

# %%
new_sample = Sample()
new_sample.load(sample_save_fname)

show_sample(new_sample)


