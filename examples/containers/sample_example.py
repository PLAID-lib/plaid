# %% [markdown]
# # Sample Examples
#
# This Jupyter Notebook demonstrates various operations and methods involving a sample data structure using the PLAID library. It includes examples of:
#
# 1. Initializing an Empty Sample and Adding Data
# 2. Accessing and Modifying Sample Data
# 3. Set and Get default values
# 4. Saving and Loading Samples
#
# This notebook provides detailed examples of using the Sample class to manage and manipulate sample data structures.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
from pathlib import Path

# %%
# Import necessary libraries and functions
import CGNS.PAT.cgnskeywords as CGK
import numpy as np
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid.containers.sample import Sample
from plaid.utils import cgns_helper as CGH


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
points = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.5],
    ]
)

triangles = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [2, 4, 3],
    ]
)

Mesh = MCT.CreateMeshOfTriangles(points, triangles)
Mesh.nodeFields["test_node_field_1"] = np.random.randn(5)
Mesh.elemFields["test_elem_field_1"] = np.random.randn(3)
tree = MeshToCGNS(Mesh)

# Display CGNS Tree
CGH.show_cgns_tree(tree)

# %%
# Print CGNS tree summary for a concise overview
CGH.summarize_cgns_tree(tree, verbose=True)

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
sample.add_scalar("rotation", np.random.randn())

show_sample(sample)

# %%
# Add a more scalars to this Sample
sample.add_scalar("speed", np.random.randn())
sample.add_scalar("other", np.random.randn())

show_sample(sample)

# %% [markdown]
# ### Add time series to a Sample

# %%
# Add a time series named 'stuff'
sample.add_time_series("stuff", np.arange(10), np.random.randn(10))

# Add a time series named 'bluff'
sample.add_time_series("bluff", np.arange(2, 6), np.random.randn(4))

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
meshes_dict = {0.0: tree, 0.5: tree, 1.0: tree}

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
sample.init_base(2, 3, "SurfaceMesh", time=0.0)

show_sample(sample)

# %% [markdown]
# ### Initialize CGNS tree zone

# %%
# Init CGNS tree zone to a base at time 0.
shape = np.array((len(points), len(triangles), 0))
sample.init_zone(shape, zone_name="TestZoneName", base_name="SurfaceMesh", time=0.0)

show_sample(sample)

# %% [markdown]
# ### Set the coordinates of nodes for a specified base and zone

# %%
points = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.5],
    ]
)

# Set the coordinates of nodes for a specified base and zone at a given time.
# set_points == set_nodes == set_vertices
sample.set_nodes(points, base_name="SurfaceMesh", zone_name="TestZoneName", time=0.0)

show_sample(sample)

# %% [markdown]
# ### Add a field to a specified zone in the grid

# %%
# Add a field to a specified zone
sample.add_field(
    "Pressure",
    np.random.randn(len(points)),
    base_name="SurfaceMesh",
    zone_name="TestZoneName",
    time=0.0,
)

show_sample(sample)

# %%
# Add another field
sample.add_field(
    "Temperature",
    np.random.randn(len(points)),
    base_name="SurfaceMesh",
    zone_name="TestZoneName",
    time=0.0,
)

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
sample.add_time_series("stuff", np.arange(10), np.random.randn(10))

print(f"{sample.get_time_series_names() = }")
print(f"{sample.get_time_series('S') = }")
print(f"{sample.get_time_series('stuff') = }")

# %% [markdown]
# ### Access to points coordinates

# %%
# It will look for a default base if no base and zone are given
print(f"{sample.get_nodes() = }")
print(f"{sample.get_points() = }")  # same as get_nodes
print(f"{sample.get_vertices() = }")  # same as get_nodes

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
sample.add_tree(tree, 1.0)

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
            f"    - {z_name} -> type: {sample.get_zone_type(z_name, b_name)} | full: {f_z_name}"
        )

# %% [markdown]
# ## Section 3: Set and Get default values
#
# This section demonstrates how to use default CGNS values in a Sample.

# %% [markdown]
# ### Set and use default time in a Sample

# %%
# Without a provided default time, it searches the first time available in all mesh times
print(f"{sample.get_all_mesh_times() = }")
print(f"{sample.get_time_assignment() = }", end="\n\n")

# Set default time
sample.set_default_time(1.0)
# Now that default time has been assigned, there's no need to specify it in function calls.
print(f"{sample.get_time_assignment() = }", end="\n\n")

# Print the tree at time 1.0
sample.show_tree()  # == sample.show_tree(1.0)

# %%
# If time is specified as an argument in a function, it takes precedence over the default time.
sample.show_tree(0.0)  # Print the tree at time 0.0 even if default time is 1.0

# %% [markdown]
# ### Set and use default base and time in a Sample

# %%
# Reset default time
sample._defaults["active_time"] = None

# Without a provided default time, it searches the first time available in all mesh times
print(f"{sample.get_time_assignment() = }", end="\n\n")

# Create new bases
sample.init_base(1, 1, "new_base", 0.0)
print(f"{sample.get_topological_dim('new_base', 0.0) = }")
print(f"{sample.get_physical_dim('new_base', 0.0) = }")

# %%
# Attempting to get a base when the default base is not set, and there are multiple bases available.
print(f"{sample.get_base_names() = }", end="\n\n")
try:
    sample.get_base_assignment()
except KeyError as e:
    print(str(e))

# %%
# Set default base and time
sample.set_default_base("SurfaceMesh", 0.0)

# Now that default base and time have been assigned, it is no longer necessary to specify them in function calls.
print(f"{sample.get_time_assignment() = }")
print(f"{sample.get_base_assignment() = }", end="\n\n")

# Print the topological and physical dim for the default base == 'SurfaceMesh'
print(f"{sample.get_topological_dim() = }")
print(f"{sample.get_physical_dim() = }")

# %%
# If base is specified as an argument in a function, it takes precedence over the default base.
print(
    f"{sample.get_physical_dim('new_base') = }"
)  # Print the 'new_base' physical dim instead of the default base physical dim

# %% [markdown]
# ### Set and use default base, zone and time in a Sample

# %%
# Reset default base and time
sample._defaults["active_time"] = None
sample._defaults["active_base"] = None

# Without a provided default time, it searches the first time available in all mesh times
print(f"{sample.get_time_assignment() = }", end="\n\n")

# Create a new zone in 'SurfaceMesh' base
sample.init_zone(
    zone_shape=np.array([5, 3, 0]),
    zone_type=CGK.Structured_s,
    zone_name="new_zone",
    base_name="SurfaceMesh",
)
print(f"{sample.get_zone_type('TestZoneName', 'SurfaceMesh') = }")
print(f"{sample.get_zone_type('new_zone', 'SurfaceMesh') = }")

# %%
# Set default base
sample.set_default_base("SurfaceMesh")

# Attempting to get a zone when the default zone is not set, and there are multiple zones available in the default base.
print(f"{sample.get_zone_names() = }", end="\n\n")
try:
    sample.get_zone_assignment()
except KeyError as e:
    print(str(e))

# %%
# Reset default base and time
sample._defaults["active_time"] = None
sample._defaults["active_base"] = None

# Set default base, zone and time
sample.set_default_zone_base("TestZoneName", "SurfaceMesh", 0.0)

# Now that default base, zone and time have been assigned, it is no longer necessary to specify them in function calls.
print(f"{sample.get_time_assignment() = }")
print(f"{sample.get_base_assignment() = }")
print(f"{sample.get_zone_assignment() = }", end="\n\n")

# Print the type of the default zone (from the default base)
print(f"{sample.get_zone_type() = }")

# Print the default zone content (from the default base)
print(f"{sample.get_zone() = }")

# %%
# If zone is specified as an argument in a function, it takes precedence over the default zone.
print(
    f"{sample.get_zone_type('new_zone') = }"
)  # Print the 'new_zone' type instead of the default zone type

# %% [markdown]
# ### More information on how default values work

# %% [markdown]
# ![Alt text](../../docs/source/images/default_value_selection.png "default values flowchart")

# %% [markdown]
# ## Section 4: Saving and Loading Sample
#
# This section demonstrates how to save and load a Sample from a directory.

# %% [markdown]
# ### Save Sample to as a file tree

# %%
test_pth = Path(f"/tmp/test_safe_to_delete_{np.random.randint(low=1, high=2_000_000_000)}")
test_pth.mkdir(parents=True, exist_ok=True)

sample_save_fname = test_pth / "test"
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
new_sample_2 = Sample.load_from_dir(test_pth / "test")

show_sample(new_sample)

# %% [markdown]
# ### Load the Sample from a directory via a Sample instance

# %%
new_sample = Sample()
new_sample.load(sample_save_fname)

show_sample(new_sample)

new_sample.add_scalar("a", 2.1)
serialized_sample = new_sample.model_dump()

unserialized_sample = Sample.model_validate(serialized_sample)
print()
show_sample(unserialized_sample)
