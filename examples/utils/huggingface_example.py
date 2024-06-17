import numpy as np
from tqdm import tqdm
import pickle

from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from plaid.utils.split import split_dataset
from plaid.utils import huggingface

##############################################
# CONSTRUCT A PLAID DATASET
##############################################

nodes_3D = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.5, 1.0],
    ])

triangles = np.array([
        [0, 1, 2],
        [0, 1, 4],
        [0, 2, 3],
        [0, 3, 4],
        [1, 2, 4],
        [2, 4, 3],
    ])

for triangle in triangles:
    triangle_nodes = nodes_3D[triangle]
    triangle_nodes = np.concatenate((triangle_nodes, [triangle_nodes[0]]))


nb_meshes = 20
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
    Mesh.nodeFields['node_field'] = np.random.randn(5)

    """ Add field defined over the elements (all the elements).
        The keys are the names of the fields
        the values are the actual data of size (nb elements, nb of components)"""
    Mesh.elemFields['elem_field'] = np.random.randn(6)

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
out_scalars_names = ["max_von_mises","max_q","max_U2_top","max_sig22_top"]
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

dataset = Dataset()

infos: dict = {
        "legal": {
            "owner": "Bob",
            "license": "my_license"},
        "data_production": {
            "type": "simulation",
            "physics": "3D example"}
    }

# Set information for the PLAID dataset
dataset.set_infos(infos)

# %%
# Add PLAID samples to the dataset
dataset.add_samples(samples)
print(f"{dataset = }")

problem = ProblemDefinition()
problem.add_input_scalars_names(in_scalars_names)
problem.add_output_scalars_names(out_scalars_names)
problem.add_output_fields_names(out_fields_names)
problem.add_input_meshes_names(['/Base_2_2/Zone'])

problem.set_task('regression')

# Set startegy options for the split
options = {
    'shuffle': False,
    'split_sizes': {
        'train': 12,
        'test': 8,
    },
}

split = split_dataset(dataset, options)
print(f"{split = }")


problem.set_split(split)
print(f"{problem = }")


##############################################
# EXAMPLE OF HUGGINGFACE utils
##############################################

hf_dataset = huggingface.convert_dataset_to_huggingface(dataset, problem)
print()
print(f"{hf_dataset = }")
print(f"{hf_dataset.description = }")


def generator():
    for id in range(len(dataset)):
        yield {
            "sample" : pickle.dumps(dataset[id]),
        }

hf_dataset_gen = huggingface.generate_huggingface_dataset(generator, infos, problem)
print()
print(f"{hf_dataset_gen = }")
print(f"{hf_dataset_gen.description = }")


dataset_2, problem_2 = huggingface.convert_huggingface_to_dataset(hf_dataset)
print()
print(f"{dataset_2 = }")
print(f"{problem_2 = }")

card_text = huggingface.create_string_for_huggingface_dataset_card(
    hf_dataset.description,
    download_size_bytes = 395462557,
    dataset_size_bytes = 864904041,
    nb_samples = 702,    
    owner = "Safran",
    license = "cc-by-sa-4.0",
    zenodo_url = "https://zenodo.org/records/10124594",
    arxiv_paper_url = "https://arxiv.org/pdf/2305.12871",
    pretty_name = "2D quasistatic non-linear structural mechanics solutions",
    size_categories = "n<1K",
    task_categories = "other",
    tags = ["physics learning", "geometry learning"],
    dataset_long_description = """
This dataset contains 2D quasistatic non-linear structural mechanics solutions, under geometrical variations.

A description is provided in the [MMGP paper ](https://arxiv.org/pdf/2305.12871) Sections 4.1 and A.2.

The variablity in the samples are 6 input scalars and the geometry (mesh). Outputs of interest are 4 scalars and 6 fields.

Seven nested training sets of sizes 8 to 500 are provided, with complete input-output data. A testing set of size 200, as well as two out-of-distribution samples, are provided, for which outputs are not provided.

Dataset created using the [PLAID](https://plaid-lib.readthedocs.io/) library and datamodel.
    """,
    url_illustration = "https://i.ibb.co/Js062hF/preview.png"
)

print("========================================")
print("Huggingface dataset card")
print(card_text)
print("========================================")
