from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
from .utils import faces_to_edges
import torch


def samc_eprouvettes_sample_to_geometric(sample: Sample, sample_id: int, problem_definition: ProblemDefinition) -> Data:
    vertices = sample.get_vertices()
    edge_index = []
    n_elem_types = len(sample.get_elements())
    coalesce = True
    assert len(sample.get_elements())==1, "More than one element type"
    if n_elem_types>1:
        coalesce = False
    for _, faces in sample.get_elements().items():
        edge_index.append(faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=coalesce))
    edge_index = np.concatenate(edge_index, axis=0)
    if not coalesce:
        edge_index = coalesce(edge_index)

    v1 = vertices[edge_index[:, 0]]
    v2 = vertices[edge_index[:, 1]]
    edge_weight = np.linalg.norm(v2 - v1, axis=1)

    input_fields_names   = ["x", "y", "z", "NormalsX", "NormalsY", "NormalsZ"] # problem_definition.get_input_fields_names()
    output_fields_names  = ["UX", "UY", "UZ"] # problem_definition.get_output_fields_names()

    normals = np.stack([sample.get_field(fname) for fname in ["NormalsX", "NormalsY", "NormalsZ"]], axis=0).T
    u = np.stack([sample.get_field(fname) for fname in ["UX", "UY", "UZ"]], axis=0).T
    input_fields = np.concatenate((vertices, normals), axis=1)
    output_fields = normals

    vertices        = torch.tensor(vertices, dtype=torch.float32)
    edge_weight     = torch.tensor(edge_weight, dtype=torch.float32)
    edge_index      = torch.tensor(edge_index, dtype=torch.long)
    faces           = torch.tensor(faces, dtype=torch.long)

    input_fields    = torch.tensor(input_fields, dtype=torch.float32)

    if None not in output_fields:
        output_fields   = torch.tensor(output_fields, dtype=torch.float32)

        data = Data(
            pos = vertices,
            x = input_fields,
            output_fields = output_fields,
            edge_index = edge_index.T,
            edge_weight = edge_weight,
            faces = faces,
            sample_id = sample_id,
            input_fields_names=input_fields_names,
            output_fields_names=output_fields_names,
            input_scalars_names=[],
            output_scalars_names=[]
        )
        
        return data

    data = Data(
        pos = vertices,
        x = input_fields,
        edge_index = edge_index.T,
        edge_weight = edge_weight,
        faces = faces,
        sample_id = sample_id,
        input_fields_names=input_fields_names,
        output_fields_names=output_fields_names,
        input_scalars_names=[],
        output_scalars_names=[]
    )

    return data
