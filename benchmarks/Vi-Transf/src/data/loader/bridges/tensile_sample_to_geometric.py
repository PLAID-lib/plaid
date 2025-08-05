from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
from .utils import faces_to_edges
import torch
from .multiscale_sample_to_geometric import get_distance_to_ids

def extract_border_edges(faces):
    edge_dict = {}

    for face in faces:
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if edge in edge_dict:
                edge_dict[edge] += 1
            else:
                edge_dict[edge] = 1

    border_edges = [edge for edge, count in edge_dict.items() if count == 1]
    return np.array(border_edges)


def get_border_ids(vertices, faces):
    bars = extract_border_edges(faces)
    border_ids = np.unique(np.ravel(bars))
    return border_ids



def tensile_sample_to_geometric(sample: Sample, sample_id: int, problem_definition: ProblemDefinition):
    vertices = sample.get_vertices()
    edge_index = []
    n_elem_types = len(sample.get_elements())
    coalesce = True
    if n_elem_types>1:
        coalesce = False
    assert len(sample.get_elements())==1, "More than one element type"
    for _, faces in sample.get_elements().items():
        edge_index.append(faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=coalesce))
    edge_index = np.concatenate(edge_index, axis=0)
    if not coalesce:
        edge_index = coalesce(edge_index)

    v1 = vertices[edge_index[:, 0]]
    v2 = vertices[edge_index[:, 1]]
    edge_weight = np.linalg.norm(v2 - v1, axis=1)

    # loading scalars
    input_scalars_names     = ['P', 'p1', 'p2', 'p3', 'p4', 'p5']
    output_scalars_names    = ['max_von_mises', 'max_U2_top', 'max_sig22_top']
    input_fields_names      = []
    output_fields_names     = ['U1', 'U2', 'sig11', 'sig22', 'sig12']

    input_scalars   = []
    output_scalars  = []
    for name in input_scalars_names:
        input_scalars.append(sample.get_scalar(name))
    for name in output_scalars_names:
        output_scalars.append(sample.get_scalar(name))

    # sdf and one hot encoding
    border_ids = get_border_ids(vertices, faces)
    sdf, projection_vectors = get_distance_to_ids(vertices, border_ids)
    # labels = np.array(list(map(int, is_border))).reshape(-1, 1)
    # labels[labels == 1] = 6
    # labels = torch.tensor(labels)
    # one_hot = one_hot(labels[:, 0].long(), num_classes=9)

    

    if len(input_fields_names) > 0:
        if input_fields_names[0]=="cell_ids":  input_fields_names.pop(0)

    input_fields = np.concatenate((vertices, sdf, projection_vectors), axis=1)
    input_fields_names = ["x", "y", "sdf", "dist_vect_x", "dist_vect_y"]

    output_fields   = []
    for field_name in output_fields_names:
        output_fields.append(sample.get_field(field_name))
    output_fields = np.vstack(output_fields).T    

    # torch tensor conversion
    input_scalars   = torch.tensor(input_scalars, dtype=torch.float32).reshape(1, -1)
    input_fields    = torch.tensor(input_fields, dtype=torch.float32)

    vertices        = torch.tensor(vertices, dtype=torch.float32)
    edge_weight     = torch.tensor(edge_weight, dtype=torch.float32)
    edge_index      = torch.tensor(edge_index, dtype=torch.long)
    faces           = torch.tensor(faces, dtype=torch.long)
    
    # Extracting special nodal tags
    nodal_tags = {}
    for k, v in sample.get_nodal_tags().items():
        nodal_tags[k + "_id"] = torch.tensor(v, dtype=torch.long)
    nodal_tags["Border_id"] = torch.tensor(border_ids, dtype=torch.long)

    if None not in output_scalars and None not in output_fields:
        output_scalars  = torch.tensor(output_scalars, dtype=torch.float32).reshape(1, -1)
        output_fields   = torch.tensor(output_fields, dtype=torch.float32)

        data = Data(
            pos = vertices,
            input_scalars = input_scalars,
            x = input_fields,
            output_scalars = output_scalars,
            output_fields = output_fields,
            edge_index = edge_index.T,
            edge_weight = edge_weight,
            faces = faces,
            sample_id = sample_id,
            input_fields_names=input_fields_names,
            output_fields_names=output_fields_names,
            input_scalars_names=input_scalars_names,
            output_scalars_names=output_scalars_names,
            **nodal_tags
        )
        
        return data

    data = Data(
        pos = vertices,
        input_scalars = input_scalars,
        x = input_fields,
        edge_index = edge_index.T,
        edge_weight = edge_weight,
        faces = faces,
        sample_id = sample_id,
        input_fields_names=input_fields_names,
        output_fields_names=output_fields_names,
        input_scalars_names=input_scalars_names,
        output_scalars_names=output_scalars_names,
        **nodal_tags
    )
