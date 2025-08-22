from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
from .utils import faces_to_edges
import torch

from .multiscale_sample_to_geometric import get_distance_to_ids


def profile_sample_to_geometric(
    sample: Sample, sample_id: int, problem_definition: ProblemDefinition
) -> Data:
    vertices = sample.get_vertices()
    edge_index = []
    faces = sample.get_elements()["TRI_3"]
    edge_index = faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=True)

    v1 = vertices[edge_index[:, 0]]
    v2 = vertices[edge_index[:, 1]]
    edge_weight = np.linalg.norm(v2 - v1, axis=1)

    # data names
    output_fields_names = ["Mach", "Pressure", "Velocity-x", "Velocity-y"]

    input_fields = vertices

    output_fields = []
    for field_name in output_fields_names:
        output_fields.append(sample.get_field(field_name))
    output_fields = np.vstack(output_fields).T

    # Extracting special nodal tags
    nodal_tags = {}
    for k, v in sample.get_nodal_tags().items():
        nodal_tags[k + "_id"] = torch.tensor(v, dtype=torch.long)

    airfoil_ids = sample.get_nodal_tags()["Airfoil"]
    sdf, projection_vectors = get_distance_to_ids(vertices, airfoil_ids)
    input_fields = np.concatenate((vertices, sdf, projection_vectors), axis=1)
    input_fields_names = ["x", "y", "sdf", "dist_vect_x", "dist_vect_y"]

    # torch tensor conversion
    input_fields = torch.from_numpy(input_fields).to(torch.float32)

    vertices = torch.tensor(vertices, dtype=torch.float32)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    faces = torch.tensor(faces, dtype=torch.long)

    if None not in output_fields:
        output_fields = torch.tensor(output_fields, dtype=torch.float32)

        data = Data(
            pos=vertices,
            x=input_fields,
            output_fields=output_fields,
            edge_index=edge_index.T,
            edge_weight=edge_weight,
            faces=faces,
            sample_id=sample_id,
            input_fields_names=input_fields_names,
            output_fields_names=output_fields_names,
            input_scalars_names=[],
            output_scalars_names=[],
            **nodal_tags,
        )

        return data

    data = Data(
        pos=vertices,
        x=input_fields,
        edge_index=edge_index.T,
        edge_weight=edge_weight,
        faces=faces,
        sample_id=sample_id,
        input_fields_names=input_fields_names,
        output_fields_names=output_fields_names,
        input_scalars_names=[],
        output_scalars_names=[],
        **nodal_tags,
    )

    return data
