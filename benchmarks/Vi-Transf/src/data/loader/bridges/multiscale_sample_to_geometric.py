from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
from .utils import faces_to_edges
import torch
from sklearn.neighbors import KDTree


def get_distance_to_ids(vertices, boundary_ids):
    boundary_vertices = vertices[boundary_ids, :]
    search_index = KDTree(boundary_vertices)
    sdf, projection_id = search_index.query(vertices, return_distance=True)

    projection_vertices = boundary_vertices[projection_id.ravel()]
    projection_vectors = projection_vertices - vertices
    projection_vectors_norm = np.linalg.norm(projection_vectors, axis=1)
    projection_vectors_norm[projection_vectors_norm == 0] = 1
    projection_vectors = projection_vectors / projection_vectors_norm[:, None]

    return sdf, projection_vectors


def multiscale_sample_to_geometric(
    sample: Sample, sample_id: int, problem_definition: ProblemDefinition
) -> Data:
    """
    Converts a Plaid sample to PytorchGeometric Data object

    Args:
        sample (plaid.containers.sample.Sample): data sample

    Returns:
        Data: the converted data sample
    """

    vertices = sample.get_vertices()
    edge_index = []
    faces = sample.get_elements()["TRI_3"]
    edge_index = faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=True)

    v1 = vertices[edge_index[:, 0]]
    v2 = vertices[edge_index[:, 1]]
    edge_weight = np.linalg.norm(v2 - v1, axis=1)

    # data names
    input_fields_names = []
    output_fields_names = ["u1", "u2", "P11", "P12", "P22", "P21", "psi"]
    output_scalars_names = ["effective_energy"]
    input_scalars_names = ["C11", "C12", "C22"]

    input_scalars = []
    output_scalars = []
    for name in input_scalars_names:
        input_scalars.append(sample.get_scalar(name))
    for name in output_scalars_names:
        output_scalars.append(sample.get_scalar(name))

    input_fields = vertices
    input_fields_names = ["x", "y"]

    output_fields = []
    for field_name in output_fields_names:
        output_fields.append(sample.get_field(field_name))
    output_fields = np.vstack(output_fields).T

    # Extracting special nodal tags
    nodal_tags = {}
    for k, v in sample.get_nodal_tags().items():
        nodal_tags[k + "_id"] = torch.tensor(v, dtype=torch.long)

    holes_ids = sample.get_nodal_tags()["Holes"]
    sdf, projection_vectors = get_distance_to_ids(vertices, holes_ids)
    input_fields = np.concatenate((vertices, sdf, projection_vectors), axis=1)
    input_fields_names = ["x", "y", "sdf", "dist_vect_x", "dist_vect_y"]

    # torch tensor conversion
    input_scalars = torch.tensor(input_scalars, dtype=torch.float32).reshape(1, -1)
    input_fields = torch.tensor(input_fields, dtype=torch.float32)

    vertices = torch.tensor(vertices, dtype=torch.float32)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    faces = torch.tensor(faces, dtype=torch.long)

    if None not in output_scalars and None not in output_fields:
        output_scalars = torch.tensor(output_scalars, dtype=torch.float32).reshape(
            1, -1
        )
        output_fields = torch.tensor(output_fields, dtype=torch.float32)

        data = Data(
            pos=vertices,
            input_scalars=input_scalars,
            x=input_fields,
            output_scalars=output_scalars,
            output_fields=output_fields,
            edge_index=edge_index.T,
            edge_weight=edge_weight,
            faces=faces,
            sample_id=sample_id,
            input_fields_names=input_fields_names,
            output_fields_names=output_fields_names,
            input_scalars_names=input_scalars_names,
            output_scalars_names=output_scalars_names,
            **nodal_tags,
        )

        return data

    data = Data(
        pos=vertices,
        input_scalars=input_scalars,
        x=input_fields,
        edge_index=edge_index.T,
        edge_weight=edge_weight,
        faces=faces,
        sample_id=sample_id,
        input_fields_names=input_fields_names,
        output_fields_names=output_fields_names,
        input_scalars_names=input_scalars_names,
        output_scalars_names=output_scalars_names,
        **nodal_tags,
    )

    return data
