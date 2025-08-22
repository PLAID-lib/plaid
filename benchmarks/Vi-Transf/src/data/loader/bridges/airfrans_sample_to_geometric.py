from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import torch
from .utils import faces_to_edges
import numpy as np


def airfrans_sample_to_geometric(
    sample: Sample, sample_id: int, problem_definition: ProblemDefinition = None
) -> Data:
    """
    Converts a Plaid sample to PytorchGeometric Data object

    Args:
        sample (plaid.containers.sample.Sample): data sample

    Returns:
        Data: the converted data sample
    """

    vertices = sample.get_vertices()
    faces = sample.get_elements()["QUAD_4"]
    edge_index = faces_to_edges(faces, num_nodes=vertices.shape[0])

    airfoil_ids = sample.get_nodal_tags()["Airfoil"]

    v1 = vertices[edge_index[:, 0]]
    v2 = vertices[edge_index[:, 1]]
    edge_weight = np.linalg.norm(v2 - v1, axis=1)

    # loading scalars
    aoa = sample.get_scalar("angle_of_attack")
    inlet_velocity = sample.get_scalar("inlet_velocity")
    u_inlet = [np.cos(aoa) * inlet_velocity, np.sin(aoa) * inlet_velocity]
    cl = sample.get_scalar("C_L")
    cd = sample.get_scalar("C_D")
    output_scalars = np.array([cl, cd])

    # loading fields
    nut = sample.get_field("nut")
    ux = sample.get_field("Ux")
    uy = sample.get_field("Uy")
    p = sample.get_field("p")
    implicit_distance = sample.get_field("implicit_distance")

    # inlet velocities
    # u_inlet_field = np.array([np.cos(aoa), np.sin(aoa)]) * np.ones((nut.shape[0], 2))

    # TODO: Normals
    # normals = np.zeros_like(u_inlet)

    # converting to torch tensor
    u_inlet = torch.tensor(u_inlet, dtype=torch.float32)
    output_scalars = torch.tensor(output_scalars, dtype=torch.float32)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    nut = torch.tensor(nut, dtype=torch.float32)
    ux = torch.tensor(ux, dtype=torch.float32)
    uy = torch.tensor(uy, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    implicit_distance = torch.tensor(implicit_distance, dtype=torch.float32)

    # u_inlet             = torch.tensor(u_inlet)
    # normals             = torch.tensor(normals)

    # input / output features
    input_scalars = u_inlet.reshape(1, -1)

    output_scalars = output_scalars.reshape(1, -1)

    input_fields = torch.concatenate(
        [
            vertices,
            implicit_distance.reshape(-1, 1),
        ],
        dim=1,
    )

    output_fields = torch.concatenate(
        [nut.reshape(-1, 1), p.reshape(-1, 1), ux.reshape(-1, 1), uy.reshape(-1, 1)],
        dim=1,
    )

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
        airfoil_ids=airfoil_ids,
        input_fields_names=["x", "y", "implicit_distance"],
        output_fields_names=["nut", "p", "Ux", "Uy"],
        input_scalars_names=["ux_inlet", "uy_inlet"],
        output_scalars_names=["C_L", "C_D"],
    )

    return data
