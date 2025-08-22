from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
from src.data.loader.bridges.utils import faces_to_edges
from src.data.loader.bridges.multiscale_sample_to_geometric import get_distance_to_ids
import torch

from Muscat.Containers import MeshModificationTools as MMT
from Muscat.Bridges.CGNSBridge import CGNSToMesh
from Muscat.Containers.Filters import FilterObjects as FO

import warnings

# warnings.filterwarnings("ignore", module="Muscat")
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


def elasto_plasto_dynamics_sample_to_geometric(
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
    coalesce = True
    for _, faces in sample.get_elements().items():
        edge_index.append(
            faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=coalesce)
        )
    edge_index = np.concatenate(edge_index, axis=0)

    mesh = CGNSToMesh(sample.get_mesh())
    MMT.ComputeSkin(mesh, md=2, inPlace=True, skinTagName="Skin")
    nfSkin = FO.NodeFilter(eTag="Skin")
    nodeIndexSkin = nfSkin.GetNodesIndices(mesh)
    mesh.GetNodalTag("Skin").AddToTag(nodeIndexSkin)
    border_ids = mesh.GetNodalTag("Skin").GetIds()

    sdf, projection_vectors = get_distance_to_ids(vertices, border_ids)
    sdf = sdf.reshape(-1, 1)
    input_fields = np.concatenate((vertices, sdf, projection_vectors), axis=1)
    input_fields_names = ["x", "y", "sdf", "dist_vect_x", "dist_vect_y", "U_x", "U_y"]
    output_fields_names = ["U_x", "U_y"]
    input_scalars_names = ["time"]
    output_scalars_names = []

    timestep_list = sample.get_all_mesh_times()

    output_fields = np.vstack(
        (
            sample.get_field("U_x", time=timestep_list[0]),
            sample.get_field("U_y", time=timestep_list[0]),
        )
    ).T
    data_list = []

    vertices_torch = torch.from_numpy(vertices).to(
        torch.float32
    )  # torch.tensor(vertices, dtype=torch.float32)
    edge_index = torch.from_numpy(edge_index).to(
        torch.long
    )  # torch.tensor(edge_index, dtype=torch.long)

    for t0, t1 in zip(timestep_list[:-1], timestep_list[1:]):
        output_fields_t1 = output_fields
        if output_fields_t1[0, 0] is not None:
            output_fields = np.vstack(
                (sample.get_field("U_x", time=t1), sample.get_field("U_y", time=t1))
            ).T

            input_scalars = np.array([t0])
            input_fields = np.column_stack(
                (vertices, sdf, projection_vectors, output_fields_t1)
            )

            # torch tensor conversion
            input_scalars = (
                torch.from_numpy(input_scalars).to(torch.float32).reshape(1, -1)
            )  # torch.tensor(input_scalars, dtype=torch.float32).reshape(1, -1)
            input_fields = torch.from_numpy(input_fields).to(
                torch.float32
            )  # torch.tensor(input_fields, dtype=torch.float32)

            # Extracting special nodal tags
            nodal_tags = {}
            for k, v in sample.get_nodal_tags().items():
                nodal_tags["border_id"] = torch.tensor(border_ids, dtype=torch.int)

            if None not in output_fields:
                output_fields = torch.from_numpy(output_fields).to(
                    torch.float32
                )  # torch.tensor(output_fields, dtype=torch.float32)

                data = Data(
                    pos=vertices_torch,
                    input_scalars=input_scalars,
                    x=input_fields,
                    output_fields=output_fields,
                    edge_index=edge_index.T,
                    sample_id=sample_id,
                    input_fields_names=input_fields_names,
                    output_fields_names=output_fields_names,
                    input_scalars_names=input_scalars_names,
                    output_scalars_names=output_scalars_names,
                    time=t0,
                    timestep_list=timestep_list,
                    **nodal_tags,
                )
            else:
                data = Data(
                    pos=vertices_torch,
                    input_scalars=input_scalars,
                    x=input_fields,
                    edge_index=edge_index.T,
                    sample_id=sample_id,
                    input_fields_names=input_fields_names,
                    output_fields_names=output_fields_names,
                    input_scalars_names=input_scalars_names,
                    output_scalars_names=output_scalars_names,
                    time=t0,
                    timestep_list=timestep_list,
                    **nodal_tags,
                )
            data_list.append(data)

    return data_list
