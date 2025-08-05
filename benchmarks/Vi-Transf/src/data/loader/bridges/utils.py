import numpy as np
import torch
from torch_geometric.utils._coalesce import coalesce as geometric_coalesce


def my_coalesce(edges: torch.Tensor | np.ndarray, num_nodes: int, reduce="add"):
    if isinstance(edges, np.ndarray):
        edges = torch.tensor(edges).T
        return geometric_coalesce(edges, num_nodes=num_nodes, reduce=reduce).T.numpy()
    edges = geometric_coalesce(edges.T, num_nodes=num_nodes, reduce=reduce).T
    return edges

def faces_to_edges(faces: np.ndarray, num_nodes: int, coalesce: bool=True):
    """Creates a list of edges from a Faces array

    Args:
        faces (np.ndarray): Array of faces shape (n_faces, face_dim)

    Returns:
        np.ndarray: the edge list of shape (n, 2)
    """

    assert len(faces.shape)==2, "Wrong shape for the faces, should be a 2D array"

    # Generate edges (without duplicates in one pass)
    rolled = np.roll(faces, -1, axis=1)
    edges = np.vstack((faces.ravel(), rolled.ravel())).T
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)

    # Ensure unique edges by sorting each edge and using np.unique
    if coalesce:
        edges = my_coalesce(edges, num_nodes)

    return edges