import h5py
import math
import torch
import numpy as np
from sklearn.neighbors import KDTree
from Muscat.Containers.MeshInspectionTools import ComputeMeshMinMaxLengthScale


def get_bandwidth(mesh) -> float:
    lengthscale = ComputeMeshMinMaxLengthScale(mesh)
    return lengthscale


def relative_rmse_field(
    y_true: list[torch.Tensor], y_pred: list[torch.Tensor], threshold: float = 0.0
) -> torch.Tensor:
    return torch.sqrt(
        torch.mean(
            torch.stack(
                [
                    torch.linalg.norm(y - y_hat, axis=0) ** 2
                    / (len(y) * step_function_field(y, threshold) ** 2)
                    for y, y_hat in zip(y_true, y_pred)
                ]
            ),
            dim=0,
        )
    )


def save_fields(filename: str, fields: list[torch.Tensor]) -> None:
    with h5py.File(filename, "w", libver="latest") as f:
        for idx, field in enumerate(fields):
            dset = f.create_dataset(str(idx), data=field.cpu().numpy())
    return None


def save_scalars(file_path, data_list):
    with h5py.File(file_path, "w") as f:
        for i, data_array in enumerate(data_list):
            f.create_dataset(f"array_{i}", data=data_array)


def load_fields(filename: str) -> list[torch.Tensor]:
    fields = []
    with h5py.File(filename, "r") as f:
        # Sort the keys numerically
        for name in sorted(f.keys(), key=int):
            data = f[name][()]
            tensor = torch.from_numpy(data)
            fields.append(tensor)
    return fields


def load_scalars(file_path):
    data_list = []
    with h5py.File(file_path, "r") as f:
        # Sort the keys numerically
        for key in sorted(f.keys(), key=lambda x: int(x.split("_")[1])):
            data_array = f[key][()]
            data_list.append(data_array)
    return data_list


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


def get_distances_to_borders(pos, cells):
    faces = np.array(cells)
    points = np.array(pos)
    bars = extract_border_edges(faces)
    border_bars_node_ids = np.unique(np.ravel(bars))
    is_border = np.zeros(len(points), dtype=bool)
    is_border[border_bars_node_ids] = True
    search_index = KDTree(points[is_border])
    sdf, _ = search_index.query(points, return_distance=True)
    return is_border, sdf


def sinusoidal_embedding(
    x: torch.Tensor, num_basis: int = 8, max_coord: float = 2.0, spacing: float = 1.0
) -> torch.Tensor:
    # Normalize and compute frequencies
    x = x / spacing
    max_seq = max_coord / spacing
    exponents = -math.log(max_seq * 4 / math.pi) / (num_basis - 1)
    div_term = torch.exp(torch.arange(num_basis, device=x.device) * exponents)
    emb = x.unsqueeze(-1) * div_term
    sin_emb = emb.sin()
    cos_emb = emb.cos()
    # Concat and flatten the last two dims: -> [*, D * 2 * num_basis]
    return torch.cat([sin_emb, cos_emb], dim=-1).flatten(-2, -1)


def angles_to_planes(coords):
    x, y = coords[:, 0], coords[:, 1]
    angles = torch.stack(
        [
            torch.atan2(y, x),
            torch.atan2(y, -x),
            torch.atan2(-y, -x),
            torch.atan2(-y, x),
        ],
        dim=1,
    )
    return angles


def spherical_harmonics(angle: torch.Tensor, l_max: int = 3) -> torch.Tensor:
    cos_t = torch.cos(angle).cpu().numpy()
    harmonics = []
    for l in range(l_max + 1):
        coeffs = np.zeros(l + 1)
        coeffs[-1] = 1
        P_l = np.polynomial.legendre.Legendre(coeffs)
        harmonics.append(
            torch.tensor(P_l(cos_t), device=angle.device, dtype=angle.dtype).unsqueeze(
                -1
            )
        )
    return torch.cat(harmonics, dim=-1)  # [*, l_max+1]
