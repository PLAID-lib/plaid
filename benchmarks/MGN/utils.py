import h5py
import torch
import networkx
import numpy as np
from typing import List, Tuple
from sklearn.metrics import r2_score
from Muscat.Types import MuscatFloat
from Muscat.Containers.MeshInspectionTools import ComputeMeshMinMaxLengthScale
from Muscat.Containers.AnisotropicMetricComputation import ComputeGradient, ComputeHessian


def get_bandwidth(mesh) -> float:
    lengthscale = ComputeMeshMinMaxLengthScale(mesh)
    return lengthscale


def relative_rmse_field(y_true: List[torch.Tensor], y_pred: List[torch.Tensor], threshold: float=0.0) -> torch.Tensor:
    return torch.sqrt(
        torch.mean(
            torch.stack(
                [torch.linalg.norm(y - y_hat, axis=0)**2/(len(y)*step_function_field(y, threshold)**2) for y, y_hat in zip(y_true, y_pred)]
            ), dim=0
        )
    )


def save_fields(filename: str, fields: List[torch.Tensor]) -> None:
    with h5py.File(filename, "w", libver='latest') as f:
        for idx, field in enumerate(fields):
            dset = f.create_dataset(str(idx), data=field.cpu().numpy())
    return None


def save_scalars(file_path, data_list):
    with h5py.File(file_path, 'w') as f:
        for i, data_array in enumerate(data_list):
            f.create_dataset(f"array_{i}", data=data_array)


def load_fields(filename: str) -> List[torch.Tensor]:
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
    with h5py.File(file_path, 'r') as f:
        # Sort the keys numerically
        for key in sorted(f.keys(), key=lambda x: int(x.split('_')[1])):
            data_array = f[key][()]
            data_list.append(data_array)
    return data_list
