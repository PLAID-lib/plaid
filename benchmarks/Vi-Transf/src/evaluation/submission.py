from torch_geometric.data import Data
from typing import List
from ..data.loader.loader import PlaidDataset
import torch.nn as nn
import tqdm as tqdm
import pickle
import os


def create_submission(prediction_dataset: List[Data], plaid_dataset: PlaidDataset, save_dir: str="."):
    pyg_id_dico = {}
    for n_data, data in enumerate(prediction_dataset):
        pyg_id_dico[data.sample_id] = n_data
    
    ids_test        = plaid_dataset.get_sample_ids()
    pyg_sample_ids  = pyg_id_dico.keys()

    assert set(ids_test)==set(pyg_sample_ids), "Dataset sample ids do not match!"

    output_fields_names = prediction_dataset[0].output_fields_names
    output_scalars_names = prediction_dataset[0].output_scalars_names if hasattr(prediction_dataset[0], "output_scalars_names") else []
    
    reference = []
    for i, id in enumerate(ids_test):
        reference.append({})
        sample = prediction_dataset[pyg_id_dico[id]]
        output_fields_prediction    = sample.fields_prediction.detach().cpu().numpy()               if hasattr(sample, "fields_prediction")  and sample.fields_prediction  is not None    else None
        output_scalars_predictions  = sample.scalars_prediction.reshape(-1).detach().cpu().numpy()  if hasattr(sample, "scalars_prediction") and sample.scalars_prediction is not None    else None

        for n_field, fn in enumerate(output_fields_names):
            reference[i][fn] = output_fields_prediction[:, n_field]

        for n_scalar, sn in enumerate(output_scalars_names):
            reference[i][sn] = output_scalars_predictions[n_scalar]

    with open(os.path.join(save_dir, "reference.pkl"), "wb") as file:
        pickle.dump(reference, file)
