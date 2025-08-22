from plaid.containers.dataset import Dataset as PlaidDataset
from typing import List
import copy
from torch_geometric.data import Data

def extract_plaid_dataset(plaid_dataset: PlaidDataset, ids: List[int]) -> PlaidDataset:
    extracted_dataset = copy.deepcopy(plaid_dataset)
    sample_dict = plaid_dataset.get_samples(ids=ids)
    extracted_dataset._samples = sample_dict

    return extracted_dataset

def split_plaid_train_test(plaid_dataset: PlaidDataset, train_ids: List[int], test_ids: List[int]) -> tuple[PlaidDataset, PlaidDataset]:
    plaid_train_dataset = extract_plaid_dataset(plaid_dataset, ids=train_ids)
    plaid_test_dataset = extract_plaid_dataset(plaid_dataset, ids=test_ids)

    return plaid_train_dataset, plaid_test_dataset

def split_pyg_train_test(pyg_dataset: List[Data], train_ids: List[int], test_ids: List[int]) -> tuple[List[Data], List[Data]]:
    train_dataset   = []
    test_dataset    = []

    for data in pyg_dataset:
        if data.sample_id in train_ids:
            train_dataset.append(data)
        elif data.sample_id in test_ids:
            test_dataset.append(data)

    return train_dataset, test_dataset

def split_temporal_pyg_train_test(pyg_dataset: List[Data], train_ids: List[int], test_ids: List[int]) -> tuple[List[Data], List[Data]]:
    train_dataset   = []
    test_dataset    = []

    for data_list in pyg_dataset:
        if data_list[0].sample_id in train_ids:
            train_dataset.append(data_list)
        elif data_list[0].sample_id in test_ids:
            test_dataset.append(data_list)

    return train_dataset, test_dataset
