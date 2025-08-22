import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from plaid.containers.dataset import Dataset as PlaidDataset


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def preprocess(
        self,
        pyg_dataset: list[Data],
        plaid_dataset: PlaidDataset,
        seed: int,
        type: str,
        **kwargs,
    ):
        pass

    def forward(self, data: Batch):
        pass

    def postprocess(
        self, pyg_dataset: list[Data], plaid_dataset: PlaidDataset, type: str, **kwargs
    ):
        return pyg_dataset

    def evaluate(self, data: Data):
        pass
