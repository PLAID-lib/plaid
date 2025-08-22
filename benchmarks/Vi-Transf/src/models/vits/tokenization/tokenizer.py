from typing import Optional

import torch.nn as nn
from torch_geometric.data import Batch, Data


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()

    def preprocess(self, dataset: list[Data], seed: Optional[int] = None) -> list[Data]:
        pass

    def forward(self, data: Batch):
        pass

    def untokenize(self, data: Batch):
        pass
