import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import List, Optional

class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def preprocess(self, dataset: List[Data], seed: Optional[int]=None) -> List[Data]:
        pass
        
    def forward(self, data: Batch):
        pass

    def untokenize(self, data: Batch):
        pass