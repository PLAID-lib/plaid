import torch
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import ReLU, GELU
from typing import List

class MLP(torch.nn.Module):
    def __init__(self, layers_sizes: List[int], dropout: float=0.1, activation: str='relu'):
        activation_layer = GELU if activation == 'gelu' else ReLU
        super(MLP, self).__init__()
        layers = [Linear(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes)-1)]
        self.mlp = Sequential()
        for i, layer in enumerate(layers):
            self.mlp.append(layer)
            if i < len(layers) - 1:
                self.mlp.append((activation_layer()))
                self.mlp.append(torch.nn.Dropout(p=dropout))
        self.mlp.append(torch.nn.Dropout(p=dropout))

    def forward(self, x):
        return self.mlp(x)