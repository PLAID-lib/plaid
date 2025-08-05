from ..model import BaseModel
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Union, Callable, Optional
from torch_geometric.data import Data, Batch
from .tokenization.tokenizer import Tokenizer
from ..blocks import MLP
from einops import rearrange
import torch.nn.functional as F
import random

# B: Batch size
# T: Number of tokens
# L: Latent dimension of the tokens
# Sin: Number of input scalars
# Fin: Number of input fields
# Sout: Number of output scalars
# Fout: Number of output fields
# Nv: Number of vertices per subdomain

class FlatFormerCLSLess(BaseModel):
    def __init__(self,
                n_vertices_per_subdomain: int,
                n_head: int,
                dim_ff: int,
                num_layers: int,
                tokenizer: Tokenizer,
                input_field_dim: int=None,
                input_scalar_dim: int=None,
                output_scalar_dim: int=None,
                output_field_dim: int=None,
                activation: Union[str, Callable[[Tensor], Tensor]]="relu",
                norm_first: bool=True,
                dropout: float=0.1,
                latent_dim: Optional[int]=None,
                **kwargs
                ):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_scalar_dim = output_scalar_dim
        self.output_field_dim = output_field_dim
        self.input_scalar_dim = input_scalar_dim
        self.input_field_dim = input_field_dim

        self.criterion = nn.MSELoss(reduction="none")

        # Tokenizer
        token_dim = n_vertices_per_subdomain * input_field_dim
        self.tokenizer = tokenizer
        self.n_head = n_head
        
        # Linear projection of the flattened tokens onto the latent space
        self.encoder = nn.Linear(token_dim, latent_dim - input_scalar_dim, bias=False)

        # CLS token for the transformer encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim), requires_grad=True) # 1 1 L
        nn.init.xavier_uniform_(self.cls_token)

        # Transformer encoder
        encoder_layers  = TransformerEncoderLayer(d_model=latent_dim,
                                                  nhead=n_head,
                                                  dim_feedforward=dim_ff,
                                                  dropout=dropout,
                                                  batch_first=True,
                                                  norm_first=norm_first,
                                                  dtype=torch.float32,
                                                  activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # MLP decoder
        self.decoder     = nn.Linear(latent_dim + input_scalar_dim, n_vertices_per_subdomain*(self.output_field_dim + self.output_scalar_dim), bias=False)
        
    def preprocess(self, pyg_dataset: List[Data], seed: Optional[int]=None, **kwargs) -> List[Data]:
        if seed is None: seed = random.randint(0, 2**32 - 1)
        print(f"Using seed: {seed} for data preprocessing")
        dataset = self.tokenizer.preprocess(pyg_dataset, seed=seed)
        
        return dataset

    def forward(self, data_batch: Data | Batch):
        tokens, src_key_padding_mask = self.tokenizer(data_batch) # B T (Nv * F)
        if self.input_scalar_dim > 0:
            input_scalars = data_batch.input_scalars.to(self._device)
        else:
            input_scalars = None

        field_predictions, scalar_predictions = self.forward_batch(input_scalars, tokens, src_key_padding_mask)
        field_predictions   = self.tokenizer.untokenize_predictions(field_predictions, data_batch)

        return field_predictions, scalar_predictions

    def forward_batch(self, input_scalars, tokens, src_key_padding_mask):
        # latent projection
        latent_tokens = self.encoder(tokens) # B T (Nv * F) -> B T (L - Sin)

        # concatenating the scalars to each token
        if self.input_scalar_dim > 0:
            scalar_concat_tensor    = input_scalars.unsqueeze(1).expand((-1, latent_tokens.shape[1], -1)) # B Sin -> B T Sin
            latent_tokens           = torch.concat((latent_tokens, scalar_concat_tensor), dim=2) # B T (L - Sin) -> B T L
        
        # transformer encoder
        encoded_tokens = self.transformer_encoder(latent_tokens, src_key_padding_mask=src_key_padding_mask) # B T L
        
        # reinjecting the scalars
        if self.input_scalar_dim > 0:
            scalar_concat_tensor    = input_scalars.unsqueeze(1).expand((-1, encoded_tokens.shape[1], -1)) # B Sin -> B T Sin
            encoded_tokens          = torch.concat((encoded_tokens, scalar_concat_tensor), dim=2) # B T L -> B T (L + Sin)

        # field predictions
        predictions = self.decoder(encoded_tokens) # B T (Fout * Nv)
        predictions = rearrange(predictions, 'b t (n f) -> b t n f', f=(self.output_field_dim + self.output_scalar_dim)) # B T Nv (Fout + Sout)

        field_predictions   = predictions[:, :, :, :self.output_field_dim]
        scalar_predictions  = predictions[:, :, :, self.output_field_dim:].mean(dim=(1, 2))
        if self.output_scalar_dim == 0:
            scalar_predictions = None
        
        return field_predictions, scalar_predictions

    def compute_loss(self, data_batch: Data | Batch) -> torch.Tensor:
        tokens, src_key_padding_mask = self.tokenizer(data_batch)
        tokens, src_key_padding_mask = tokens.to(self._device), src_key_padding_mask.to(self._device)

        if self.input_scalar_dim > 0:
            input_scalars = data_batch.input_scalars.to(self._device)
        else:
            input_scalars = None

        # Forward pass
        field_predictions, scalar_predictions = self.forward_batch(input_scalars, tokens, src_key_padding_mask)

        # Removing padding tokens
        field_predictions = field_predictions[~src_key_padding_mask, ...]
        expanded_output_fields = data_batch.expanded_output_fields.to(self._device)[~src_key_padding_mask, ...].reshape(field_predictions.shape)

        # Computing losses. It is also computed for padding nodes thanks to the expanded_output_fields tensor
        field_loss  = self.criterion(field_predictions, expanded_output_fields).mean(dim=(0, 1))
        if self.output_scalar_dim > 0:
            scalar_loss = self.criterion(scalar_predictions, data_batch.output_scalars.to(self._device)).mean(dim=0)
        else:
            scalar_loss = torch.tensor(0, dtype=torch.float32, requires_grad=False)

        return field_loss, scalar_loss

    def predict(self, data: Data):
        tokens, src_key_padding_mask = self.tokenizer(data)
        tokens, src_key_padding_mask = tokens.to(self._device), src_key_padding_mask.to(self._device)
        if self.input_scalar_dim > 0:
            input_scalars = data.input_scalars.to(self._device)
        else:
            input_scalars = None
        field_predictions, scalar_predictions = self.forward_batch(input_scalars, tokens, src_key_padding_mask)
        
        # removing padding tokens
        field_predictions = field_predictions[~src_key_padding_mask, ...]
        
        # reconstructing the output_field matrix for the data sample
        field_predictions = self.tokenizer.untokenize(field_predictions.cpu(), data, keep_list=True)

        field_predictions = field_predictions[0]
        if self.output_scalar_dim > 0:
            scalar_predictions = scalar_predictions[0].cpu()

        return field_predictions, scalar_predictions

    @property
    def _device(self):
        return self.encoder.weight.device