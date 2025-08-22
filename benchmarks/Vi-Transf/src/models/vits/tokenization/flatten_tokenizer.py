from .tokenizer import Tokenizer
from .partitioners.partitioner import Partitioner
import torch
from torch.multiprocessing import Pool
from typing import Optional, Literal
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import os
from .flatten_data_tokenizers import data_tokenizer_registry
import random
from einops import rearrange


class FlattenTokenizer(Tokenizer):
    def __init__(self,
                 partitioner: Partitioner,
                 output_field_dim: int,
                 tokenization_type: Literal["morton", "simple"]="morton",
                 processes_number=1):
        super().__init__()
        self.partitioner                    = partitioner
        self.n_vertices_per_subdomain       = partitioner.n_vertices_per_subdomain
        self.data_tokenizer_fn_name: str    = tokenization_type
        self.data_tokenizer_fn: callable    = data_tokenizer_registry[tokenization_type]
        self.processes_number               = os.cpu_count() if processes_number == -1 else processes_number
        self.output_field_dim              = output_field_dim

    def _tokenize(self, dataset: list[Data]) -> list[Data]:
        """Tokenizes the dataset using the specified partitioner."""
        token_dim = dataset[0].x.shape[1] * self.n_vertices_per_subdomain
        n_tokens_per_sim = max([datapoint.n_communities for datapoint in dataset])

        tokenized_dataset = []
        print(f"Using {self.processes_number} processes for the tokenizer preprocessing.")
        if self.processes_number==0 or self.processes_number==1:
            for datapoint in tqdm(dataset):
                data = process_data_tuple(self.data_tokenizer_fn_name, datapoint, self.n_vertices_per_subdomain, n_tokens_per_sim)
                tokenized_dataset.append(data)
        else:
            with Pool(self.processes_number) as p:
                for processed_datapoint in tqdm(
                    p.starmap(process_data_tuple, zip([self.data_tokenizer_fn_name]*len(dataset), dataset, [self.n_vertices_per_subdomain]*len(dataset), [n_tokens_per_sim]*len(dataset))),
                    total=len(dataset)
                ):
                    tokenized_dataset.append(processed_datapoint)
        return tokenized_dataset


    def preprocess(self, dataset, seed: Optional[int]=None) -> list[Data]:
        # partitioning each datapoint in the dataset
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print("Using random seed for partitioning:", seed)
        dataset = self.partitioner.partition(dataset, seed=seed)

        # tokenizing each datapoint in the dataset
        dataset = self._tokenize(dataset)
        return dataset

    def forward(self, data):
        # Flatten the input data
        return data.tokens, data.attn_mask

    def untokenize(self, full_predictions: torch.Tensor, data_batch: Batch | Data, keep_list=False) -> torch.tensor:
        # full_predictions: B T D
        result_list = []

        if isinstance(data_batch, Batch):
            data_batch = data_batch.to_data_list()
        else:
            data_batch = [data_batch]
            full_predictions = full_predictions[None, ...]

        for i, data in enumerate(data_batch):
            new_result = untokenize_prediction_data(full_predictions[i], data, self.output_field_dim)
            result_list.append(new_result)

        if keep_list:
            return result_list

        return torch.vstack(result_list)

def untokenize_prediction_data(full_predictions, data, pred_dim):
    """Unflattens and removes padding-associated outputs"""
    return rearrange(full_predictions, "t n d -> (t n) d")[data.community_reverse_orders]


def process_data_tuple(data_tokenizer_fn_name, datapoint, n_vertices_per_subdomain, n_tokens_per_sim):
    data_tokenizer_fn = data_tokenizer_registry[data_tokenizer_fn_name]
    data = data_tokenizer_fn(datapoint, n_vertices_per_subdomain)

    cross_domain_padded_token, attn_mask = pad_subdomains(data.tokens, n_tokens_per_sim)
    data.tokens = cross_domain_padded_token.unsqueeze(0)
    data.attn_mask = attn_mask.unsqueeze(0)

    if hasattr(data, 'expanded_output_fields'):
        cross_domain_padded_expanded_output_fields, _ = pad_subdomains(data.expanded_output_fields, n_tokens_per_sim)
        data.expanded_output_fields = cross_domain_padded_expanded_output_fields.unsqueeze(0)

    return data


def pad_subdomains(tokens, n_tokens_per_sim):
    token_dim = tokens.shape[1]
    n_sequence_tokens = tokens.shape[0]
    pad_token = torch.zeros((1, token_dim))

    if n_tokens_per_sim > n_sequence_tokens:
        tokens = torch.cat([tokens, pad_token.tile((n_tokens_per_sim - n_sequence_tokens, 1))])
    else: assert n_tokens_per_sim == n_sequence_tokens, f"n_tokens_per_sim ({n_tokens_per_sim}) must be equal to the number of sequence tokens ({n_sequence_tokens}) or greater."
    mask = torch.ones(n_tokens_per_sim)
    mask[n_sequence_tokens:] = 0
    mask = (mask==0).to(tokens.device)

    return tokens, mask
