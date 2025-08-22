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
from .flatten_tokenizer import FlattenTokenizer, process_data_tuple, pad_subdomains


class TemporalFlattenTokenizer(FlattenTokenizer):
    def __init__(
        self,
        partitioner: Partitioner,
        output_field_dim: int,
        tokenization_type: Literal["morton", "simple"] = "morton",
        processes_number=1,
    ):
        super().__init__(
            partitioner, output_field_dim, tokenization_type, processes_number
        )

    def preprocess(self, dataset: list[list[Data]], seed: int):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print("Using random seed for partitioning:", seed)
        first_samples_dataset = [ds[0] for ds in dataset]
        first_sample_dataset = self.partitioner.partition(first_samples_dataset)
        n_tokens_per_sim = max([data.n_communities for data in first_sample_dataset])
        first_samples_dataset = [
            process_data_tuple(
                self.data_tokenizer_fn_name,
                sample,
                n_vertices_per_subdomain=self.n_vertices_per_subdomain,
                n_tokens_per_sim=n_tokens_per_sim,
            )
            for sample in first_samples_dataset
        ]

        tokenized_dataset = []
        for n, sample_list in enumerate(dataset):
            for sample in sample_list:
                sample.communities = first_samples_dataset[n].communities
                sample.n_communities = first_samples_dataset[n].n_communities
                sample.padded_community_orders = first_samples_dataset[
                    n
                ].padded_community_orders
                sample.community_reverse_orders = first_samples_dataset[
                    n
                ].community_reverse_orders

                tokens = rearrange(
                    sample.x[sample.padded_community_orders], "t n d -> t (n d)"
                )
                cross_domain_padded_token, attn_mask = pad_subdomains(
                    tokens, n_tokens_per_sim
                )
                sample.tokens = cross_domain_padded_token.unsqueeze(0)
                sample.attn_mask = attn_mask.unsqueeze(0)

                if hasattr(sample, "output_fields"):
                    expanded_output_fields = rearrange(
                        sample.output_fields[sample.padded_community_orders],
                        "t n d -> t (n d)",
                    )
                    cross_domain_padded_expanded_output_fields, _ = pad_subdomains(
                        expanded_output_fields, n_tokens_per_sim
                    )
                    sample.expanded_output_fields = (
                        cross_domain_padded_expanded_output_fields.unsqueeze(0)
                    )

                tokenized_dataset.append(sample)

        return tokenized_dataset
