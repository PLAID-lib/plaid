import torch
from torch_geometric.data import Data
import numpy as np
from .morton import compute_morton_order
from einops import rearrange

def flatten_tokenizer(data: Data, n_vertices_per_subdomain: int, tokenization_type="morton"):
        n_communities               = data.n_communities
        padded_community_orders     = torch.empty((n_communities, n_vertices_per_subdomain), dtype=torch.int)
        community_reverse_orders    = torch.empty(data.x.shape[0], dtype=torch.int)

        for i in range(n_communities):
            community_map = (data.communities == i)
            community_size = torch.sum(community_map).item()
            community_pos = data.pos[community_map, :]

            n_repeat = n_vertices_per_subdomain // community_size + 1
            if n_vertices_per_subdomain // community_size == 0:
                raise ValueError(f"n_vertices_per_subdomain ({n_vertices_per_subdomain}) is smaller than the number of features ({data.x.shape[1]}).")

            if tokenization_type == "morton":
                local_community_order = compute_morton_order(community_pos)
            else:
                local_community_order = np.arange(community_size)
            
            padded_local_community_order = np.tile(local_community_order, n_repeat)[:n_vertices_per_subdomain]
            local_reverse_community_order = np.zeros((community_size,), dtype=int)
            local_reverse_community_order[local_community_order] = np.arange(community_size)
                
            local_to_global = np.where(community_map)[0]
            padded_community_order = local_to_global[padded_local_community_order]
            reverse_community_order = local_reverse_community_order + i*n_vertices_per_subdomain

            padded_community_orders[i, :]           = torch.from_numpy(padded_community_order).to(torch.int)
            community_reverse_orders[community_map] = torch.from_numpy(reverse_community_order).to(torch.int)

        data.padded_community_orders    = padded_community_orders # T n_vertices_per_subdomain
        data.community_reverse_orders   = community_reverse_orders # N
        data.tokens                     = rearrange(data.x[padded_community_orders], "t n d -> t (n d)") # T n_vertices_per_subdomain d -> T (n_vertices_per_subdomain d)
        if hasattr(data, "output_fields"):
            data.expanded_output_fields = rearrange(data.output_fields[padded_community_orders], "t n d -> t (n d)")
        
        return data


def flatten_simple(data: Data, n_vertices_per_subdomain: int):
    return flatten_tokenizer(data, n_vertices_per_subdomain, "simple")


def flatten_morton_ordered(data: Data, n_vertices_per_subdomain: int):
    return flatten_tokenizer(data, n_vertices_per_subdomain, "morton")


data_tokenizer_registry = {
    "morton": flatten_morton_ordered,
    "simple": flatten_simple,
}