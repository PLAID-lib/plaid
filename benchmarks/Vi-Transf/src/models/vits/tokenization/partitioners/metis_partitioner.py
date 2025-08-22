import random
from typing import Optional

import numpy as np
import pymetis
import torch
from pymetis import Options
from torch.multiprocessing import Pool, cpu_count
from torch_geometric.data import Data
from tqdm import tqdm

from .partitioner import Partitioner


def npoints_to_nparts(
    npoints: int, n_sim_points: int, absolute_tol: int, relative_tol: float
) -> int:
    """Computes the number of subdomains in a simulation, with a tolerance given the number of points per subdomains we want and the number of points in the simulation."""
    nparts = n_sim_points // npoints + 1
    nparts = int(np.ceil((1 + relative_tol) * nparts + absolute_tol))

    return nparts


def torch_geometric_to_metis_format(data: Data):
    """Convert torch-geometric graph data to a Pythonic adjacency list format suitable for METIS part_graph.

    Args:
        data (Data): A torch-geometric Data object containing:
            - `edge_index` (torch.Tensor): Tensor of shape (2, num_edges) representing edges.

    Returns:
        list[list[int]]: A Pythonic adjacency list, where adjacency[i] is a list of
        nodes adjacent to node i.
    """
    num_nodes = data.num_nodes
    adjacency = [[] for _ in range(num_nodes)]

    for source, target in zip(
        data.edge_index[0, :].numpy(), data.edge_index[1, :].numpy()
    ):
        adjacency[source].append(target)

    return adjacency


class MetisPartitioner(Partitioner):
    def __init__(
        self,
        n_vertices_per_subdomain: int = None,
        processes_number: int = 1,
        absolute_tol: int = 1,
        relative_tol: float = 0.05,
    ):
        super().__init__(n_vertices_per_subdomain)
        self.processes_number = processes_number
        self.absolute_tol = absolute_tol
        self.relative_tol = relative_tol

    def partition(self, dataset: list[Data], seed: Optional[int] = None) -> np.ndarray:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        if seed is not None:
            rng_generator = random.Random(seed)
            seed_vector = [
                rng_generator.randint(0, 2**30 - 1) for _ in range(len(dataset))
            ]  # metis doesn't accept seeds that are too large

        if self.processes_number == -1:
            self.processes_number = cpu_count()

        partitioned_dataset = []

        if self.processes_number > 1:
            with Pool(processes=self.processes_number) as pool:
                partitioned_dataset = list(
                    tqdm(
                        pool.starmap(
                            _partition_single,
                            zip(
                                dataset,
                                [self.n_vertices_per_subdomain] * len(dataset),
                                seed_vector,
                                [self.absolute_tol] * len(dataset),
                                [self.relative_tol] * len(dataset),
                            ),
                        ),
                        desc="Partitioning dataset with METIS",
                        total=len(dataset),
                    )
                )
        else:
            for i, data in enumerate(
                tqdm(
                    dataset, desc="Partitioning dataset with METIS", total=len(dataset)
                )
            ):
                partitioned_data = _partition_single(
                    data,
                    self.n_vertices_per_subdomain,
                    seed_vector[i],
                    self.absolute_tol,
                    self.relative_tol,
                )
                partitioned_dataset.append(partitioned_data)

        return partitioned_dataset


def _partition_single(
    data: Data, npoints, seed=None, absolute_tol: int = 1, relative_tol: float = 0.05
) -> torch.Tensor:
    if seed is None:
        seed = random.randint(0, 2**30 - 1)
    options = Options(seed=seed)

    n_sim_points = data.x.shape[0]
    n_subdomains = npoints_to_nparts(
        npoints=npoints,
        n_sim_points=n_sim_points,
        absolute_tol=absolute_tol,
        relative_tol=relative_tol,
    )

    adjacency = torch_geometric_to_metis_format(data)

    _, communities = pymetis.part_graph(
        nparts=n_subdomains,
        adjacency=adjacency,
        xadj=None,
        adjncy=None,
        vweights=None,
        eweights=None,
        recursive=None,
        contiguous=None,
        options=options,
    )
    communities = torch.tensor(communities, dtype=torch.long)
    for community in torch.unique(communities):
        assert torch.sum(communities == community) <= npoints, (
            f"community {community} is too large for sample {data.sample_id}"
        )

    data.communities = communities.to(data.x.device)
    data.n_communities = n_subdomains

    return data
