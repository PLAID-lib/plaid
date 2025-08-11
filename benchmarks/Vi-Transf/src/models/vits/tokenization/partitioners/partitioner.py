from typing import List
from torch_geometric.data import Data

class Partitioner():
    def __init__(self, n_vertices_per_subdomain):
        self.n_vertices_per_subdomain = n_vertices_per_subdomain

    def partition(self, dataset: List[Data]) -> List[Data]:
        raise NotImplementedError("This method should be implemented in the subclass")