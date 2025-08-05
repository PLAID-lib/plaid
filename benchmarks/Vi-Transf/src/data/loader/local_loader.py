from .loader import Loader
from plaid.containers.dataset import Dataset as PlaidDataset
from plaid.problem_definition import ProblemDefinition
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid
from pathlib import Path
from typing import Union, List, Tuple
from datasets import load_dataset, load_from_disk

import logging
logger = logging.getLogger()
# Set logging level for datasets library to WARNING or higher (to suppress INFO messages)
logging.getLogger("datasets").setLevel(logging.WARNING)


class LocalLoader(Loader):
    def __init__(self,
                 bridge: callable,
                 dataset_dir: Path,
                 task_split: Union[List[str], str],
                 processes_number: int=1):
        """
        Either loads a dataset from a pre-made dataset configuration or use explicit loading parameters
        """
        super().__init__(bridge, task_split, processes_number=processes_number)

        self.bridge = bridge
        self.dataset_dir = dataset_dir if type(dataset_dir)==Path else Path(dataset_dir)
        self.task_split: Union[List[str], str] = task_split
    

class LocalHFLoader(LocalLoader):
    def __init__(self, bridge: callable, dataset_dir: Path, task_split: Union[List[str], str], processes_number: int=1):
        """
        Initializes the LocalHFLoader with the given parameters.

        :param bridge: A callable that bridges the dataset loading process.
        :param dataset_dir: The directory where the dataset is stored.
        :param task_split: The split of the dataset to load (default is "all_samples").
        """
        super().__init__(bridge, dataset_dir, task_split, processes_number=processes_number)

    def load_arrow(self, hf_folder: str):
        raise NotImplementedError("This method should be implemented in the subclass")

    def load_plaid(self, verbose=True, **kwargs) -> Tuple[ProblemDefinition, PlaidDataset, ...]:
        hf_folder = self.dataset_dir / "huggingface"
        if verbose: print(f"Loading data from {hf_folder}")
        hf_dataset = self.load_arrow(str(hf_folder))
        plaid_dataset, problem_definition = huggingface_dataset_to_plaid(hf_dataset)

        return problem_definition, *self.get_dataset_split(problem_definition, plaid_dataset)
    
class LocalHFLoaderDisk(LocalHFLoader):
    def __init__(self, bridge: callable, dataset_dir: Path, task_split: Union[List[str], str], processes_number: int=1):
        super().__init__(bridge, dataset_dir, task_split, processes_number=processes_number)

    def load_arrow(self, hf_folder: str):
        return load_from_disk(str(hf_folder))

class LocalHFLoaderDataset(LocalHFLoader):
    def __init__(self, bridge: callable, dataset_dir: Path, task_split: Union[List[str], str], processes_number: int=1):
        super().__init__(bridge, dataset_dir, task_split, processes_number=processes_number)

    def load_arrow(self, hf_folder: str):
        return load_dataset(str(hf_folder), split="all_samples")


class LocalPlaidLoader(LocalLoader):
    def __init__(self, bridge: callable, dataset_dir: Path, task_split: Union[List[str], str], processes_number: int=1):
        """
        Initializes the LocalPlaidLoader with the given parameters.
        
        :param bridge: A callable that bridges the dataset loading process.
        :param dataset_dir: The directory where the dataset is stored.
        :param task_split: The split of the dataset to load (default is "all_samples").
        """
        super().__init__(bridge, dataset_dir, task_split, processes_number=processes_number)

    def load_plaid(self, verbose=True) -> Tuple[ProblemDefinition, PlaidDataset, ...]:
        # loading the problem definition
        problem_definition = ProblemDefinition()
        problem_path = self.dataset_dir / "plaid" / "problem_definition"
        problem_definition._load_from_dir_(problem_path)

        dataset_dir = self.dataset_dir / "plaid" / "dataset"
        plaid_dataset = PlaidDataset(dataset_dir, verbose=verbose, processes_number=self.processes_number)
        print(plaid_dataset.get_sample_ids())

        return problem_definition, *self.get_dataset_split(problem_definition, plaid_dataset)