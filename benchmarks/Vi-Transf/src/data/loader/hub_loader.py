from .loader import Loader
from plaid.containers.dataset import Dataset as PlaidDataset
from plaid.problem_definition import ProblemDefinition
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid
from datasets import load_dataset


class HubLoader(Loader):
    def __init__(self, bridge: callable, dataset_name: str, task_split: str, cache_dir: str, processes_number: int=1):
        """
        Initializes the HubLoader with the given parameters.

        :param bridge: A callable that bridges the dataset loading process.
        :param dataset_dir: The directory where the dataset is stored.
        :param task_split: The split of the dataset to load (default is "all_samples").
        :param load_from: The source from which to load the dataset (default is "huggingface").
        """
        super().__init__(bridge, task_split, processes_number=processes_number)
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

    def load_plaid(self, **kwargs) -> tuple[ProblemDefinition, PlaidDataset, ...]:
        hf_dataset = None
        try:
            hf_dataset = load_dataset(self.dataset_name, split="all_samples", cache_dir=self.cache_dir)
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print(f"Please refer to the documentation (https://huggingface.co/PLAID-datasets)")
            print(f"Provide a correct dataset_name with format 'PLAID-datasets/DATASET'.")
            raise e

        plaid_dataset, problem_definition = huggingface_dataset_to_plaid(hf_dataset)

        return problem_definition, *self.get_dataset_split(problem_definition, plaid_dataset)
