from plaid.containers.dataset import Dataset as PlaidDataset
from plaid.problem_definition import ProblemDefinition
from typing import List, Union
from torch_geometric.data import Data
import os
from tqdm import tqdm
from multiprocessing import Pool

class Loader():
    def __init__(self,
                 bridge: callable,
                 task_split: Union[List[str], str, List[Union[List[str], str]]],
                 processes_number: int=1):
        """
        Initializes the Loader with the given parameters.

        :param bridge: A callable that bridges the dataset loading process.
        :param task_split: The split(s) of the dataset to load.
        """

        self.bridge = bridge
        self.task_split: Union[List[str], str, List[Union[List[str], str]]] = task_split
        self.processes_number = processes_number

    def load_plaid(self, verbose: bool) -> tuple[ProblemDefinition, List[PlaidDataset]]:
        raise NotImplementedError("This method should be implemented in the subclass")

    def get_dataset_split(self,
                          problem_definition: ProblemDefinition,
                          plaid_dataset: PlaidDataset) -> Union[PlaidDataset, tuple[PlaidDataset]]:
        if not type(self.task_split)==str:
            datasets = []
            for split in self.task_split:
                if type(split)==str:    split_ids = problem_definition.get_split(split)
                else:                   split_ids = [split_id for split_str in split for split_id in problem_definition.get_split(split_str)]
                dataset = PlaidDataset()
                dataset.set_samples(plaid_dataset.get_samples(ids=split_ids))
                datasets.append(dataset)

            return tuple(datasets)

        ids = problem_definition.get_split(self.task_split)
        dataset = PlaidDataset()
        dataset.set_samples(plaid_dataset.get_samples(ids=ids))
        return (dataset, )

    def load(self,
             verbose=False) -> tuple[ProblemDefinition, tuple[List[Data], ...]]:
        """
        Load and converts a plaid dataset to torch geometric format.

        Returns:
            tuple[ProblemDefinition, Union[List[Data], tuple[List[Data], ...]]]:
            A tuple containing the problem definition and either a single list of Data objects
            or a tuple of multiple lists of Data objects.
        """

        buffer = self.load_plaid(verbose=verbose)
        problem_definition = buffer[0]
        dataset_list = buffer[1:]

        processed_list = []
        for dataset in dataset_list:
            processed_list.append(self.plaid_to_bridge(dataset, problem_definition=problem_definition, verbose=verbose))

        return problem_definition, *processed_list

    def plaid_to_bridge(self,
                        dataset: PlaidDataset,
                        problem_definition: ProblemDefinition,
                        verbose= True) -> List[Data]:
        """
        Converts a Plaid dataset to PytorchGeometric dataset

        Args:
            dataset (plaid.containers.dataset.Dataset): Plaid dataset

        Returns:
            List[Data]: the converted dataset
        """
        if verbose: print("in bridge")
        if self.processes_number == -1:
            self.processes_number = os.cpu_count()
        data_list = []
        sample_ids, samples = list(zip(*list(dataset.get_samples().items())))
        if self.processes_number==0 or self.processes_number==1:
            if verbose: iterator = tqdm(zip(samples, sample_ids), total=len(samples))
            else: iterator = zip(samples, sample_ids)
            for sample, sample_id in iterator:
                new_data = self.bridge(sample, sample_id, problem_definition)
                data_list.append(new_data)
            return data_list

        with Pool(processes=self.processes_number) as p:
            args_iter = zip(samples, sample_ids, [problem_definition]*len(samples))
            if verbose:
                iterator = tqdm(p.starmap(self.bridge, args_iter), total=len(samples))
            else:
                iterator = p.starmap(self.bridge, args_iter)
            for new_data in iterator:
                # if isinstance(new_data, list):
                #     data_list.extend(new_data)
                # else:
                data_list.append(new_data)

        return data_list
