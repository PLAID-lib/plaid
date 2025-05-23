from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from torch_geometric.data import Data
import numpy as np
import torch
from torch_geometric.utils._coalesce import coalesce as geometric_coalesce
from plaid.containers.dataset import Dataset as PlaidDataset
from plaid.problem_definition import ProblemDefinition
from typing import List, Tuple, Union, Annotated
from datasets import load_dataset
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid
from torch_geometric.data import Data
import os
from tqdm import tqdm
from multiprocessing import Pool

def my_coalesce(edges: torch.Tensor | np.ndarray, num_nodes: int, reduce="add"):
    if isinstance(edges, np.ndarray):
        edges = torch.tensor(edges).T
        return geometric_coalesce(edges, num_nodes=num_nodes, reduce=reduce).T.numpy()
    edges = geometric_coalesce(edges.T, num_nodes=num_nodes, reduce=reduce).T
    return edges

def faces_to_edges(faces: np.ndarray, num_nodes: int):
    """Creates a list of edges from a Faces array

    Args:
        faces (np.ndarray): Array of faces shape (n_faces, face_dim)

    Returns:
        np.ndarray: the edge list of shape (n, 2)
    """

    assert len(faces.shape)==2, "Wrong shape for the faces, should be a 2D array"

    # Generate edges (without duplicates in one pass)
    rolled = np.roll(faces, -1, axis=1)
    edges = np.vstack((faces.ravel(), rolled.ravel())).T
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)

    return edges


def sample_to_pyg(sample: Sample, sample_id: int, problem_definition: ProblemDefinition, base_name: str) -> Data:
    """
    Converts a Plaid sample to PytorchGeometric Data object

    Args:
        sample (plaid.containers.sample.Sample): data sample
        sample_id (int): Plaid sample id
        problem_definition (ProblemDefinition)
        base_name (str): Name of the base to extract

    Returns:
        Data: the converted data sample
    """

    vertices = sample.get_vertices(base_name=base_name)
    edge_index = []
    faces_dict = {}
    for key, faces in sample.get_elements(base_name=base_name).items():
        edge_index.append(faces_to_edges(faces, num_nodes=vertices.shape[0], coalesce=False))
        faces_dict[key] = torch.tensor(faces, dtype=torch.long)
    edge_index = np.concatenate(edge_index, axis=0)
    edge_index = my_coalesce(edge_index, num_nodes=vertices.shape[0])

    v1 = vertices[edge_index[:, 0]]
    v2 = vertices[edge_index[:, 1]]
    edge_weight = np.linalg.norm(v2 - v1, axis=1)

    # loading scalars
    input_scalars_names     = problem_definition.get_input_scalars_names()
    output_scalars_names    = problem_definition.get_output_scalars_names()

    input_scalars   = []
    output_scalars  = []
    for name in input_scalars_names:
        input_scalars.append(sample.get_scalar(name))
    for name in output_scalars_names:
        output_scalars.append(sample.get_scalar(name))

    # loading fields
    input_fields_names   = problem_definition.get_input_fields_names()
    output_fields_names  = problem_definition.get_output_fields_names()

    if "cell_ids" in input_fields_names:
        input_fields_names.remove("cell_ids")

    new_input_fields_names = []
    input_fields    = []
    if len(input_fields_names)>=1:
        for field_name in input_fields_names:
            field = sample.get_field(field_name, base_name=base_name)
            if field is None: continue
            input_fields.append(field)
            new_input_fields_names.append(field_name)
    if len(input_fields) >= 1:
        input_fields = np.vstack(input_fields).T
        input_fields = np.concatenate((vertices, input_fields), axis=1)
        new_input_fields_names = ["x", "y", *new_input_fields_names]
    else:
        input_fields = vertices
        new_input_fields_names = ["x", "y"]
    
    input_fields_names = new_input_fields_names

    output_fields   = []
    new_output_fields_names = []
    for field_name in output_fields_names:
        field = sample.get_field(field_name, base_name=base_name)
        if field is None: continue
        output_fields.append(field)
        new_output_fields_names.append(field_name)
    output_fields = np.vstack(output_fields).T
    
    output_fields_names = new_output_fields_names

    # torch tensor conversion
    input_scalars   = torch.tensor(input_scalars, dtype=torch.float32)
    input_fields    = torch.tensor(input_fields, dtype=torch.float32)

    vertices        = torch.tensor(vertices, dtype=torch.float32)
    edge_weight     = torch.tensor(edge_weight, dtype=torch.float32)
    edge_index      = torch.tensor(edge_index, dtype=torch.long)
    
    # Extracting special nodal tags
    nodal_tags = {}
    for k, v in sample.get_nodal_tags(base_name=base_name).items():
        nodal_tags["id_" + k] = torch.tensor(v, dtype=torch.long)

    if None not in output_scalars and None not in output_fields:
        output_scalars  = torch.tensor(output_scalars, dtype=torch.float32)
        output_fields   = torch.tensor(output_fields, dtype=torch.float32)

        data = Data(
            pos = vertices,
            x = input_fields,
            y = output_fields,
            x_scalars = input_scalars.reshape(1, -1),
            y_scalars = output_scalars.reshape(1, -1),
            x_fields_names=input_fields_names,
            y_fields_names=output_fields_names,
            x_scalars_names=input_scalars_names,
            y_scalars_names=output_scalars_names,
            edge_index = edge_index.T,
            edge_weight = edge_weight,
            **faces_dict,
            **nodal_tags,
            sample_id = sample_id
        )
        
        return data

    data = Data(
        pos = vertices,
        x_scalars = input_scalars.reshape(1, -1),
        x = input_fields,
        x_fields_names=input_fields_names,
        y_fields_names=output_fields_names,
        x_scalars_names=input_scalars_names,
        y_scalars_names=output_scalars_names,
        edge_index = edge_index.T,
        edge_weight = edge_weight,
        **faces_dict,
        **nodal_tags,
        sample_id = sample_id
    )

    return data


class Loader():
    """Loader class to load a PLAID Dataset and convert it to Pytorch Geometric"""
    def __init__(self,
                 dataset_name: str=None,
                 cache_dir: str=None): 

        self.dataset_name   = dataset_name
        self.cache_dir      = cache_dir

    def get_dataset_split(self,
                          plaid_dataset: PlaidDataset,
                          problem_definition: ProblemDefinition,
                          task_split: Union[list[str], str]):
        if type(task_split)==list:
            assert all(split in problem_definition._split.keys() for split in task_split), f"task_split {task_split} not in set of split keys {problem_definition._split.keys()}"
            datasets = []
            for split in task_split:
                split_ids = problem_definition.get_split(split)
                dataset = PlaidDataset()
                dataset.set_samples(plaid_dataset.get_samples(ids=split_ids))
                datasets.append(dataset)
                return tuple(datasets)
        assert task_split in problem_definition._split.keys(), f"task_split {task_split} not in set of split keys {problem_definition._split.keys()}"
        ids = problem_definition.get_split(task_split)
        dataset = PlaidDataset()
        dataset.set_samples(plaid_dataset.get_samples(ids=ids))
        return dataset
        

    def load_plaid(self,
                    task_split: Union[List[str], str]=None) -> Union[PlaidDataset, Tuple[PlaidDataset]]:
        hf_dataset = None
        try:
            hf_dataset = load_dataset(self.dataset_name, split="all_samples", cache_dir=self.cache_dir)
        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print(f"Please refer to the documentation (https://huggingface.co/PLAID-datasets) to first download the dataset with the command:")
            print(f"load_dataset('PLAID-datasets/DATASET', split='all_samples', cache_dir='cache_dir')")
            raise e

        plaid_dataset, problem_definition = huggingface_dataset_to_plaid(hf_dataset)

        return problem_definition, self.get_dataset_split(plaid_dataset, problem_definition, task_split)

    def load(self,
             task_split: Union[List[str], str]=None,
             base_name: str=None,
             processes_number: Annotated[int, ">=-1"]=1,
             verbose=False) -> Tuple[ProblemDefinition, List[Data], ...]:
        """
        Load and converts a plaid dataset to torch geometric format
        """
        if processes_number == -1: processes_number = os.cpu_count()
        if verbose: print(f"Number of processes: {processes_number}")
        
        if type(task_split)==list:
            problem_definition, dataset_list = self.load_plaid(task_split=task_split)
            processed_list = []
            for dataset in dataset_list:
                processed_list.append(self.plaid_to_bridge(dataset, problem_definition=problem_definition, base_name=base_name, processes_number=processes_number, verbose=verbose))
            bridged_dataset = tuple(processed_list)
        else:
            problem_definition, dataset = self.load_plaid(task_split=task_split)
            bridged_dataset = [self.plaid_to_bridge(dataset, problem_definition=problem_definition, base_name=base_name, processes_number=processes_number, verbose=verbose)]

        return problem_definition, *bridged_dataset

    def plaid_to_bridge(self,
                        dataset: PlaidDataset,
                        problem_definition: ProblemDefinition,
                        base_name: str=None,
                        processes_number: Annotated[int, ">=-1"]=1,
                        verbose= True) -> List[Data]:
        """
        Converts a Plaid dataset to PytorchGeometric dataset

        Args:
            dataset (plaid.containers.dataset.Dataset): Plaid dataset

        Returns:
            List[Data]: the converted dataset
        """
        if verbose: print("in bridge")
        data_list = []
        sample_ids, samples = list(zip(*list(dataset.get_samples().items())))
        if processes_number==0 or processes_number==1:
            if verbose: iterator = tqdm(zip(samples, sample_ids), total=len(samples))
            else: iterator = zip(samples, sample_ids)
            for sample, sample_id in iterator:
                new_data = sample_to_pyg(sample, sample_id, problem_definition, base_name=base_name)
                data_list.append(new_data)
            return data_list

        with Pool(processes=processes_number) as p:
            if verbose: iterator = tqdm(p.starmap(sample_to_pyg, zip(samples, sample_ids, [problem_definition]*len(samples), [base_name]*len(samples))), total=len(samples))
            else: iterator = p.starmap(sample_to_pyg, zip(samples, sample_ids, [problem_definition]*len(samples)))
            for new_data in iterator:
                data_list.append(new_data)

        return data_list