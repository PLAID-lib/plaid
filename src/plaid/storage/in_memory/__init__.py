
from typing import Any, Union, Sequence, Optional, Dict, Iterable, overload
from pathlib import Path 

import numpy as np

from ...containers.sample import Sample

def __find_first_missing(d: Iterable[int]) -> int:
    key = 0  # Or 0, depending on your starting preference
    while key in d:
        key += 1
    return key

class InMemoryBackend():
    name= "in_memory"

    @staticmethod
    def init_from_disk(path: Union[str, Path]) -> Any: 
        raise NotImplementedError("inMemoryBackend does not support inint from disk")
    
    def download_from_hub (self, repo_id: str, local_dir: Union[str, Path]) -> str: 
        raise NotImplementedError("InMemoryBackend download_from_hub not implemented")
        #return 
    
    def init_datasetdict_streaming_from_hub(self, repo_id: str) -> dict[str, Any]:
        raise NotImplementedError("InMemoryBackend init_datasetdict_streaming_from_hub not implemented")
        #return generate_datasetdict_to_disk(self, output_folder)
    
   
    def generate_datasetdict_to_disk(self, output_folder: Union[str, Path]) -> None:
        raise NotImplementedError("InMemoryBackend generate_datasetdict_to_disk not implemented")
        #return generate_datasetdict_to_disk(self, output_folder: Union[str, Path])
    
    def push_local_datasetdict_to_hub(self, repo_id: str, local_dir: Union[str, Path]) -> None:
        raise NotImplementedError("InMemoryBackend push_local_datasetdict_to_hub not implemented")
        #return push_local_datasetdict_to_hub(self, repo_id, local_dir: Union[str, Path])
    
    def configure_dataset_card(self, repo_id: str, infos: dict) -> None:
        raise NotImplementedError("InMemoryBackend configure_dataset_card not implemented")
        #return configure_dataset_card(repo_id, infos)

    def __init__(self):
        self._samples: Dict[int, Sample] = {}

    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, key):
        if isinstance(key, (slice, Sequence)):
            return [self._samples[k] for k in (range(*key.indices(len(self))) if isinstance(key, slice) else key)]
        return self._samples[key]
    
    def add_sample(self, samples:  Union[Sample,Sequence[Sample]], id: Optional[Union[int,Sequence[int]]] = None):
        if isinstance(samples, Sample):

            if id is None:
                id = len(self)
            elif not isinstance(id, int):
                raise TypeError("id must be a int if the sample is of type Sample")
            
            self.set_sample(sample=samples, id=id )
            return id
        elif isinstance(samples, Sequence):
            if id is None:
                id = list(range(len(self), len(self) + len(samples)))
            elif not isinstance(id, Sequence):
                raise TypeError("id must be a sequence if the sample is of type sequence")
            else:
                if len(id) != len(np.unique(id)):
                    raise ValueError("the ids must be differents")

            if len(samples) != len(id):
                raise ValueError("The length of the list of samples to add and the list of IDs are different")
            
            return [self.add_sample(samples = s, id=i) for i,s in zip(id,samples)  ]
        else:
            raise TypeError(f"sample must be a Sample of sequence[Samples], not : {type(samples)}")
        
    @overload
    def set_sample(self, sample: Sample, id: Optional[int] ) -> int: ...

    @overload
    def set_sample(self, sample: Iterable[Sample], id: Iterable[int] ) -> list[int]: ...

    def set_sample(self, sample: Union[Sample, Iterable[Sample]], id: Union[Optional[int],Iterable[int]] ) -> Union[int,list[int]]:
        """Set the samples of the data set, overwriting the existing ones.

        Args:
            samples Sample ): A dictionary of samples to set inside the dataset.

        Raises:
            TypeError: If the 'samples' parameter is not of type dict[int, Sample].
            TypeError: If the 'id' inside a sample is not of type int.
            ValueError: If the 'id' inside a sample is negative (id >= 0 is required).
            TypeError: If the values inside the 'samples' dictionary are not of type Sample.
        """
        if isinstance(sample, Iterable) and not isinstance(sample, Sample):
            if id is None:
                return [self.set_sample(s) for s in sample]
            return [self.set_sample(s,i) for s,i in zip(sample,id)]
                
            

        if not (isinstance(sample, Sample)):
             raise TypeError(
                 f"sample should be of type Sample but is {type(sample)=}"
             )
        
        if int is None:
            id = __find_first_missing(self._samples)
        elif not (isinstance(id, int)):
            raise TypeError(f"id should be of type {int.__class__} but {type(id)=}")
        if id < 0:
            raise ValueError(f"id should be positive (id>=0) but {id=}")

        self._samples[id] = sample

        return id

    def merge_dataset(self, dataset) -> list[int]:
        """Merges samples of another dataset into this one.

        Args:
            dataset (Dataset): The data set to be merged into this one (self).
            in_place (bool, option): If True, modifies the current dataset in place.

        Returns:
            list[int]: ids of added :class:`Samples <plaid.containers.sample.Sample>` from input :class:`Dataset <plaid.containers.dataset.Dataset>`.

        Raises:
            ValueError: If the provided dataset value is not an instance of Dataset
        """
        if dataset is None:
            return

        return [self.add_sample(dataset[i]) for i in range(len(dataset)) ]