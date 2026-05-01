
from typing import Any, Union, Sequence, Optional, Dict
from pathlib import Path 

import numpy as np

from ...containers.sample import Sample

class InMemoryBackend():
    name= "in_memory"

    def init_from_disk(self, path: Union[str, Path]) -> Any: 
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
        
    def set_sample(self, sample: Sample, id: int ) -> None:
        """Set the samples of the data set, overwriting the existing ones.

        Args:
            samples (dict[int,Sample]): A dictionary of samples to set inside the dataset.

        Raises:
            TypeError: If the 'samples' parameter is not of type dict[int, Sample].
            TypeError: If the 'id' inside a sample is not of type int.
            ValueError: If the 'id' inside a sample is negative (id >= 0 is required).
            TypeError: If the values inside the 'samples' dictionary are not of type Sample.
        """
        if not (isinstance(sample, Sample)):
             raise TypeError(
                 f"sample should be of type Sample but is {type(samples)=}"
             )
        
        if not (isinstance(id, int)):
            raise TypeError(f"id should be of type {int.__class__} but {type(id)=}")
        if id < 0:
            raise ValueError(f"id should be positive (id>=0) but {id=}")


        self._samples[id] = sample

