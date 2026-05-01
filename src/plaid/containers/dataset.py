# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Optional, Sequence, Union, Literal, Any
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator, PrivateAttr
import numpy as np

from ..problem_definition import ProblemDefinition
from .sample import Sample
from ..types.common import NDArrayInt
from ..storage.registry import BackendModule
from ..storage.registry import get_backend, get_backend_class


class Dataset(BaseModel):
    """A lazy-loading dataset that reads samples from disk on demand.

    Args:
        path: Path to the PLAID dataset directory on disk.
        stage: Dataset stage ("training" or "evaluating").
        split: Dataset split key to load from disk (for example, "train" or "eval").
        problem_definition: Problem definition for this dataset.
        indices: Optional array of sample indices to restrict the dataset view.
            Can be "all" to include all samples, a sequence of indices, or None
            to use all samples from the split.
        label: Optional semantic label for the dataset. If not provided, defaults
            to the split name. This can be used to give a custom name to a dataset
            view (e.g., "train" or "eval" for train_eval_split results).

    Attributes:
        path: Path to the dataset directory.
        stage: Stage associated with this dataset instance.
        split: Actual data source split (immutable after construction).
        label: Semantic label for this dataset (can be changed).
        problem_definition: Problem definition for this dataset.
        init_feats: Optional feature subset requested at sample retrieval.
    """
    model_config = ConfigDict(revalidate_instances = 'always', validate_assignment = True, extra='forbid')

    path: Optional[Union[str, Path]] = Field(default=None)
    stage: Optional[Literal["training", "evaluating"]] = Field(default=None)
    split: Optional[Any] = Field(default=None)
    problem_definition: ProblemDefinition =  Field(default_factory=ProblemDefinition)
    indices: NDArrayInt | Literal["all"]  = Field(default="all")

    #ids : NDArrayInt = Field(default_factory=lambda: np.empty(0, dtype=int))
    conv: Any = Field(default=None)
    
    #self._ds = None
    # self._conv = None
#   #      self._ids = []
    #backend_type : str = Field(default="in_memory")
    _backend = PrivateAttr(default_factory=lambda: get_backend("in_memory") )
    _backend_new = PrivateAttr(default_factory=lambda: get_backend_class("in_memory")() )


    def __init__(self, **data):
#        path: Optional[Union[str, Path]] = None,
#         stage: Literal["training", "evaluating"] = "training",
#         split: str | None = None,  # Internal use only, not part of the public API
#         problem_definition: ProblemDefinition | None = None,
#         indices: np.ndarray | Literal["all"] | None = None,
#         label: str | None = None,
#     ):
#         self.path = path
#         self.stage = stage
#         self._split = split  # Immutable data source identifier
#         self.label = label if label is not None else "default"  # Semantic label
#         self.problem_definition = problem_definition
#         self.init_feats = None
#         self._ds = None
#         self._conv = None
#         self._ids = []

        path = data.get("path",None)
        
        
        if path is not None:
            split = data.get("split",None)
            self.load(path= path, split=split)
        else:
            # init for in memory storage
            self._backend_new = get_backend_class("in_memory")()
            

        super().__init__(**data)


    # to set the name, task only once 
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["split"]:
            current_value = getattr(self, name, None)
            if current_value is not None and value is not None:
                raise AttributeError(f"'{name}' is already set and cannot be changed.")
        super().__setattr__(name, value)

    def get_backend_new(self) -> BackendModule:
        return self._backend_new
    
    def load(self, path: Union[str, Path], split: Optional[Any]= None):
        path = Path(path)
        if path.is_file():
            #inputdir = path.parent / f"tmploaddir_{generate_random_ASCII()}"
            import tempfile
            inputdir = Path(tempfile.mkdtemp(prefix="temp_plaid_load"))
       
            try:
                # First : untar file <path> to a directory <inputdir>
                # TODO: avoid using subprocess by using a lib tarfile
                arguments = ["tar", "-xf", path, "-C", inputdir]
                import subprocess

                subprocess.call(arguments)

                # Then : load data from directory <inputdir>
                from plaid.storage import init_from_disk    
                datasetdict, converterdict = init_from_disk(inputdir)
                self._ds = datasetdict[split]
                self._conv = converterdict[split]
                self._ids = np.arange(len(self._ds))
            finally: 
                #shutil.rmtree(inputdir)
                #register deletion at exit
                import atexit
                import shutil
                atexit.register(shutil.rmtree, inputdir)

        elif path.is_dir():
            from plaid.storage import init_from_disk    
            datasetdict, converterdict = init_from_disk(path)
            print(f"{split=}")
            print(list(datasetdict.keys()))
            self._ds = datasetdict[split]
            self._conv = converterdict[split]
            self._ids = np.arange(len(self._ds))
        else:
            raise FileNotFoundError(f"Error! path '{path}' does not exist")


    @classmethod
    def from_training_split(
        cls,
        path: Path,
        pb_def_name: str = "PLAID_benchmark",
    ) -> "Dataset":
        """Create a Dataset from the training split defined in the problem definition.

        Args:
            path: Path to the PLAID dataset directory on disk.
            pb_def_name: Name of the problem definition to load.

        Returns:
            Dataset instance loaded from the training split.
        """
        problem_definition = ProblemDefinition(path=path, name=pb_def_name)
        split, indices = next(iter(problem_definition.training_split.items()))

        # Convert indices to numpy array if needed
        indices_array: np.ndarray | Literal["all"] | None
        if indices == "all":
            indices_array = "all"
        elif indices is not None:
            indices_array = np.array(indices)
        else:
            indices_array = None

        return cls(
            path=path,
            stage="training",
            split=split,
            problem_definition=problem_definition,
            indices=indices_array,
        )

    def __getitem__(self, idx: int):
        """Return a single converted sample from the current dataset view.

        Args:
            idx: Position inside the current ``_ids`` view.

        Returns:
            Sample converted to PLAID format by the split converter.
        """
#        assert self._ds is not None
#        assert self.conv is not None
#        assert self._ids is not None
#        assert self.init_feats is not None, (
#             "self.init_feats not initialized, did you call set_transform_stage(transform_stage) ?"
#         )
        return self.get_backend_new()[idx]
    
        return self._conv.to_plaid(self._ds, self._ids[idx], features=self.init_feats)

    def __len__(self):
        """Return the number of samples currently exposed by this dataset.

        Returns:
            Number of indices currently stored in ``_ids``.
        """
        if isinstance(self.indices, str):
            if self.indices == "all":
                return len(self.get_backend_new())
            else:
                raise RuntimeError(f"'{self.indices}' not a valid value")
            
        return len(self.indices)


    def get_samples(self, ids: Optional[list[int]] = None) -> list[Sample]:
        """Return a list of samples corresponding to the given IDs.

        Args:
            ids: Optional list of sample IDs to retrieve. If None, retrieves all samples in the dataset.

        Returns:
            List of Sample objects corresponding to the specified IDs.
        """
        if ids is None:
            if self.indices == "all":
                return [self[i] for i in range(len(self))]    
            else:
                return [self[i] for i in self.indices]
        else:
            return [self[i] for i in ids]

        
    def get_sample_ids(self):
        if self.indices == "all":
            return range(len(self))
        else:
            return self.indices

