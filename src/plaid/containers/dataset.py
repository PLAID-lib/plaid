# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Optional, Sequence, Union, Literal, Any
from pathlib import Path
import copy
from packaging.version import Version

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator, PrivateAttr
import numpy as np

from plaid.constants import AUTHORIZED_INFO_KEYS
from ..version import __version__
from ..problem_definition import ProblemDefinition
from .sample import Sample
from ..types.common import NDArrayInt
from ..storage.registry import BackendModule
from ..storage.registry import get_backend


class Dataset(BaseModel):
    """A lazy-loading dataset that reads samples from disk on demand.

    """
    model_config = ConfigDict(revalidate_instances = 'always', validate_assignment = True, extra='forbid')

    path: Optional[Union[str, Path]] = Field(default=None, description="Path to the PLAID dataset directory on disk.")
    stage: Optional[Literal["training", "evaluating"]] = Field(default=None, description="Dataset stage ('training' or 'evaluating')")
    split: Optional[str] = Field(default=None, description="Actual data source split (immutable after construction).")
    problem_definition: ProblemDefinition =  Field(default_factory=ProblemDefinition, description="Problem definition for this dataset.")
    indices: NDArrayInt | Literal["all"]  = Field(default="all", description="""Optional array of sample indices to restrict the dataset view.
            Can be "all" to include all samples, a sequence of indices, or None
            to use all samples from the split.""")
    infos : dict[str, dict[str, str]] = Field(default_factory=dict)
    _conv: Any = PrivateAttr(default=None)
    _ids : Any = PrivateAttr(default=None)
    _backend : BackendModule = PrivateAttr(default_factory=lambda: get_backend("in_memory")())
    label : str = Field(default="")


    # to set the name, task only once 
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["split", "path" ]:
            current_value = getattr(self, name, None)
            if current_value is not None and value is not None and current_value != value:
                raise AttributeError(f"'{name}' is already set and cannot be changed.")
            if current_value == value:
                return 
        super().__setattr__(name, value)

    def get_backend(self) -> BackendModule:
        return self._backend
    
    @staticmethod
    def from_path(
        path: str | Path,
        *,
        split: Any = None,
        stage: Literal["training", "evaluating"] | None = None,
        problem_definition: ProblemDefinition | None = None,
        indices: NDArrayInt | Literal["all"] = "all",
    ) -> "Dataset":
        dataset = Dataset(
            path=path,
            split=split,
            stage=stage,
            problem_definition=problem_definition or ProblemDefinition(),
            indices=indices,
        )
        dataset.load(path=path, split=split)
        return dataset

    def set_infos(self, infos: dict[str, dict[str, str]], warn: bool = True) -> None:
        """Set information to the :class:`Dataset <plaid.containers.dataset.Dataset>`, overwriting the existing one.

        Args:
            infos (dict[str,dict[str,str]]): Information to associate with this data set (Dataset).
            warn (bool, optional): If True, warns when replacing existing infos. Defaults to True.

        Raises:
            KeyError: Invalid category key format in provided infos.
            KeyError: Invalid info key format in provided infos.

        Example:
            .. code-block:: python

                from plaid import Dataset
                dataset = Dataset()
                infos = {"legal":{"owner":"CompX", "license":"li_X"}}
                dataset.set_infos(infos)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}}
        """
        for cat_key in infos.keys():  # Format checking on "infos"
            if cat_key != "plaid":
                if cat_key not in AUTHORIZED_INFO_KEYS:
                    raise KeyError(
                        f"{cat_key=} not among authorized keys. Maybe you want to try among these keys {list(AUTHORIZED_INFO_KEYS.keys())}"
                    )
                for info_key in infos[cat_key].keys():
                    if info_key not in AUTHORIZED_INFO_KEYS[cat_key]:
                        raise KeyError(
                            f"{info_key=} not among authorized keys. Maybe you want to try among these keys {AUTHORIZED_INFO_KEYS[cat_key]}"
                        )

        # Check if there are any non-plaid infos being replaced
        has_user_infos = any(key != "plaid" for key in self.infos.keys())
        if has_user_infos and warn:
            logger.warning("infos not empty, replacing it anyway")
        self.infos = copy.deepcopy(infos)

        if "plaid" not in self.infos:
            self.infos["plaid"] = {}
        if "version" not in self.infos["plaid"]:
            self.infos["plaid"]["version"] = Version(__version__)


    # load data from disk if path and split are given 
    def model_post_init(self, __context):
        if self.path is not None and self.split is not None:
            self.load()

    def load(self, path: Optional[Union[str, Path]] = None, split: Optional[Any]= None):

        if path is None:
            path = self.path

        if split is None:
            split = self.split
        if split is None:
            raise RuntimeError("Need a split name to be loaded")
        
        self.split = split

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
                self.indices = np.arange(len(self._ds))
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
            self.indices = np.arange(len(self._ds))
        else:
            raise FileNotFoundError(f"Error! path '{path}' does not exist")


    @classmethod
    def from_train_split(
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
        problem_definition = ProblemDefinition.from_path(path=path, name=pb_def_name)
        split, indices = next(iter(problem_definition.train_split.items()))

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
            #problem_definition=problem_definition,
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
        return self._backend[idx]
    
        return self._conv.to_plaid(self._ds, self._ids[idx], features=self.init_feats)

    def __len__(self):
        """Return the number of samples currently exposed by this dataset.

        Returns:
            Number of indices currently stored in ``_ids``.
        """
        if isinstance(self.indices, str):
            if self.indices == "all":
                return len(self._backend)
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

