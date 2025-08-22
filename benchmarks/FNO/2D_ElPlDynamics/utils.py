from plaid.containers.dataset import Dataset as Plaid_Dataset
import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalFractureReader(Dataset):
    def __init__(self, data_path, processes_number, step_definition=1, subset=None):
        """Class used for autoregressive methods, the goal is to predict the next timestep give the previous timestep
        the time step of our simulation is very small thus one might want to predict between longer time intervals
        The dataset projects all meshes on a reference rectangular mesh so that all samples in the dataset get the same size
        THe reference mesh is taken rectangular so as to use the data with an FNO or a variant

        Parameters
        ----------
        data_path : str
            path of the plaid dataset
        processes_number : int
            number of processes used to load the plaid dataset, usually taken equal to the number of processes of the job
        step_definition : int, optional
            number of timesteps between the input and the output, by default 20
        subset : list[int], optional
            list containing a subset of indexes to load, it allows to load only a smaller part of the dataset in the , by default None
        """
        self.subset = subset
        self.processes_number = processes_number
        dataset = Plaid_Dataset()
        dataset._load_from_dir_(
            savedir=data_path,
            verbose=True,
            processes_number=processes_number,
            ids=subset,
        )
        self.plaid_dataset = dataset
        self.timedelta = 0.001
        self.step_definition = step_definition

    def get_fields(self, index_config, index_timestep):
        ### Function to get a transfered field for a given config and given timestep.
        sample = self.plaid_dataset.get_samples([index_config])[index_config]
        fields = np.stack(
            (
                sample.get_field(name="U_x", time=index_timestep * self.timedelta),
                sample.get_field(name="U_y", time=index_timestep * self.timedelta),
                sample.get_field(
                    name="EROSION_STATUS", time=index_timestep * self.timedelta
                ),
                sample.get_field(
                    name="Signed_Distance", time=index_timestep * self.timedelta
                ),
            ),
            axis=1,
        )
        return fields

    def __getitem__(self, index):
        ### Function returning a transfered field as an initial step and a final timestep taken step_definition timesteps after
        index_config = index // (40 - self.step_definition)
        index_timestep = index % (40 - self.step_definition)
        if self.subset is not None:
            if index_config not in self.subset:
                raise IndexError(
                    "Sample has not been loaded in this dataset, include it in the subset or "
                )

        input = self.get_fields(index_config, index_timestep)
        output = self.get_fields(index_config, index_timestep + self.step_definition)
        return torch.permute(
            torch.tensor(input).view(150 * 2 + 1, 150 + 1, 4), (2, 0, 1)
        ), torch.permute(torch.tensor(output).view(150 * 2 + 1, 150 + 1, 4), (2, 0, 1))

    def __len__(self):
        return len(self.plaid_dataset) * (40 - self.step_definition)
