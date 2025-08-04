
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from physicsnemo.models.fno.fno import FNO
from utils import TemporalFractureReader


def renormalize(values):
    """
    Function to normalize input of the FNO
    """
    # Computing the smooth mask using signed distance
    values[:, [3]] = torch.sigmoid(values[:, [3]]/0.1)
    # The smooth mask is used intead of xhi
    xhi = values[:, [3]]
    # We remove the EROSION_STATUS from the inputs (channel 2)
    input_values = values[:, [0, 1, 3]]
    # Scaling the displacements with respect to the max displacement in the simulation
    input_values[:, [0, 1]] /= 20
    return input_values, xhi


def train(model, optimizer, dataloader, epochs, device, dtype):
    """Training loop for the models

    Parameters
    ----------
    model : nn.Module
        pytorch model that takes the input fields and the mask and predict the variation of the input
    optimizer : torch.optim.Optimizer
        pytorch optimizer
    dataloader : torch.data.DataLoader
        Dataloader that return the input fields and output fields
    epochs : int
        number of iterations on the dataset
    device : torch.device
        device on which to run computations
    dtype : torch.dtype
        data type of the model
    """
    for epoch in tqdm(range(epochs)):
        for input_field, output_field in dataloader:
            input_values, xhi = renormalize(input_field)
            output_values, _ = renormalize(output_field)
            input_values, output_values, xhi = input_values.to(device=device, dtype=dtype), output_values.to(
                device=device, dtype=dtype), xhi.to(device=device, dtype=dtype)
            # The model predicts the variation rather than the output field
            loss = torch.mean(torch.mean(
                (model(input_values) + input_values-output_values)**2, dim=(1, 2, 3)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def main(model, dataset_path, num_workers, batch_size, epochs, learning_rate, pin_memory, save_file):
    """Function for starting the training and saving a model
    We advise using DDP to scale this function to multi-node multi-gpu to reduce computation time

    Parameters
    ----------
    model : torch.nn.Model
        model to train
    dataset_path : str
        location of the dataset
    num_workers : int
        number of cpus
    batch_size : int
        batch size for computing the loss
    epochs : int
        number of epochs to perform
    learning_rate : float
        learning rate for the gradient descent
    pin_memory : bool
        pin_memory parameter for the dataloader
    save_file : str
        path where to save the trained model
    """

    device = torch.device("cuda")
    dtype = torch.float

    temp_dataset = TemporalFractureReader(
        dataset_path, num_workers, step_definition=1)

    model.to(device=device, dtype=dtype)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ## making sure we are training on the training set only
    sampler = SubsetRandomSampler(range(1000))
    dataloader = DataLoader(temp_dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=pin_memory,sampler=sampler)

    train(model, optimizer, dataloader, epochs, device, dtype)

    torch.save(model.state_dict(), save_file)


if __name__ == "__main__":

    num_workers = 32
    batch_size = 60
    epochs = 100
    learning_rate = 0.0003
    dataset_path = "/path/to/plaid/dataset"
    pin_memory = True
    save_file = "saved_model.pt"

    # Selection of the model either FNO or DAFNO

    model = FNO(
             in_channels=3,
             out_channels=3,
             decoder_layers=1,
             decoder_layer_size=64,
             dimension=2,
             latent_channels=64,
             num_fno_layers=8,
             num_fno_modes=20,
             padding=8,
         )


    main(model, dataset_path, num_workers, batch_size,
         epochs, learning_rate, pin_memory, save_file)
