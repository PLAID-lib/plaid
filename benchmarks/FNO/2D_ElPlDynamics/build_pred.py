from Muscat.Bridges.CGNSBridge import CGNSToMesh
import numpy as np
from Muscat.MeshTools.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.Fields.FEField import FEField
from Muscat.FE.Spaces.FESpaces import LagrangeSpaceGeo
from Muscat.FE.DofNumbering import ComputeDofNumbering
from Muscat.Bridges.CGNSBridge import CGNSToMesh
import Muscat.MeshContainers.ElementsDescription as ED
from Muscat.MeshTools.ConstantRectilinearMeshTools import CreateConstantRectilinearMesh
from Muscat.MeshTools.MeshTetrahedrization import Tetrahedrization

from scipy.sparse import coo_matrix
import copy
import torch
from physicsnemo.models.fno.fno import FNO
import torch.nn as nn
from Muscat.MeshTools.MeshTools import ComputeSignedDistance
from plaid import Dataset as Plaid_Dataset
import pickle


dataset_path = # path to update to location where rectilinear dataset is created


def renormalize(values):
    """
    Function to normalize input of the FNO
    """
    input_values = copy.deepcopy(values)
    input_values[:, [2]] = torch.sigmoid(input_values[:, [2]]/0.1)
    xhi = input_values[:, [2]]
    input_values[:, [0, 1]] /= 20
    return input_values, xhi


def denormalize(values):
    """
    Inverse function of renormalise except the mask that we do not need
    """
    input_values = copy.deepcopy(values)
    xhi = input_values[:, [2]]
    input_values[:, [0, 1]] *= 20
    return input_values, xhi


def preproccess_sample(sample):
    """Function to project the sample to a rectilinear grid using Muscat
    This function only return the initial timestep as it is the only information needed
    in the autoregressive method (when testing we simply apply recusively the model)

    Parameters
    ----------
    sample : plaid.Sample
        plaid sample to be projected to the rectilinear grid

    Returns
    -------
    input_fields: torch.Tensor
        2D fields (shape: (1,3,301,151)) ready to be used as a 2D image
    xhi: torch.Tensor
        2D mask (shape: (1,1,301,151)) ready to be used as a 2D image
    """
    old_mesh = CGNSToMesh(sample.get_mesh(time=0))
    # Switching to 2D mesh instead of 3D shell, need C order for cython
    old_mesh.nodes = (old_mesh.nodes[:, [0, 1]]).copy(order='C')
    size = 150
    ref_mesh = Tetrahedrization(CreateConstantRectilinearMesh(
        [size*2+1, size+1], [0, 0], [100/size, 100/size]))

    indexes = np.zeros(ref_mesh.GetNumberOfNodes(), int)
    values = np.zeros((ref_mesh.GetNumberOfNodes(), 15))
    for connect in ref_mesh.GetElementsOfType(ED.Triangle_3).connectivity:
        values[connect, indexes[connect]] = 1
        indexes[connect] += 1
    average = 1/indexes
    data, i, j = np.zeros(0), np.zeros(0).astype(int), np.zeros(0).astype(int)
    for elem_index, connect in enumerate(ref_mesh.GetElementsOfType(ED.Triangle_3).connectivity):
        data = np.concatenate((data, average[connect]), axis=0)
        i = np.concatenate((i, connect), axis=0)
        j = np.concatenate((j, [elem_index]*3), axis=0)

    # This operator is needed if one want to project the EROSION_STATUS element field to the nodes
    operator_elem_to_node = coo_matrix((data, (i, j)))

    # Compute Field Transfer operator for node fields
    space = LagrangeSpaceGeo
    numbering = ComputeDofNumbering(old_mesh, space, fromConnectivity=True)
    displacement_field = FEField("FakeField", old_mesh, space, numbering)
    op, _, _ = GetFieldTransferOp(
        displacement_field, ref_mesh.nodes, method="Interp/Clamp")

    triangle_centers = np.mean(
        ref_mesh.nodes[ref_mesh.GetElementsOfType(ED.Triangle_3).connectivity], axis=-2)
    _, _, entities = GetFieldTransferOp(
        displacement_field, triangle_centers, method="Interp/Clamp")
    data = np.ones_like(entities.squeeze())
    i = old_mesh.GetElementsOfType(ED.Triangle_3).GetNumberOfElements()
    op_elem = coo_matrix(
        (data, (np.arange(entities.shape[0]), entities.squeeze())))

    # getting input values for the first timestep
    ux = op@sample.get_field(name="U_x", zone_name="Zone",
                             base_name="Base_2_2", time=0)
    uy = op@sample.get_field(name="U_y", zone_name="Zone",
                             base_name="Base_2_2", time=0)
    signed_distance = ComputeSignedDistance(
        copy.deepcopy(old_mesh), ref_mesh.nodes)
    fields = np.stack((ux, uy, signed_distance), axis=1)
    fields_pt = torch.permute(torch.tensor(
        fields).view(150*2+1, 150+1, 3), (2, 0, 1))
    renormalised_fields, xhi = renormalize(fields_pt.unsqueeze(0))
    return renormalised_fields, xhi


def postprocess_sample(prediction, sample):
    """Function to switch back from rectilinear mesh to the original mesh

    Parameters
    ----------
    prediction : torch.Tensor
        prediction of the neural network that needs to be projected back to the original mesh
    sample : plaid.Sample
        Plaid sample containing the mesh of the orginal geometry

    Returns
    -------
    ux: np.array
        displacements on the x axis
    uy: np.array
        displacements on the y axis
    erosion_node_field: np.array
        erosion field projected on the nodes
    """
    old_mesh = CGNSToMesh(sample.get_mesh(time=0))
    # Switching to 2D mesh instead of 3D shell, need C order for cython
    old_mesh.nodes = (old_mesh.nodes[:, [0, 1]]).copy(order='C')
    size = 150
    ref_mesh = Tetrahedrization(CreateConstantRectilinearMesh(
        [size*2+1, size+1], [0, 0], [100/size, 100/size]))

    # Compute Field Transfer operator for node fields
    space = LagrangeSpaceGeo
    numbering = ComputeDofNumbering(ref_mesh, space, fromConnectivity=True)
    displacement_field = FEField("FakeField", ref_mesh, space, numbering)
    op, _, _ = GetFieldTransferOp(
        displacement_field, old_mesh.nodes, method="Interp/Clamp")
    prediction, xhi = denormalize(prediction)
    # getting input values for the first timestep by projecting back
    ux = op@(prediction[0, 0].cpu().numpy().reshape(-1))
    uy = op@(prediction[0, 1].cpu().numpy().reshape(-1))
    erosion_node_field = op@(prediction[0, 2].cpu().numpy().reshape(-1))
    return ux, uy, erosion_node_field


def compute_predictions(dataset_path, ids, model, save_name, device, dtype):
    """Function to compute the predictions from a list of ids in the dataset
    ideally we compute the prediction on unseen samples

    Parameters
    ----------
    dataset_path : str
        path of the plaid dataset
    ids : iterable
        iterable of ids where to compute the prediction
    model : torch.nn.model
        Pytorch model trained on the dataset, we assume that it was trained for predicting variation between t and t+1, not directly t+1
    save_name : str
        String where to save the prediction results
    device : torch.device
        device where to run the computation (needs to be the same than the model)
    dtype : torch.dtype
        dtype of the model (needs to be the same than the model)
    """

    processes_number = 32

    dataset_size = Plaid_Dataset()._load_number_of_samples_(dataset_path)
    dataset = Plaid_Dataset()
    dataset._load_from_dir_(savedir=dataset_path, verbose=True,
                            processes_number=processes_number, ids=list(ids))

    predictions = []
    # For all meshes in the test set
    for i in ids:
        sample = dataset[i]
        print(sample.get_field_names())
        print(sample)
        UXs = []
        UYs = []
        EROSIONS = []

        # Making auto regression to predict the outcome on the test tensile
        with torch.inference_mode():
            input_model, xhi = preproccess_sample(sample)
            input_model = input_model.to(device=device, dtype=dtype)
            out = input_model
            xhi = xhi.to(device=device, dtype=dtype)
            for _ in range(40):
                out = (model(out) + out)
                out[:, [2]] = torch.clamp(out[:, [2]], 0, 1)
                ux_predicted, uy_predicted, erosions_predicted = postprocess_sample(
                    out, sample)
                UXs.append(ux_predicted)
                UYs.append(uy_predicted)
                EROSIONS.append(erosions_predicted)
        # Registering the results
        predictions.append({
            "U_x": np.stack([np.zeros_like(UXs[0])]+UXs),
            "U_y": np.stack([np.zeros_like(UYs[0])]+UYs),
            "EROSION_STATUS": np.stack([np.ones_like(EROSIONS[0])]+EROSIONS)
        })
    with open(save_name, 'wb') as file:
        pickle.dump(predictions, file)



if __name__ == "__main__":

    # Ids where to test the prediction
    # Make sure it does not contain the training set
    ids = range(900, 1000)
    save_file = "saved_model.pt"

    device = torch.device("cuda")
    dtype = torch.float

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

    model.load_state_dict(torch.load(save_file))
    model.to(device=device, dtype=dtype)

    save_name = 'predictions.pkl'
    compute_predictions(dataset_path, ids, model, save_name, device, dtype)
