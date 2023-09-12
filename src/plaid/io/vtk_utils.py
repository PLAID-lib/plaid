# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import json
import logging
import os

import numpy as np
import vtk
from BasicTools.Bridges.vtkBridge import MeshToVtk, VtkToMesh

from plaid.utils.base import BTMesh

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s',
    level=logging.INFO)

# %% Functions


def write_vtk(fname: str, vtk_mesh):
    """Write VTK (Visualization Toolkit) mesh data to a file with the specified filename and appropriate file extension based on the type of mesh provided.

    Args:
      fname (str) The name of the output VTK file, including the appropriate file extension (e.g., '.vtk', '.vtp', '.vtu', '.vti').
      vtk_mesh: The VTK mesh data to be written to the file. The type of mesh should match the selected writer and file extension.
    """
    # TODO: select writer and extension (.vtk/.vtp/.vtu) depending on type of mesh, see: https://stackoverflow.com/a/59319614
    # If for images (.vti)
    # vtkXMLPolyDataWriter for unstructured
    # can also use the vtkXMLUnstructuredGridWriter (.vtu)
    # Or use vtkExtractSurface to convert to PolyData and then save to .vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(vtk_mesh)
    writer.SetFileName(fname)
    writer.Write()


def read_vtk(fname: str):
    """Reads VTK files with various extensions and returns the data contained in them.

    Args:
      fname (str) The file path to the VTK file you want to read.

    Returns:
        vtk.vtkDataObject: A VTK data object containing the data from the VTK file.

    Raises:
        ValueError: If the file extension is not recognized.
    """
    if fname.endswith(".vtk"):
        reader = vtk.vtkPolyDataReader()
    elif fname.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fname.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif fname.endswith(".vts"):
        reader = vtk.vtkXMLStructuredGridReader()
    elif fname.endswith(".vti"):
        reader = vtk.vtkXMLImageDataReader()
    else:
        raise ValueError(f"unknown extension {fname.split('.')[-1]}")
    reader.SetFileName(fname)
    reader.Update()
    return reader.GetOutput()


def save_temporal_vtk(savedir: str, name: str,
                      meshes: BTMesh, timestamps: np.ndarray):
    """Save temporal VTK mesh data as a series of files and create a metadata file.

    Args:
        savedir (str): The directory where the VTK files and metadata file will be saved.
        name (str): The base name for the VTK files and metadata file.
        meshes (BTMesh): A collection of VTK meshes indexed by timestamps.
        timestamps (np.ndarray): An array of timestamps corresponding to each VTK mesh in the series.
    """
    # save temporal .vtk -> https://stackoverflow.com/a/69889294

    # ---# Save each mesh
    files_list = []
    for timestamp in timestamps:
        vtk_mesh = MeshToVtk(meshes[timestamp], TagsAsFields=True)
        # TODO: select writer and extension (.vtk/.vtp/.vtu/.vti) depending on
        # type of mesh
        fname = os.path.join(savedir, f'{name}_{timestamp}.vtk')
        write_vtk(fname, vtk_mesh)

        files_list.append({'name': fname, 'time': timestamp})

    # ---# Save serie
    json_dict = {"file-series-version": "1.0", "files": files_list}
    fname = os.path.join(savedir, f'{name}.vtu.series')
    with open(fname, 'w') as f:
        json.dump(json_dict, f)
        json.write('\n')


def load_temporal_vtk(fname: str):
    """Load temporal VTK data from a JSON file, where each entry contains a VTK file name and a timestamp.

    Args:
        fname (str): The file path to the JSON file containing VTK file load information.

    Returns:
        Tuple[List[Mesh], List[float]]: A tuple containing a list of Mesh objects and a list of corresponding timestamps.
    """
    with open(fname, 'r') as f:
        data = json.load(f)
        assert ('files' in data)
        timestamps = [float(file['time']) in data['files']]
        meshes = []
        for file in data['files']:
            vtk_mesh = read_vtk(file['name'])
            meshes.append(VtkToMesh(vtk_mesh))
        sorting_inds = np.argsort(timestamps)
        timestamps = timestamps[sorting_inds]
        meshes = meshes[sorting_inds]
        return meshes, timestamps
