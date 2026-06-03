#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
# exec(open("C:/Users/User/paraview/ParaViewPlugin.py","r").read())
# This file is intended to be used inside ParaView as a plugin
# compatible with ParaView 5.7+

import json
import os
import time
from urllib import request
from typing import Union
import logging

import numpy as np
from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproperty, smproxy, smhint, smdomain
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyVertex,
    vtkUnstructuredGrid,
)


_ELEMENT_TYPE_TO_VTK: dict[str, tuple[int, int]] = {
    "BAR_2": (3, 2),
    "TRI_3": (5, 3),
    "QUAD_4": (9, 4),
    "TETRA_4": (10, 4),
    "PYRA_5": (14, 5),
    "PENTA_6": (13, 6),
    "HEXA_8": (12, 8),
}

_start_time = time.time()
debug = bool(os.environ.get("PARAVIEW_LOG_PLUGIN_VERBOSITY", True))


def print_debug(message: str) -> None:
    """Print a debug message when plugin verbosity is enabled."""
    if debug:
        print(message, time.time() - _start_time)


print_debug("Loading libs")


def find_closest_numpy(arr, target):
    # Convert input to array if it isn't one
    arr = np.asarray(arr)
    # Find the index of the minimum absolute difference
    idx = np.abs(arr - target).argmin()
    # Return the value at that index
    return arr[idx]


# 1. Define your custom base class
class PlaidClientBase(VTKPythonAlgorithmBase):
    def __init__(
        self,
        nInputPorts,
        nOutputPorts,
        inputType="vtkUnstructuredGrid",
        outputType="vtkUnstructuredGrid",
    ):
        # Correctly initialize the underlying VTK C++ layer
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=nInputPorts,
            nOutputPorts=nOutputPorts,
            inputType=inputType,
            outputType=outputType,
        )
        self.host: str = "127.0.0.1"
        self.port: int = 8000

    def SharedLogMethod(self, request_type):
        """A utility method available to all child filters."""
        logging.info(f"Executing step: {request_type} in {self.__class__.__name__}")

    @smproperty.stringvector(name="Host", default_values="127.0.0.1")
    def SetHost(self, value):
        value = str(value)
        if self.host != value:
            self.host = value
            self.timestep_values_cache = None
            self.Modified()

    @smproperty.intvector(
        name="Port", default_values=os.environ.get("PLAID_PORT", "8000")
    )
    def SetPort(self, value):
        value = int(value)
        if self.port != value:
            self.port = value
            self.timestep_values_cache = None
            self.Modified()

    def _request_json(
        self, endpoint: str, payload: dict[str, object]
    ) -> dict[str, object]:

        method = "GET"
        print(f"{endpoint}")
        print(json.dumps(payload))
        req = request.Request(
            url=f"http://{self.host}:{self.port}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _find_feature_by_suffix(
        payload: dict[str, object], suffix: str
    ) -> object | None:
        for key, value in payload.items():
            if key.endswith(suffix):
                return value
        return None

    def _extract_points_and_fields(
        self,
        sample_payload: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = self._find_feature_by_suffix(sample_payload, "/GridCoordinates/CoordinateX")
        y = self._find_feature_by_suffix(sample_payload, "/GridCoordinates/CoordinateY")
        z = self._find_feature_by_suffix(sample_payload, "/GridCoordinates/CoordinateZ")

        if x is None or y is None:
            raise ValueError("Could not find coordinate arrays in sample payload")

        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        z_arr = (
            np.asarray(z, dtype=np.float64).reshape(-1)
            if z is not None
            else np.zeros_like(x_arr)
        )
        n_points = len(x_arr)
        points = np.column_stack((x_arr, y_arr, z_arr))

        fields: dict[str, np.ndarray] = {}
        for key, value in sample_payload.items():
            # print(key)
            if "/VertexFields/" not in key:
                continue
            arr = np.asarray(value)
            name = key.split("/")[-1]
            if arr.ndim == 1 and arr.size >= n_points:
                fields[name] = arr.reshape(-1)[:n_points]
            elif arr.ndim == 2 and arr.shape[0] >= n_points:
                fields[name] = arr[:n_points]

        return points, fields

    def _build_unstructured_grid(
        self,
        sample_payload: dict[str, object],
        output: vtkUnstructuredGrid,
    ) -> None:
        points_np, fields = self._extract_points_and_fields(sample_payload)

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points_np, deep=True))
        output.SetPoints(vtk_points)

        n_points = points_np.shape[0]

        element_entries: list[tuple[int, np.ndarray]] = []
        for key, value in sample_payload.items():
            if not key.endswith("/ElementConnectivity"):
                continue
            element_node = key.split("/")[-2]
            element_type_name = element_node.removeprefix("Elements_").upper()
            element_spec = _ELEMENT_TYPE_TO_VTK.get(element_type_name)
            if element_spec is None:
                continue

            vtk_cell_type, n_vertices_per_element = element_spec
            connectivity = np.asarray(value, dtype=np.int64)
            if connectivity.size == 0:
                continue
            if connectivity.ndim == 1:
                if connectivity.size % n_vertices_per_element != 0:
                    continue
                connectivity = connectivity.reshape(-1, n_vertices_per_element)
            elif connectivity.ndim != 2:
                continue
            elif connectivity.shape[1] != n_vertices_per_element:
                continue

            if connectivity.min() >= 1:
                connectivity = connectivity - 1

            if connectivity.min() < 0 or connectivity.max() >= n_points:
                continue

            for cell_nodes in connectivity:
                element_entries.append(
                    (vtk_cell_type, np.asarray(cell_nodes, dtype=np.int64))
                )

        if element_entries:
            output.Allocate(len(element_entries))
            for vtk_cell_type, cell_nodes in element_entries:
                output.InsertNextCell(
                    int(vtk_cell_type),
                    int(cell_nodes.size),
                    cell_nodes.astype(np.int64),
                )
        else:
            cells = vtkCellArray()
            poly_vertex = vtkPolyVertex()
            poly_vertex.GetPointIds().SetNumberOfIds(n_points)
            for i in range(n_points):
                poly_vertex.GetPointIds().SetId(i, i)
            cells.InsertNextCell(poly_vertex)
            output.SetCells(poly_vertex.GetCellType(), cells)

        for name, array in fields.items():
            vtk_array = numpy_to_vtk(np.asarray(array), deep=True)
            vtk_array.SetName(name)
            output.GetPointData().AddArray(vtk_array)


@smproxy.source(name="PlaidExplorer", label="Plaid Dataset Explorer")
class PlaidDatasetExplorer(PlaidClientBase):
    """ParaView source plugin fetching data from Maestro serve endpoints."""

    def __init__(self):
        super().__init__(
            nInputPorts=0,
            nOutputPorts=1,
            inputType=None,
            outputType="vtkUnstructuredGrid",
        )

        self.sample_id: int = 0
        self.timestep_values_cache: list[float] | None = None
        self.input_features = ""
        self.filename_or_url = "/home/fbw/repos/Safran/Datasets/Tensile2d/"
        self._available_splits = {}
        self._selected_split = ""

    @smproperty.stringvector(
        name="SelectPlaidDataset",
        label="Dataset",
        default_values="/home/fbw/repos/Safran/Datasets/Tensile2d/",
    )
    @smdomain.filelist()
    @smhint.xml("<Property show_directory_only='1' />")
    def SetExternalFilePath(self, path: Union[str, None]):
        if path is None:
            path = ""
        if self.filename_or_url != path:
            self.filename_or_url = path
            self._available_splits = {}
            self._selected_split = None
            self.Modified()

    # Step 2: Create an Information Property to leak the server list to the GUI client
    @smproperty.stringvector(name="AvailableSplitsInfo", information_only="1")
    def GetAvailableSplits(self):
        if len(self.filename_or_url) == 0:
            return []
        # ParaView expects information arrays to be returned as a vtkStringArray or list
        ts_response = self._request_json(
            "/splits",
            {
                "sample_ids": [self.sample_id],
                "dataset": self.filename_or_url,
                "split": self._selected_split,
            },
        )
        self._available_splits = ts_response.get("splits", {})

        return list(self._available_splits.keys())

    # Step 3: Create the user-facing Dropdown that dynamically copies the server list

    # 1. Define the user-facing dropdown property
    @smproperty.stringvector(name="SelectSplit", default_values="")
    @smdomain.xml("""
        <StringListDomain name="array_list">
            <RequiredProperties>
                <Property name="AvailableSplitsInfo" function="ArrayList" />
            </RequiredProperties>
        </StringListDomain>
    """)
    def SetSelectedSplit(self, value):
        if self._selected_split != value:
            self._selected_split = value
            self.Modified()

    @smproperty.intvector(name="SampleIdRangeInfo", information_only="1")
    def GetSampleIdRange(self):
        """Return [min, max] bounds for the SampleId slider."""
        split_max = self._available_splits.get(self._selected_split, 0) - 1
        try:
            split_max = int(split_max)
        except (TypeError, ValueError):
            split_max = 0
        return [0, max(0, split_max)]

    @smproperty.xml("""
          <IntVectorProperty name="SampleId"
                             command="SetSampleId"
                             number_of_elements="1"
                             default_values="0">
            <IntRangeDomain name="range">
              <RequiredProperties>
                <Property name="SampleIdRangeInfo" function="RangeInfo" />
              </RequiredProperties>
            </IntRangeDomain>
            <Hints>
              <Widget type="slider" />
            </Hints>
          </IntVectorProperty>""")
    def SetSampleId(self, value):
        value = int(value)
        max_value = self.GetSampleIdRange()[1]
        value = max(0, min(value, max_value))
        if self.sample_id != value:
            self.sample_id = value
            self.timestep_values_cache = None
            self.Modified()

    # @smproperty.stringvector(name="InputFeatures", default_values="")
    # @smhint.xml(r"<Widget type='multi_line'/>")

    @smproperty.xml("""
          <StringVectorProperty name="InputFeatures"
                                command="SetInputFeatures"
                                number_of_elements="1"
                                default_values="">
              <Hints>
                  <Widget type="multi_line" />
              </Hints>
          </StringVectorProperty>

          <PropertyGroup label="Feature Settings">
              <Property name="InputFeatures" />
          </PropertyGroup>
    """)
    def SetInputFeatures(self, value):
        if value is None:
            value = ""
        value = str(value)
        if self.input_features != value:
            self.input_features = value
            self.Modified()

    @smproperty.doublevector(
        name="TimestepValues",
        information_only="1",
        si_class="vtkSITimeStepsProperty",
    )
    def GetTimestepValues(self):
        if self.filename_or_url in ["", None]:
            return [0.0]

        if self._selected_split in ["", None]:
            return [0.0]
        print(f"{self.filename_or_url=}, {self._selected_split=}")
        if self.timestep_values_cache is None:
            # Optional warmup/readability check against /sample
            # self._request_json("/sample", {"sample_ids": [self.sample_id]})

            ts_response = self._request_json(
                "/timesteps",
                {
                    "sample_ids": [self.sample_id],
                    "dataset": self.filename_or_url,
                    "split": self._selected_split,
                },
            )
            entries = ts_response.get("time_steps", [])
            if not entries:
                self.timestep_values_cache = [0.0]
            else:
                first = entries[0]
                times = first.get("times", [0.0]) if isinstance(first, dict) else [0.0]
                self.timestep_values_cache = [float(t) for t in times]

        return self.timestep_values_cache

    def RequestInformation(self, request, in_info_vec, out_info_vec):
        executive = self.GetExecutive()
        out_info = out_info_vec.GetInformationObject(0)

        time_steps = self.GetTimestepValues()
        if len(time_steps) == 0:
            return 1

        out_info.Remove(executive.TIME_STEPS())
        out_info.Remove(executive.TIME_RANGE())

        if len(time_steps) > 1:
            for t in time_steps:
                out_info.Append(executive.TIME_STEPS(), t)
            out_info.Append(executive.TIME_RANGE(), time_steps[0])
            out_info.Append(executive.TIME_RANGE(), time_steps[-1])

        return 1

    def RequestData(self, request, in_info_vec, out_info_vec):
        out_info = out_info_vec.GetInformationObject(0)
        executive = self.GetExecutive()

        if out_info.Has(executive.UPDATE_TIME_STEP()):
            requested_time = float(out_info.Get(executive.UPDATE_TIME_STEP()))
        else:
            values = self.GetTimestepValues()
            requested_time = float(values[0]) if values else 0.0

        requested_time = find_closest_numpy(self.GetTimestepValues(), requested_time)
        endpoint = "/predict_step" if self.usePredict else "/samples_step"
        # print(endpoint)

        # "angle_in":40
        payload = {
            "sample_ids": [self.sample_id],
            "time": requested_time,
            "dataset": self.filename_or_url,
            "split": self._selected_split,
        }
        if len(self.input_features):
            #  {          <- this is added automaticaly
            #  toto = 5             the = is converted to : and add the "" around
            # 'tata' : 6            the ' are converted to "
            # "titi" : 3.5
            #
            #  }            <- this is added automaticaly

            def ensureEncluse(string, start, end):
                string = string.strip()
                if not string.startswith(start):
                    string = start + string
                if not string.endswith(end):
                    string = string + end
                return string

            treated_input_features = self.input_features.replace("'", '"').strip()
            treated_input_features = self.input_features.replace("=", ":").strip()
            clean_treated_input_features = []
            for line in treated_input_features.splitlines():
                k, v = line.split(":")
                k = ensureEncluse(k, '"', '"')
                clean_treated_input_features.append(k + ":" + v)

            treated_input_features = ",".join(clean_treated_input_features)
            treated_input_features = ensureEncluse(treated_input_features, "{", "}")
            # print(f"{treated_input_features=}")

            payload["input_features"] = [json.loads(treated_input_features)]

        step_response = self._request_json(
            endpoint,
            payload,
        )
        sample_payload = step_response.get("samples", [None])[0]
        if sample_payload is None:
            raise ValueError(
                "/sample_step response does not contain a valid sample payload"
            )

        output = vtkUnstructuredGrid.GetData(out_info)
        self._build_unstructured_grid(sample_payload, output)

        # 2. Get the field data object
        field_data = output.GetFieldData()

        for key, value in sample_payload.items():
            if key.startswith("Global/") and not key.endswith("_times"):
                scalar_array = vtkDoubleArray()
                # scalar_array.SetName(key.split("/")[-1])
                scalar_array.SetName(key)
                scalar_array.SetNumberOfComponents(len(value))
                for v in value:
                    scalar_array.InsertNextValue(v)
                field_data.AddArray(scalar_array)
        return 1


@smproxy.source(name="MaestroExplorer", label="Maestro Explorer")
class MaestroExplorer(PlaidClientBase):
    """ParaView source plugin fetching data from Maestro serve endpoints."""

    def __init__(self):
        super().__init__(
            nInputPorts=0,
            nOutputPorts=1,
            inputType=None,
            outputType="vtkUnstructuredGrid",
        )

        self.sample_id: int = 0
        self.timestep_values_cache: list[float] | None = None
        self.usePredict: bool = False
        self.input_features = ""
        self.experiment = "/home/fbw/repos/Safran/Datasets/Tensile2d/"
        self._available_splits = {}
        self._selected_split = ""

    @smproperty.stringvector(
        name="SelectMaestroExperiment",
        label="Experiment",
        default_values="tensile2d_transolver",
    )
    # @smdomain.filelist()
    # @smhint.xml("<Property show_directory_only='1' />")
    def SetExternalFilePath(self, path: Union[str, None]):
        if path is None:
            path = ""
        if self.experiment != path:
            self.experiment = path
            self._available_splits = {}
            self._selected_split = None
            self.Modified()

    # Step 2: Create an Information Property to leak the server list to the GUI client
    @smproperty.stringvector(name="AvailableSplitsInfo", information_only="1")
    def GetAvailableSplits(self):
        if len(self.experiment) == 0:
            return []
        # ParaView expects information arrays to be returned as a vtkStringArray or list
        ts_response = self._request_json(
            "/splits",
            {
                "sample_ids": [self.sample_id],
                "dataset": self.experiment,
                "split": self._selected_split,
            },
        )
        self._available_splits = ts_response.get("splits", {})

        return list(self._available_splits.keys())

    # Step 3: Create the user-facing Dropdown that dynamically copies the server list

    # 1. Define the user-facing dropdown property
    @smproperty.stringvector(name="SelectSplit", default_values="")
    @smdomain.xml("""
        <StringListDomain name="array_list">
            <RequiredProperties>
                <Property name="AvailableSplitsInfo" function="ArrayList" />
            </RequiredProperties>
        </StringListDomain>
    """)
    def SetSelectedSplit(self, value):
        if self._selected_split != value:
            self._selected_split = value
            self.Modified()

    @smproperty.intvector(name="SampleIdRangeInfo", information_only="1")
    def GetSampleIdRange(self):
        """Return [min, max] bounds for the SampleId slider."""
        split_max = self._available_splits.get(self._selected_split, 0) - 1
        try:
            split_max = int(split_max)
        except (TypeError, ValueError):
            split_max = 0
        return [0, max(0, split_max)]

    @smproperty.xml("""
          <IntVectorProperty name="SampleId"
                             command="SetSampleId"
                             number_of_elements="1"
                             default_values="0">
            <IntRangeDomain name="range">
              <RequiredProperties>
                <Property name="SampleIdRangeInfo" function="RangeInfo" />
              </RequiredProperties>
            </IntRangeDomain>
            <Hints>
              <Widget type="slider" />
            </Hints>
          </IntVectorProperty>""")
    def SetSampleId(self, value):
        value = int(value)
        max_value = self.GetSampleIdRange()[1]
        value = max(0, min(value, max_value))
        if self.sample_id != value:
            self.sample_id = value
            self.timestep_values_cache = None
            self.Modified()

    @smproperty.xml("""
          <IntVectorProperty name="Predict"
                             command="SetPredict"
                             number_of_elements="1"
                             default_values="0"
                            panel_visibility="default">
              <BooleanDomain name="bool"/>
              <Documentation>
                This property indicates if we use the sample or the predict endpoint
              </Documentation>
          </IntVectorProperty>""")
    def SetPredict(self, value):
        bool_value = str(value).lower() in ["true", "1"]
        # print(value, bool_value)
        if self.usePredict != bool_value:
            self.usePredict = bool_value
            self.Modified()

    @smproperty.xml("""
          <StringVectorProperty name="InputFeatures"
                                command="SetInputFeatures"
                                number_of_elements="1"
                                default_values="">
              <Hints>
                  <Widget type="multi_line" />
              </Hints>
          </StringVectorProperty>

          <PropertyGroup label="Feature Settings">
              <Property name="InputFeatures" />
          </PropertyGroup>
    """)
    def SetInputFeatures(self, value):
        if value is None:
            value = ""
        value = str(value)
        if self.input_features != value:
            self.input_features = value
            self.Modified()

    @smproperty.doublevector(
        name="TimestepValues",
        information_only="1",
        si_class="vtkSITimeStepsProperty",
    )
    def GetTimestepValues(self):
        if self.experiment in ["", None]:
            return [0.0]

        if self._selected_split in ["", None]:
            return [0.0]
        print(f"{self.experiment=}, {self._selected_split=}")
        if self.timestep_values_cache is None:
            # Optional warmup/readability check against /sample
            # self._request_json("/sample", {"sample_ids": [self.sample_id]})

            ts_response = self._request_json(
                "/timesteps",
                {
                    "sample_ids": [self.sample_id],
                    "dataset": self.experiment,
                    "split": self._selected_split,
                },
            )
            entries = ts_response.get("time_steps", [])
            if not entries:
                self.timestep_values_cache = [0.0]
            else:
                first = entries[0]
                times = first.get("times", [0.0]) if isinstance(first, dict) else [0.0]
                self.timestep_values_cache = [float(t) for t in times]

        return self.timestep_values_cache

    def RequestInformation(self, request, in_info_vec, out_info_vec):
        executive = self.GetExecutive()
        out_info = out_info_vec.GetInformationObject(0)

        time_steps = self.GetTimestepValues()
        if len(time_steps) == 0:
            return 1

        out_info.Remove(executive.TIME_STEPS())
        out_info.Remove(executive.TIME_RANGE())

        if len(time_steps) > 1:
            for t in time_steps:
                out_info.Append(executive.TIME_STEPS(), t)
            out_info.Append(executive.TIME_RANGE(), time_steps[0])
            out_info.Append(executive.TIME_RANGE(), time_steps[-1])

        return 1

    def RequestData(self, request, in_info_vec, out_info_vec):
        out_info = out_info_vec.GetInformationObject(0)
        executive = self.GetExecutive()

        if out_info.Has(executive.UPDATE_TIME_STEP()):
            requested_time = float(out_info.Get(executive.UPDATE_TIME_STEP()))
        else:
            values = self.GetTimestepValues()
            requested_time = float(values[0]) if values else 0.0

        requested_time = find_closest_numpy(self.GetTimestepValues(), requested_time)
        endpoint = "/predict_step" if self.usePredict else "/samples_step"
        # print(endpoint)

        # "angle_in":40
        payload = {
            "sample_ids": [self.sample_id],
            "time": requested_time,
            "dataset": self.experiment,
            "split": self._selected_split,
        }
        if len(self.input_features):
            #  {          <- this is added automaticaly
            #  toto = 5             the = is converted to : and add the "" around
            # 'tata' : 6            the ' are converted to "
            # "titi" : 3.5
            #
            #  }            <- this is added automaticaly

            def ensureEncluse(string, start, end):
                string = string.strip()
                if not string.startswith(start):
                    string = start + string
                if not string.endswith(end):
                    string = string + end
                return string

            treated_input_features = self.input_features.replace("'", '"').strip()
            treated_input_features = self.input_features.replace("=", ":").strip()
            clean_treated_input_features = []
            for line in treated_input_features.splitlines():
                k, v = line.split(":")
                k = ensureEncluse(k, '"', '"')
                clean_treated_input_features.append(k + ":" + v)

            treated_input_features = ",".join(clean_treated_input_features)
            treated_input_features = ensureEncluse(treated_input_features, "{", "}")
            # print(f"{treated_input_features=}")

            payload["input_features"] = [json.loads(treated_input_features)]

        step_response = self._request_json(
            endpoint,
            payload,
        )
        sample_payload = step_response.get("samples", [None])[0]
        if sample_payload is None:
            raise ValueError(
                "/sample_step response does not contain a valid sample payload"
            )

        output = vtkUnstructuredGrid.GetData(out_info)
        self._build_unstructured_grid(sample_payload, output)

        # 2. Get the field data object
        field_data = output.GetFieldData()

        for key, value in sample_payload.items():
            if key.startswith("Global/") and not key.endswith("_times"):
                scalar_array = vtkDoubleArray()
                # scalar_array.SetName(key.split("/")[-1])
                scalar_array.SetName(key)
                scalar_array.SetNumberOfComponents(len(value))
                for v in value:
                    scalar_array.InsertNextValue(v)
                field_data.AddArray(scalar_array)
        return 1
