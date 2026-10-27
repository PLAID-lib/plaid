"""Plaid ParaView Plugin.

This file is intended to be used inside ParaView as a plugin
compatible with ParaView 5.11+
"""

import json
import os
import socket
import time
from typing import Optional
from urllib import error, request

import numpy as np
import vtk

try:
    from paraview.util.vtkAlgorithm import (
        VTKPythonAlgorithmBase,
        smdomain,
        smhint,
        smproperty,
        smproxy,
    )
except ImportError:
    from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase

try:
    from plaid.utils.cgns_json import (
        cgns_tree_from_json_payload,
        cgns_tree_to_json_payload,
    )
    from plaid.utils.cgns_vtk import CGNSTreeToVtk, VtkToCGNSTree
except ImportError:
    # this import are in a try because for some cases the plaid library is not available (client server)
    # inthat case the body of the 2 include are injected into the plugin at run time using the function
    # get_ParaView_plugin_path_one_file
    pass

# this line is to include the import to make the plugin self-contain
# do not modify the next line (see file function get_ParaView_plugin_path_one_file for the use case)
# ##INCLUDE PLACEHOLDER##

## utility funcitons
##//////////////////////////////////////////////////////////

_start_time = time.time()
debug = bool(os.environ.get("PARAVIEW_LOG_PLUGIN_VERBOSITY", False))


def vtk_to_json_payload(data_object, ensure_3D_points=False):
    """Convert a VTK data object to the existing CGNS JSON payload format.

    Args:
        data_object: VTK dataset or multiblock dataset to serialize.
        ensure_3D_points: Whether metadata-free planar VTK input should retain a
            third coordinate array.

    Returns:
        JSON-compatible CGNS tree payload suitable for a PLAID request.
    """
    trees_payload = [
        {
            "time": 0.0,
            "tree": cgns_tree_to_json_payload(
                VtkToCGNSTree(data_object, ensure_3D_points=ensure_3D_points)
            ),
        }
    ]

    return {
        "format": "plaid-sample-json",
        "version": 1,
        "trees": trees_payload,
    }


def print_debug(message: str) -> None:
    """Print a debug message when plugin verbosity is enabled."""
    if debug:
        print(message, time.time() - _start_time)


print_debug("Loading libs")

paraview_plugin_name = "Plaid ParaView Plugin"
paraview_plugin_version = "5.11.1"

plaid_explorer_server_default_host = os.environ.get("PLAID_HOST", "127.0.0.1")
plaid_explorer_server_default_port = int(os.environ.get("PLAID_PORT", "8000"))
plaid_process_server_default_host = os.environ.get("PLAID_PROCESS_HOST", "127.0.0.1")
plaid_process_server_default_port = int(os.environ.get("PLAID_PROCESS_PORT", "8001"))


def find_closest_numpy(arr, target):
    """Find the value in arr that is closest to the target using numpy."""
    # Convert input to array if it isn't one
    arr = np.asarray(arr)
    # Find the index of the minimum absolute difference
    idx = np.abs(arr - target).argmin()
    # Return the value at that index
    return arr[idx]


class PlaidDataSetBase(VTKPythonAlgorithmBase):
    """Base class for Plaid dataset readers and clients, providing common properties and caching logic."""

    def __init__(
        self,
        nInputPorts,
        nOutputPorts,
        inputType="vtkUnstructuredGrid",
        outputType="vtkUnstructuredGrid",
    ):
        # Correctly initialize the underlying VTK C++ layer
        super().__init__(
            nInputPorts=nInputPorts,
            nOutputPorts=nOutputPorts,
            inputType=inputType,
            outputType=outputType,
        )

        self.sample_id: int = 0
        self._selected_split: str = ""
        self._info_cache: Optional[dict] = None
        self._problem_definition_cache: Optional[dict] = None
        self._timestep_values_cache = None
        self._sample_cache: Optional[dict] = None

    def _CleanCache(self):
        self._info_cache = None
        self._problem_definition_cache = None
        self._timestep_values_cache = None
        self._selected_split = ""
        self._sample_cache = None
        self.Modified()

    @smproperty.stringvector(
        name="SelectSplit",
        default_values="",
        panel_visibility="default",
        immediate_update="1",
    )
    @smdomain.xml("""
        <StringListDomain name="array_list">
            <RequiredProperties>
                <Property name="AvailableSplitsInfo" function="AvailableSplitsInfo"  immediate_update="1"/>
            </RequiredProperties>
        </StringListDomain>
    """)
    def SetSelectedSplit(self, value):
        """Set the currently selected data split (e.g., 'training', 'validation', 'test')."""
        if self._selected_split != value:
            self._selected_split = value
            self.Modified()
            if isinstance(self._selected_split, str):
                max_sample_id = self.GetSampleIdRange()[1]
                self.sample_id = max(0, min(self.sample_id, max_sample_id))
                self._sample_cache = None
                self.Modified()

    @smproperty.intvector(
        name="SampleIdRangeInfo",
        information_only="1",
        panel_visibility="default",
        immediate_update="1",
    )
    def GetSampleIdRange(self):
        """Return [min, max] bounds for the SampleId slider."""
        infos = self.GetInfos()
        if infos is None or self._selected_split not in infos["num_samples"]:
            return (0, 0)
        num_samples = int(infos["num_samples"][self._selected_split])
        print_debug(f"GetSampleIdRange {(0, max(0, num_samples - 1))}")
        return (0, max(0, num_samples - 1))

    @smproperty.stringvector(name="AvailableSplitsInfo", information_only="1")
    def GetAvailableSplits(self):
        """Return a list of available data splits (e.g., 'training', 'validation', 'test') for the current dataset."""
        info = self.GetInfos()
        if self._selected_split is None and len(info["num_samples"].keys()):
            self.SetSelectedSplit(list(info["num_samples"])[0])
        print_debug(f"GetAvailableSplits {list(info['num_samples'].keys())}")
        return list(info["num_samples"].keys())

    @smproperty.stringvector(
        name="ReadOnly",
        panel_visibility="default",
        information_only="1",
        repeatable="1",
        number_of_elements_per_command="2",
    )
    def GetSomeTable(self):
        """Return a table of information about the dataset, such as the number of samples in each split."""
        info = self.GetInfos()
        print_debug(
            f"GetSomeTable {['Split Name', 'Nb Samples'] + [[str(k), str(v)] for k, v in info['num_samples'].items()]}"
        )
        return ["Split Name", "Nb Samples"] + [
            [str(k), str(v)] for k, v in info["num_samples"].items()
        ]

    @smproperty.intvector(
        name="SampleId",
        default_values="0",
        panel_visibility="default",
        immediate_update="1",
    )
    @smdomain.xml(
        """<IntRangeDomain name="range" >
                <RequiredProperties>
                    <Property name="SampleIdRangeInfo" function="RangeInfo" immediate_update="1" />
                </RequiredProperties>
           </IntRangeDomain>
        """
    )
    def SetSampleId(self, value):
        """Set the current sample ID to view, with bounds checking against the available sample range."""
        value = int(value)
        max_value = self.GetSampleIdRange()[1]
        value = max(0, min(value, max_value))
        if self.sample_id != value:
            self.sample_id = value
            self.timestep_values_cache = None
            self._sample_cache = None
            self.Modified()

    def RequestInformation(self, request, in_info_vec, out_info_vec):  # noqa: ARG002
        """Provide time step information to ParaView based on the currently selected sample and split."""
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

        print_debug(f" end RequestInformation-----------------------------{time_steps}")
        return 1

    def RequestData(self, request, in_info_vec, out_info_vec):  # noqa: ARG002
        """Fetch the CGNS tree for the currently requested time step and convert it to a VTK object for visualization."""
        out_info = out_info_vec.GetInformationObject(0)
        executive = self.GetExecutive()

        if out_info.Has(executive.UPDATE_TIME_STEP()):
            requested_time = float(out_info.Get(executive.UPDATE_TIME_STEP()))
        else:
            # values = self.GetTimestepValues()
            # requested_time = float(values[0]) if values else 0.0
            requested_time = 0.0

        sample_data = self.GetSampleData()

        if sample_data == "None":
            return 1

        requested_time = find_closest_numpy(
            np.array(list(sample_data.keys())), requested_time
        )
        cgnstree = sample_data[requested_time]

        new_output = CGNSTreeToVtk(cgnstree)
        info = out_info_vec.GetInformationObject(0)

        info.Set(vtk.vtkDataObject.DATA_OBJECT(), new_output)
        return 1

    @smproperty.doublevector(
        name="TimestepValues",
        information_only="1",
    )
    def GetTimestepValues(self):
        """Return a list of available time steps for the currently selected sample and split, with caching."""
        if (
            (self._timestep_values_cache is None)
            and (self._selected_split != "" and self._selected_split is not None)
            and self.sample_id > -1
        ):
            print_debug(f"{self._timestep_values_cache=}")
            print_debug(f"{self._selected_split=}{type(self._selected_split)}")
            print_debug(f"{self.sample_id=}{type(self.sample_id)}")
            sample_data = self.GetSampleData()
            self._timestep_values_cache = [float(t) for t in sample_data.keys()]

        if self._timestep_values_cache is None:
            print_debug(f"GetTimestepValues {[0]}")
            return [0]
        print_debug(f"GetTimestepValues {self._timestep_values_cache}")
        return self._timestep_values_cache


class PlaidClientBase(PlaidDataSetBase):
    """Base class for Plaid clients, providing common properties and methods for interacting with a server."""

    def __init__(
        self,
        nInputPorts,
        nOutputPorts,
        inputType="vtkUnstructuredGrid",
        outputType="vtkUnstructuredGrid",
    ):
        # Correctly initialize the underlying VTK C++ layer
        super().__init__(
            nInputPorts=nInputPorts,
            nOutputPorts=nOutputPorts,
            inputType=inputType,
            outputType=outputType,
        )
        self.host_name: str = plaid_explorer_server_default_host
        self.port_nb: int = plaid_explorer_server_default_port
        self.server_ok = False

    def _CleanCache(self):
        super()._CleanCache()
        self.Modified()

    @smproperty.stringvector(
        name="Host", default_values=plaid_explorer_server_default_host
    )
    def SetHost(self, value):
        """Set the server host address to connect to for fetching dataset information and samples."""
        value = str(value)
        if self.host_name != value:
            self.host_name = value
            self._CleanCache()

    @smproperty.intvector(
        name="PortNB", default_values=str(plaid_explorer_server_default_port)
    )
    def SetPortNB(self, value):
        """Set the server port to connect to for fetching dataset information and samples."""
        value = int(value)
        if self.port_nb != value:
            self.port_nb = value
            self._CleanCache()

    def isServerOK(self):
        """Check if the Plaid server is reachable and responding to health checks."""
        self.server_ok = False
        try:
            with request.urlopen(
                request.Request(
                    url=f"http://{self.host_name}:{self.port_nb}/health",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=2,
            ) as response:
                self.server_ok = (
                    json.loads(response.read().decode("utf-8"))["status"] == "ok"
                )
        except error.HTTPError as exc:
            # Server was reachable, but returned e.g. 400, 404, 500
            print(f"HTTP error: {exc.code} {exc.reason}")
        except error.URLError as exc:
            # DNS failure, connection refused, unreachable host, etc.
            if isinstance(exc.reason, socket.timeout):
                print("Connection timed out")
            elif isinstance(exc.reason, ConnectionRefusedError):
                print("Connection refused")
            else:
                print(f"Connection failed: {exc.reason}")

        except (socket.timeout, TimeoutError):
            # Some timeout conditions may surface directly
            print("Connection timed out")

        self._CleanCache()
        return self.server_ok

    def _request_json(
        self, endpoint: str, payload: Optional[dict[str, object]] = None
    ) -> dict[str, object]:

        data = json.dumps(payload).encode("utf-8") if payload is not None else None

        req = request.Request(
            url=f"http://{self.host_name}:{self.port_nb}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def GetProblemDefinition(self):
        """Fetch the problem definition from the server, with caching."""
        if self._problem_definition_cache is None:
            self._problem_definition_cache = self._request_json("/problem_definition")

        return self._problem_definition_cache

    def GetInfos(self):
        """Fetch general information about the dataset from the server, with caching."""
        if self._info_cache is None:
            self._info_cache = self._request_json("/infos")
        return self._info_cache


print_debug("Loading PlaidExplorer")


@smproxy.source(name="PlaidExplorer", label="Plaid Explorer")
class PlaidExplorer(PlaidClientBase):
    """ParaView source plugin fetching data from Plaid serve endpoints."""

    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1)

        self.timestep_values_cache: list[float] | None = None
        self.useProcess: bool = False
        self.input_features = ""

    @smproperty.stringvector(
        name="Host", default_values=plaid_explorer_server_default_host
    )
    def SetHost(self, value):
        """Set the server host address to connect to for fetching dataset information and samples."""
        return super().SetHost(value)

    @smproperty.intvector(
        name="PortNB", default_values=str(plaid_explorer_server_default_port)
    )
    def SetPortNB(self, value):
        """Set the server port to connect to for fetching dataset information and samples."""
        return super().SetPortNB(value)

    @smproperty.stringvector(
        name="SelectSplit", default_values="", immediate_update="1"
    )
    @smdomain.xml("""
        <StringListDomain name="array_list">
            <RequiredProperties>
                <Property name="AvailableSplitsInfo" function="ArrayList" />
            </RequiredProperties>
        </StringListDomain>
    """)
    def SetSelectedSplit(self, value):
        """Set the currently selected data split (e.g., 'training', 'validation', 'test')."""
        print_debug(f"SetSelectedSplit {value}")
        return super().SetSelectedSplit(value)

    @smproperty.intvector(name="SampleIdRangeInfo", information_only="1")
    def GetSampleIdRange(self):
        """Return [min, max] bounds for the SampleId slider."""
        return super().GetSampleIdRange()

    @smproperty.stringvector(name="AvailableSplitsInfo", information_only="1")
    def GetAvailableSplits(self):
        """Return a list of available data splits (e.g., 'training', 'validation', 'test') for the current dataset."""
        return super().GetAvailableSplits()

    @smproperty.doublevector(
        name="TimestepValues",
        information_only="1",
    )
    def GetTimestepValues(self):
        """Return a list of available time steps for the currently selected sample and split, with caching."""
        return super().GetTimestepValues()

    @smproperty.intvector(name="SampleId", default_values="0", immediate_update="1")
    @smdomain.xml(
        """<IntRangeDomain name="range" >
                <RequiredProperties>
                    <Property name="SampleIdRangeInfo" function="RangeInfo" immediate_update="1" />
                </RequiredProperties>
           </IntRangeDomain>
        """
    )
    def SetSampleId(self, value):
        """Set the current sample ID to view, with bounds checking against the available sample range."""
        print_debug(f"SetSampleId {value}")
        return super().SetSampleId(value)

    @smproperty.stringvector(
        name="ReadOnly",
        panel_visibility="default",
        information_only="1",
        repeatable="1",
        number_of_elements_per_command="2",
    )
    def GetSomeTable(self):
        """Return a table of information about the dataset, such as the number of samples in each split."""
        return super().GetSomeTable()

    @smproperty.xml("""
          <IntVectorProperty name="Process"
                             command="SetProcess"
                             number_of_elements="1"
                             default_values="0"
                            panel_visibility="default">
              <BooleanDomain name="bool"/>
              <Documentation>
                This property indicates if we use the sample or the process endpoint
              </Documentation>
          </IntVectorProperty>""")
    def SetProcess(self, value):
        """Set whether to use the /process endpoint instead of /sample. This is a boolean property."""
        bool_value = str(value).lower() in ["true", "1"]
        if self.useProcess != bool_value:
            self.useProcess = bool_value
            self._sample_cache = None
            self.Modified()

    def GetSampleData(self):
        """Fetch sample data for the currently selected split and sample ID, with caching."""
        if self._sample_cache is None:
            endpoint = "/process" if self.useProcess else "/samples"
            payload = {
                "sample_ids": [self.sample_id],
                "split": self._selected_split,
            }

            if len(self.input_features):
                payload["input_features"] = [json.loads(self.input_features)]

            response = self._request_json(
                endpoint,
                payload,
            )
            sample_payload = response.get("samples", [None])[0].get("trees")
            sample_data = {}

            for entry in sample_payload:
                time_value = float(entry["time"])
                sample_data[time_value] = cgns_tree_from_json_payload(entry["tree"])
            self._sample_cache = sample_data

        return self._sample_cache


@smproxy.filter(name="PlaidProcess", label="Plaid Process")
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataObject"], composite_data_supported=False)
class PlaidProcess(VTKPythonAlgorithmBase):
    """ParaView filter for processing samples with a Plaid serve endpoint."""

    def __init__(self):
        super().__init__(
            nInputPorts=1,
            nOutputPorts=1,
            outputType="vtkUnstructuredGrid",
        )
        self._timestep_values_cache: list[float] | None = None
        self._sample_cache = None
        self.host: str = plaid_process_server_default_host
        self.port_nb: int = plaid_process_server_default_port
        self.process_op: str = ""
        self.ensure_3D_points = False

        # Track the input used to build the cache
        self._input_mtime = None

    @smproperty.stringvector(name="ProcessOperation", default_values="")
    def SetProcessOperation(self, value: str):
        """Set the operation name sent to the processing endpoint.

        Args:
            value: Operation name entered in the ParaView text input.

        Returns:
            None.
        """
        if self.process_op != value:
            self.process_op = value
            self._CleanCache()

    def _CleanCache(self):
        self._timestep_values_cache: list[float] | None = None
        self._sample_cache = None
        self.Modified()

    def _CheckInputChanged(self, input_data):
        if input_data is None:
            return False

        mtime = input_data.GetMTime()

        if self._input_mtime is None:
            self._input_mtime = mtime
            return False

        if mtime != self._input_mtime:
            print_debug(f"Input changed: {self._input_mtime} -> {mtime}")
            self._input_mtime = mtime
            self._InvalidateCache()

            return True

        return False

    def _InvalidateCache(self):
        self._timestep_values_cache = None
        self._sample_cache = None

    def _PropertyChanged(self):
        self._InvalidateCache()
        self.Modified()

    @smproperty.stringvector(
        name="Host", default_values=plaid_process_server_default_host
    )
    def SetHost(self, value):
        """Set the server host address to connect to for fetching dataset information and samples."""
        value = str(value)
        if self.host != value:
            self.host = value
            self._PropertyChanged()

    @smproperty.intvector(
        name="PortNB", default_values=int(plaid_process_server_default_port)
    )
    def SetPortNB(self, value):
        """Set the server port to connect to for processing data."""
        value = int(value)
        if self.port_nb != value:
            self.port_nb = value
            self._PropertyChanged()

    def _request_json(
        self, endpoint: str, payload: Optional[dict[str, object]] = None
    ) -> dict[str, object]:
        """Send a JSON request to the process server.

        Args:
            endpoint: Server endpoint path.
            payload: Optional JSON-serializable request body.

        Returns:
            Decoded JSON response from the server.

        Raises:
            HTTPError: The server rejected the request.
            URLError: The server could not be reached.
            TimeoutError: The request timed out.
        """
        data = json.dumps(payload).encode("utf-8") if payload is not None else None

        req = request.Request(
            url=f"http://{self.host}:{self.port_nb}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            # Server was reachable, but returned e.g. 400, 404, 500
            print(f"HTTP error: {exc.code} {exc.reason}")
            raise
        except error.URLError as exc:
            # DNS failure, connection refused, unreachable host, etc.
            if isinstance(exc.reason, socket.timeout):
                print("Connection timed out")
            elif isinstance(exc.reason, ConnectionRefusedError):
                print("Connection refused")
            else:
                print(f"Connection failed: {exc.reason}")
            raise

        except (socket.timeout, TimeoutError):
            # Some timeout conditions may surface directly
            print("Connection timed out")
            raise

    def RequestData(self, request, in_info_vec, out_info_vec):  # noqa: ARG002
        """Convert the cached CGNS tree for the requested time to VTK."""
        out_info = out_info_vec.GetInformationObject(0)
        executive = self.GetExecutive()
        from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkUnstructuredGrid

        if out_info.Has(executive.UPDATE_TIME_STEP()):
            requested_time = float(out_info.Get(executive.UPDATE_TIME_STEP()))
        else:
            # values = self.GetTimestepValues()
            # requested_time = float(values[0]) if values else 0.0
            requested_time = 0.0

        if self._sample_cache is None:
            input0 = vtkUnstructuredGrid.GetData(in_info_vec[0])
            if input0 is None:
                input0 = vtkPolyData.GetData(in_info_vec[0])

            self.GetSampleData(input0)

        sample_data = self._sample_cache

        if sample_data == "None":
            return 1

        requested_time = find_closest_numpy(
            np.array(list(sample_data.keys())), requested_time
        )
        cgnstree = sample_data[requested_time]

        new_output = CGNSTreeToVtk(cgnstree)
        info = out_info_vec.GetInformationObject(0)

        info.Set(vtk.vtkDataObject.DATA_OBJECT(), new_output)
        return 1

    def RequestInformation(self, request, input_vector, out_info_vec):  # noqa: ARG002
        """Update upstream and fetch sample time-step information."""
        # 1. Reach upstream and force the producer to execute right now
        input_vector[0].GetInformationObject(0)
        upstream_algorithm = self.GetInputConnection(0, 0).GetProducer()
        upstream_algorithm.Update()

        input_data = self.GetInputDataObject(0, 0)

        executive = self.GetExecutive()
        out_info = out_info_vec.GetInformationObject(0)

        self.GetSampleData(input_data)

        if len(self._timestep_values_cache) == 0:
            return 1

        time_steps = self._timestep_values_cache
        out_info.Remove(executive.TIME_STEPS())
        out_info.Remove(executive.TIME_RANGE())

        if len(time_steps) > 1:
            for t in time_steps:
                out_info.Append(executive.TIME_STEPS(), t)
            out_info.Append(executive.TIME_RANGE(), time_steps[0])
            out_info.Append(executive.TIME_RANGE(), time_steps[-1])

        print_debug(f" end RequestInformation-----------------------------{time_steps}")
        return 1

    def GetSampleData(self, input_data):
        """Fetch and cache the processed sample data."""
        self._CheckInputChanged(input_data)

        if self._sample_cache is None:
            endpoint = "/process"

            payload = {
                "operation": self.process_op,
                "sample": vtk_to_json_payload(
                    input_data, ensure_3D_points=self.ensure_3D_points
                ),
            }

            response = self._request_json(
                endpoint,
                payload,
            )
            sample_payload = response.get("samples", [None])[0].get("trees")
            sample_data = {}

            for entry in sample_payload:
                time_value = float(entry["time"])
                sample_data[time_value] = cgns_tree_from_json_payload(entry["tree"])
            self._sample_cache = sample_data
            self._timestep_values_cache = [float(t) for t in sample_data.keys()]

            if self._timestep_values_cache is None:
                print_debug(f"GetTimestepValues {[0]}")
                self._timestep_values_cache = [0]

        return self._sample_cache


class PlaidSingleSampleReaderBase(VTKPythonAlgorithmBase):
    """Read one timestamped Sample without dataset-specific controls."""

    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkDataObject")
        self._filename = ""
        self._sample_cache = None

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    def SetFileName(self, name):
        """Set the input path and invalidate cached trees when it changes.

        Args:
            name: File or directory path on the ParaView server.
        """
        if self._filename != name:
            self._filename = name
            self._sample_cache = None
            self.Modified()

    def GetSampleData(self):
        """Return cached timestamped trees, loading the input on first use."""
        if not self._filename:
            return {}
        if self._sample_cache is None:
            self._sample_cache = self._load_sample_data()
        return self._sample_cache

    def _load_sample_data(self):
        """Return timestamped trees decoded by a concrete reader."""
        raise NotImplementedError

    @smproperty.doublevector(name="TimestepValues", information_only="1")
    def GetTimestepValues(self):
        """Return all sample timestamps in ascending order."""
        return sorted(self.GetSampleData())

    def RequestInformation(self, request, in_info_vec, out_info_vec):  # noqa: ARG002
        """Publish sample timestamps and clear obsolete pipeline time metadata."""
        info = out_info_vec.GetInformationObject(0)
        executive = self.GetExecutive()
        info.Remove(executive.TIME_STEPS())
        info.Remove(executive.TIME_RANGE())
        times = self.GetTimestepValues()
        if times:
            for time_value in times:
                info.Append(executive.TIME_STEPS(), time_value)
            info.Append(executive.TIME_RANGE(), times[0])
            info.Append(executive.TIME_RANGE(), times[-1])
        return 1

    def _get_requested_output(self, info):
        """Return the selected VTK object and time for pipeline information.

        Args:
            info: Output pipeline information, optionally requesting a time.

        Returns:
            Pair of VTK data and selected timestamp, or empty data and None.
        """
        data = self.GetSampleData()
        if not data:
            return vtk.vtkUnstructuredGrid(), None
        times = sorted(data)
        key = self.GetExecutive().UPDATE_TIME_STEP()
        requested = info.Get(key) if info.Has(key) else times[0]
        time_value = find_closest_numpy(times, requested)
        return CGNSTreeToVtk(data[time_value]), float(time_value)

    def RequestDataObject(self, request, in_info_vec, out_info_vec):  # noqa: ARG002
        """Allocate the actual output type for structured or multiblock data."""
        info = out_info_vec.GetInformationObject(0)
        data, _ = self._get_requested_output(info)
        current = info.Get(vtk.vtkDataObject.DATA_OBJECT())
        if current is None or current.GetClassName() != data.GetClassName():
            info.Set(vtk.vtkDataObject.DATA_OBJECT(), data.NewInstance())
        return 1

    def RequestData(self, request, in_info_vec, out_info_vec):  # noqa: ARG002
        """Populate VTK output for the nearest available requested timestamp."""
        info = out_info_vec.GetInformationObject(0)
        data, time_value = self._get_requested_output(info)
        output = info.Get(vtk.vtkDataObject.DATA_OBJECT())
        output.ShallowCopy(data)
        output.GetInformation().Remove(vtk.vtkDataObject.DATA_TIME_STEP())
        if time_value is not None:
            output.GetInformation().Set(vtk.vtkDataObject.DATA_TIME_STEP(), time_value)
        return 1


@smproxy.reader(
    name="PlaidSampleJSONReader",
    label="Plaid Sample JSON Reader",
    extensions="json",
    file_description="PLAID Sample JSON",
)
class PlaidSampleJSONReader(PlaidSingleSampleReaderBase):
    """Read Sample JSON using only helpers bundled in the standalone plugin."""

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="json", file_description="PLAID Sample JSON")
    def SetFileName(self, name):
        """Set the JSON file path, clearing cached data if it changes.

        Args:
            name: Path to a serialized PLAID Sample JSON file.
        """
        return super().SetFileName(name)

    @smproperty.doublevector(name="TimestepValues", information_only="1")
    def GetTimestepValues(self):
        """Return sorted sample timestamps for the ParaView time controls."""
        return super().GetTimestepValues()

    def _load_sample_data(self):
        """Return decoded trees after validating the Sample JSON envelope."""
        with open(self._filename, encoding="utf-8") as stream:
            payload = json.load(stream)
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "plaid-sample-json"
        ):
            raise ValueError("Unsupported Sample JSON format")
        if payload.get("version") != 1:
            raise ValueError("Unsupported Sample JSON version")
        if not isinstance(payload.get("trees"), list):
            raise ValueError("Sample JSON payload must contain a list in 'trees'")
        data = {}
        for entry in payload["trees"]:
            if (
                not isinstance(entry, dict)
                or "time" not in entry
                or "tree" not in entry
            ):
                raise ValueError(
                    "Each Sample JSON tree entry must contain 'time' and 'tree'"
                )
            if not isinstance(entry["time"], (int, float)):
                raise ValueError("Sample JSON time entries must be numeric")
            data[float(entry["time"])] = cgns_tree_from_json_payload(entry["tree"])
        return data


try:
    # try to load the reader if plaid is locally available
    from plaid import Sample
    from plaid.storage.reader import init_from_disk, load_infos_from_disk

    @smproxy.reader(
        name="PlaidSampleReader",
        label="Plaid Sample Directory Reader",
        file_description="PLAID Sample Directory",
        is_directory="True",
        filename_patterns="*",
    )
    class PlaidSampleReader(PlaidSingleSampleReaderBase):
        """Read a Sample directory when PLAID is installed on the server."""

        @smproperty.stringvector(name="FileName")
        @smdomain.filelist()
        @smhint.filechooser()
        def SetFileName(self, name):
            """Set the Sample directory path, clearing cached data if it changes.

            Args:
                name: Directory previously written by Sample.save_to_dir.
            """
            return super().SetFileName(name)

        @smproperty.doublevector(name="TimestepValues", information_only="1")
        def GetTimestepValues(self):
            """Return sorted sample timestamps for the ParaView time controls."""
            return super().GetTimestepValues()

        def _load_sample_data(self):
            """Return timestamped trees loaded with Sample.load_from_dir."""
            return Sample.load_from_dir(self._filename).data

    # <SourceProxy name="PXDMFReader"  label="PXDMFReader"
    # class="vtkPXDMFReader"
    # base_proxygroup="internal_sources"
    # base_proxyname="PXDMFDocumentBaseStructure">
    # >

    @smproxy.reader(
        name="PlaidDatasetReader",
        label="Plaid Dataset Reader",
        file_description="Directory ",
        is_directory="True",
        filename_patterns="*",
    )
    class PlaidDataSetReader(PlaidDataSetBase):
        """ParaView reader plugin for reading Plaid datasets from disk."""

        def __init__(self):
            super().__init__(
                nInputPorts=0, nOutputPorts=1, outputType="vtkUnstructuredGrid"
            )
            self._filename: Optional[str] = ""
            self.datasetdict_cache = None
            self.converterdict_cache = None

        def _CleanCache(self):
            super()._CleanCache()
            self.datasetdict_cache = None
            self.converterdict_cache = None
            self._info_cache = None
            self._problem_definition_cache = None
            self._selected_split = ""
            self.Modified()

        @smproperty.stringvector(name="FileName")
        @smdomain.filelist()
        @smhint.filechooser(extensions="ext", file_description="ext" + " files")
        def SetFileName(self, name):
            """Specify filename for the file to read."""
            if self._filename != name:
                self._filename = name
                self._CleanCache()

        @smproperty.stringvector(name="SelectSplit", default_values="")
        @smdomain.xml("""
            <StringListDomain name="array_list">
                <RequiredProperties>
                    <Property name="AvailableSplitsInfo" function="ArrayList" />
                </RequiredProperties>
            </StringListDomain>
        """)
        def SetSelectedSplit(self, value):
            """Set the currently selected data split (e.g., 'training', 'validation', 'test')."""
            return super().SetSelectedSplit(value)

        @smproperty.intvector(name="SampleIdRangeInfo", information_only="1")
        def GetSampleIdRange(self):
            """Return [min, max] bounds for the SampleId slider."""
            return super().GetSampleIdRange()

        @smproperty.stringvector(name="AvailableSplitsInfo", information_only="1")
        def GetAvailableSplits(self):
            """Return a list of available data splits (e.g., 'training', 'validation', 'test') for the current dataset."""
            return super().GetAvailableSplits()

        @smproperty.intvector(name="SampleId", default_values="0", immediate_update="1")
        @smdomain.xml(
            """<IntRangeDomain name="range" >
                <RequiredProperties>
                    <Property name="SampleIdRangeInfo" function="RangeInfo" immediate_update="1" />
                </RequiredProperties>
           </IntRangeDomain>
        """
        )
        def SetSampleId(self, value):
            """Set the current sample ID to view, with bounds checking against the available sample range."""
            return super().SetSampleId(value)

        @smproperty.stringvector(
            name="ReadOnly",
            panel_visibility="default",
            information_only="1",
            repeatable="1",
            number_of_elements_per_command="2",
        )
        def GetSomeTable(self):
            """Return a table of information about the dataset, such as the number of samples in each split."""
            return super().GetSomeTable()

        @smproperty.doublevector(
            name="TimestepValues",
            information_only="1",
        )
        def GetTimestepValues(self):
            """Return a list of available time steps for the currently selected sample and split, with caching."""
            return super().GetTimestepValues()

        def GetInfos(self):
            """Fetch general information about the dataset from disk, with caching."""
            if self._info_cache is not None:
                return self._info_cache

            if (self._info_cache is None) and (
                self._filename is not None and self._filename != "None"
            ):
                self._info_cache = load_infos_from_disk(self._filename).model_dump()
                self.Modified()
            else:
                return {"num_samples": {}}

            return self._info_cache

        def GetSampleData(self) -> dict[float, list]:
            """Fetch sample data for the currently selected split and sample ID from disk, with caching."""
            if self._sample_cache is None:
                if self.datasetdict_cache is None:
                    self.datasetdict_cache, self.converterdict_cache = init_from_disk(
                        self._filename
                    )

                self._sample_cache = self.converterdict_cache[
                    self._selected_split
                ].to_plaid(self.datasetdict_cache[self._selected_split], self.sample_id)
            return self._sample_cache.data

    print_debug("Reader PlaidSampleReader Loaded")
except ImportError as exc:
    print_debug(
        f"PLAID directory readers not loaded because optional PLAID imports failed: {exc}"
    )

print_debug("Plaid ParaView Plugin Loaded")
