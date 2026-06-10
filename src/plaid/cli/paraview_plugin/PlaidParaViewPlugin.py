"""Plaid ParaView Plugin.

This file is intended to be used inside ParaView as a plugin
compatible with ParaView 5.11+
"""

import json
import os
import time
from typing import Optional
from urllib import request

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
    from plaid.utils.cgns_json import cgns_tree_from_json_payload
    from plaid.utils.cgns_vtk import CGNSTreeToVtk
except ImportError:
    # this import are in a try because for some cases the plaid library is not available (client server)
    # inthat case the body of the 2 include are injected into the plugin at run time using the function
    # get_ParaView_plugin_path_one_file
    pass

# this line is to inlcude the import to make the plugin selfcontain
# do not modify the next line (see file function get_ParaView_plugin_path_one_file for the use case)
# ##INCLUDE PLACEHOLDER##

## utility funcitons
##//////////////////////////////////////////////////////////

_start_time = time.time()
debug = bool(os.environ.get("PARAVIEW_LOG_PLUGIN_VERBOSITY", True))


def print_debug(message: str) -> None:
    """Print a debug message when plugin verbosity is enabled."""
    if debug:
        print(message, time.time() - _start_time)


print_debug("Loading libs")

paraview_plugin_name = "Plaid ParaView Plugin"
paraview_plugin_version = "5.11.1"


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
        self.host: str = "127.0.0.1"
        self.port: int = 8000

    def _CleanCache(self):
        super()._CleanCache()
        self.Modified()

    @smproperty.stringvector(name="Host", default_values="127.0.0.1")
    def SetHost(self, value):
        """Set the server host address to connect to for fetching dataset information and samples."""
        value = str(value)
        if self.host != value:
            self.host = value
            self._CleanCache()

    @smproperty.intvector(
        name="Port", default_values=os.environ.get("PLAID_PORT", "8000")
    )
    def SetPort(self, value):
        """Set the server port to connect to for fetching dataset information and samples."""
        value = int(value)
        if self.port != value:
            self.port = value
            self._CleanCache()

    def _request_json(
        self, endpoint: str, payload: Optional[dict[str, object]] = None
    ) -> dict[str, object]:

        data = json.dumps(payload).encode("utf-8") if payload is not None else None

        req = request.Request(
            url=f"http://{self.host}:{self.port}{endpoint}",
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


print_debug("Loading MaestroExplorer")


@smproxy.source(name="MaestroExplorer", label="Maestro Explorer")
class MaestroExplorer(PlaidClientBase):
    """ParaView source plugin fetching data from Maestro serve endpoints."""

    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1)

        self.timestep_values_cache: list[float] | None = None
        self.usePredict: bool = False
        self.input_features = ""

    @smproperty.stringvector(name="Host", default_values="127.0.0.1")
    def SetHost(self, value):
        """Set the server host address to connect to for fetching dataset information and samples."""
        return super().SetHost(value)

    @smproperty.intvector(
        name="Port", default_values=os.environ.get("PLAID_PORT", "8000")
    )
    def SetPort(self, value):
        """Set the server port to connect to for fetching dataset information and samples."""
        return super().SetPort(value)

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

    # """
    #             <RequiredProperties>
    #                 <Property name="SampleIdRangeInfo" function="RangeInfo"  immediate_update="1"/>
    #                 <Property name="SelectSplit" function="GetSelectSplit"  immediate_update="1"/>
    #             </RequiredProperties>

    # """
    # @smproperty.xml("""
    #         <IntVectorProperty name="SampleId"
    #                             command="SetSampleId"
    #                             number_of_elements="1"
    #                             default_values="0"
    #                      immediate_update="1">
    #             <IntRangeDomain name="range">

    #             </IntRangeDomain>
    #             <Hints>
    #             <Widget type="slider" />
    #             </Hints>
    #         </IntVectorProperty>""")

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
        """Set whether to use the /predict endpoint instead of /sample. This is a boolean property."""
        bool_value = str(value).lower() in ["true", "1"]
        if self.usePredict != bool_value:
            self.usePredict = bool_value
            self._sample_cache = None
            self.Modified()

    def GetSampleData(self):
        """Fetch sample data for the currently selected split and sample ID, with caching."""
        if self._sample_cache is None:
            endpoint = "/predict" if self.usePredict else "/samples"
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


# paraview.servermanager.LoadPlugin("/home/fbw/repos/Safran/plaid/src/plaid/cli/paraview_plugin/PlaidParaViewPlugin.py")
try:
    # try to load the reader if plaid is locally available
    from plaid.storage.reader import init_from_disk, load_infos_from_disk

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
        f"PlaidDatasetReader not loaded because optional plaid.storage.reader import failed: {exc}"
    )

print_debug("Plaid ParaView Plugin Loaded")
