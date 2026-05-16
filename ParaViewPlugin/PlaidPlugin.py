# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#

# this files is intended to be used inside paraview as a plugin
# compatible with paraview 5.7+
import os
import time
import locale


_startTime = time.time()
debug = bool(os.environ.get("PARAVIEW_LOG_PLUGIN_VERBOSITY", False))

if debug:
    def PrintDebug(mes):
        import time
        print(mes, time.time() - _startTime)
else:
    def PrintDebug(mes):
        pass

try:
    import numpy as np

    from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint
    from paraview.util.vtkAlgorithm import VTKPythonAlgorithmBase
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

    PrintDebug("Loading libs")
    from Muscat.Bridges.vtkBridge import SetOutputMuscat
    PrintDebug("Loading")

    paraview_plugin_name = "Plaid ParaView Plugin"
    paraview_plugin_version = "5.11.1"

    @smproxy.reader(name="PlaidSampleReader", label="Paid Sample Reader", extensions="pickle", file_description="pickle ")
    class PlaidSampleReader(VTKPythonAlgorithmBase):
        def __init__(self):
            VTKPythonAlgorithmBase.__init__(self, nInputPorts=0, nOutputPorts=1, outputType='vtkUnstructuredGrid')
            self._filename: Optional[str] = None
            self.metadata = None
            self.cache = None
        @smproperty.stringvector(name="FileName")
        @smdomain.filelist()
        @smhint.filechooser(extensions="pickle", file_description="pickle files")
        def SetFileName(self, name):
            """Specify filename for the file to read."""
            if self._filename != name:
                self._filename = name
                self.Modified()
                if name is not None:
                    self.GetTimestepValues()

        @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
        def GetTimestepValues(self):
            if self._filename is None or self._filename == "None":
                return None
            with open(self._filename, "rb") as f:
                self.metadata  = pickle.load(f)

            return self.metadata


        def RequestInformation(self, request, inInfoVec, outInfoVec):
            executive = self.GetExecutive()
            outInfo = outInfoVec.GetInformationObject(0)
            outInfo.Remove(executive.TIME_STEPS())
            outInfo.Remove(executive.TIME_RANGE())

            timeSteps = self.GetTimestepValues()
            if timeSteps is not None:
                for t in timeSteps:
                    outInfo.Append(executive.TIME_STEPS(), t)
                outInfo.Append(executive.TIME_RANGE(), timeSteps[0])
                outInfo.Append(executive.TIME_RANGE(), timeSteps[-1])
            return 1

        def RequestData(self, request, inInfoVec, outInfoVec):
            if self._filename is None:
                return 0
            #Read pickle files
            import pickle
            if self.cache = None:
                with open(self._filename, "rb") as f:
                    # drom timevalues
                    pickle.load(f)
                    self.cache = pickle.load(f)

            outInfo = outInfoVec.GetInformationObject(0)
            executive = self.GetExecutive()
            if outInfo.Has(executive.UPDATE_TIME_STEP()):
                time = outInfo.Get(executive.UPDATE_TIME_STEP())
            else:
                time = 0
            from Muscat.Bridges.CGNSBridge import CGNSToMesh
            mesh = CGNSToMesh(self.cache.get_zone(time=time), partitionedMesh=False)
            SetOutputMuscat(request, inInfoVec, outInfoVec, mesh, tagsAsFields=True)
            return 1


    PrintDebug("Plaid ParaView Plugin Loaded")
except Exception as ex:
    print("Error loading Muscat ParaView Plugin")
    print("Muscat in the PYTHONPATH ??? ")
    if debug:
        raise


