"""Utilities to launch ParaView with the PLAID plugin configured."""

import os
import subprocess
import tempfile
from pathlib import Path

paraview_exec = "paraview"


def get_ParaView_plugin_path():
    """Returns the path to the ParaView plugin directory."""
    return Path(__file__).parent


def get_ParaView_plugin_path_one_file():
    """Returns the path to a temporary directory containing the plugin as a single file.

    all the helper module are included in the plugin file to make it self contained and
    avoid the need of copying multiple file in the temporary directory.
    """
    plugin_path = Path(__file__).parent / "PlaidParaViewPlugin.py"
    with open(plugin_path, "r") as f:
        plugin_content = f.read()

    import plaid.utils.cgns_json as cgns_json

    with open(Path(cgns_json.__file__), "r") as f:
        sample_json_content = f.read()

    import plaid.utils.cgns_vtk as cgns_vtk

    with open(Path(cgns_vtk.__file__), "r") as f:
        cgns_vtk_content = f.read()

    plugin_content = plugin_content.replace(
        "# ##INCLUDE PLACEHOLDER##", sample_json_content + cgns_vtk_content
    )
    plugin_content = plugin_content.replace("from __future__ import annotations", "")

    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, "PlaidParaViewPlugin.py")

    # Write full plugin to the temporary file
    with open(file_path, "w") as f:
        f.write(plugin_content)
    return tmpdir


def convert_wsl_to_win(wsl_path: str) -> str:
    r"""Converts a WSL path (e.g., /mnt/c/Users) to Windows (C:\\Users)."""
    result = subprocess.run(
        ["wslpath", "-w", wsl_path], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def run_paraview_with_plugin():
    """Launches ParaView with environment variables set to load the plugin."""
    my_env = os.environ.copy()

    my_env["PV_PLUGIN_PATH"] = str(get_ParaView_plugin_path_one_file())
    my_env["PARAVIEW_LOG_PLUGIN_VERBOSITY"] = "ON"

    current_pv_path = os.environ.get("PARAVIEW_EXEC", paraview_exec)
    if current_pv_path.startswith("/mnt/") and os:
        # we are in a wsl and the paraview exec is windows, we need to convert the wsl path
        # my_env["PV_PLUGIN_PATH"] = convert_wsl_to_win(str(get_ParaView_plugin_path()))
        my_env["WSLENV"] = "PV_PLUGIN_PATH/p:PARAVIEW_LOG_PLUGIN_VERBOSITY/p"

    process = subprocess.Popen([current_pv_path], env=my_env)
    return process
