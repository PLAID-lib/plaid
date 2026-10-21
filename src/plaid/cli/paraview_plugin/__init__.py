"""Utilities to launch ParaView with the PLAID plugin configured."""

import atexit
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

paraview_exec = "paraview"
paraview_install_dir = ""


def get_ParaView_plugin_path():
    """Returns the path to the ParaView plugin directory."""
    return Path(__file__).parent


def get_ParaView_plugin_path_one_file(path: Path | str | None = None) -> str:
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

    import plaid.utils.json_codec as json_codec

    with open(Path(json_codec.__file__), "r") as f:
        json_codec_content = f.read()

    import plaid.utils.cgns_vtk as cgns_vtk

    with open(Path(cgns_vtk.__file__), "r") as f:
        cgns_vtk_content = f.read()

    plugin_content = plugin_content.replace(
        "# ##INCLUDE PLACEHOLDER##",
        json_codec_content + sample_json_content + cgns_vtk_content,
    )
    plugin_content = plugin_content.replace("from __future__ import annotations", "")

    plugin_content = plugin_content.replace(
        "from .json_codec import decode_leaf_value, encode_leaf_value", ""
    )

    if path is not None:
        tmpdir = Path(path)
    else:
        tmpdir = tempfile.mkdtemp()

    directory = Path(tmpdir)
    file_path = os.path.join(directory, "PlaidParaViewPlugin.py")

    # Write full plugin to the temporary file
    with open(file_path, "w") as f:
        f.write(plugin_content)
    return str(tmpdir)


def convert_wsl_to_win(wsl_path: str) -> str:
    r"""Converts a WSL path to Windows."""
    result = subprocess.run(
        ["wslpath", "-w", wsl_path], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def run_paraview_with_plugin_client_server(
    port: int = 8000, options: Optional[dict[str, str]] = None
) -> subprocess.Popen:
    """Launch ParaView with the plugin and a generated explorer script.

    Args:
        port: Port of the PLAID data API.
        options: Optional source settings, including ``tree_name``.

    Returns:
        The ParaView process.
    """
    my_env = os.environ.copy()

    my_env["PV_PLUGIN_PATH"] = str(get_ParaView_plugin_path_one_file())
    my_env["PARAVIEW_LOG_PLUGIN_VERBOSITY"] = "ON"
    my_env["PLAID_PORT"] = str(port)

    current_pv_path = os.environ.get("PARAVIEW_EXEC", paraview_exec)
    if current_pv_path.startswith("/mnt/"):
        my_env["WSLENV"] = (
            my_env.get("WSLENV", "") + ":" if my_env.get("WSLENV") else ""
        ) + "PV_PLUGIN_PATH/p:PARAVIEW_LOG_PLUGIN_VERBOSITY:PLAID_PORT:SCRIPT_PATH"

    if options is None:
        options = {}
    tree_name = options.get("tree_name", "PlaidExplorer1")

    script = f"""

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Plaid Explorer'
plaidExplorer1 = PlaidExplorer(registrationName={tree_name!r})

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
plaidExplorer1Display = Show(plaidExplorer1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
plaidExplorer1Display.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# update the view to ensure updated data information
renderView1.Update()
Show(plaidExplorer1)
Render()
Show(plaidExplorer1)
Render()
"""
    script_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(script_dir.name, "script.py")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(script)
        my_env["SCRIPT_PATH"] = file_path
        process = subprocess.Popen(
            [current_pv_path, "--script", file_path],
            env=my_env,
            stdin=subprocess.DEVNULL,
        )
    except BaseException:
        script_dir.cleanup()
        raise
    # ParaView may not read the script immediately after Popen returns.
    atexit.register(script_dir.cleanup)
    return process
