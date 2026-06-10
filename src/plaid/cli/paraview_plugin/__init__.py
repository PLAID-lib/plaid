from pathlib import Path
import  subprocess
import os


paraview_exec = "paraview"

def get_ParaView_plugin_path():
    return Path(__file__).parent

def convert_wsl_to_win(wsl_path: str) -> str:
    """Converts a WSL path (e.g., /mnt/c/Users) to Windows (C:\\Users)."""
    result = subprocess.run(
        ["wslpath", "-w", wsl_path], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()

def run_paraview_with_plugin():


    my_env = os.environ.copy()

    my_env["PV_PLUGIN_PATH"] = str(get_ParaView_plugin_path())
    my_env["PARAVIEW_LOG_PLUGIN_VERBOSITY"] = "ON"


    current_pv_path = os.environ.get("PARAVIEW_EXEC",paraview_exec)
    if current_pv_path.startswith("/mnt/") and os :
        # we are in a wsl and the paraview exec is windows, we need to convert the wsl path
        #my_env["PV_PLUGIN_PATH"] = convert_wsl_to_win(str(get_ParaView_plugin_path()))
        my_env["WSLENV"] = "PV_PLUGIN_PATH/p:PARAVIEW_LOG_PLUGIN_VERBOSITY/p"

    process = subprocess.Popen([os.environ.get("PARAVIEW_EXEC",paraview_exec)], env=my_env)
    return process
