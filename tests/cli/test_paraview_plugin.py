"""Tests for the ParaView plugin CLI helper module."""

from pathlib import Path
from types import SimpleNamespace

from plaid.cli import paraview_plugin


def test_get_paraview_plugin_path_returns_module_directory():
    """The plugin path points to the directory containing the helper module."""
    plugin_path = paraview_plugin.get_ParaView_plugin_path()

    assert plugin_path == Path(paraview_plugin.__file__).parent


def test_convert_wsl_to_win_uses_wslpath(monkeypatch):
    """WSL path conversion delegates to wslpath and strips its output."""
    calls = []

    def fake_run(args, capture_output, text, check):
        calls.append(
            {
                "args": args,
                "capture_output": capture_output,
                "text": text,
                "check": check,
            }
        )
        return SimpleNamespace(stdout="C:\\Users\\me\\plugin\r\n")

    monkeypatch.setattr(paraview_plugin.subprocess, "run", fake_run)

    converted = paraview_plugin.convert_wsl_to_win("/mnt/c/Users/me/plugin")

    assert converted == "C:\\Users\\me\\plugin"
    assert calls == [
        {
            "args": ["wslpath", "-w", "/mnt/c/Users/me/plugin"],
            "capture_output": True,
            "text": True,
            "check": True,
        }
    ]


def test_run_paraview_with_plugin_uses_default_executable(monkeypatch):
    """Launching ParaView sets plugin-related environment variables."""
    popen_calls = []

    def fake_popen(args, env):
        process = SimpleNamespace(args=args, env=env)
        popen_calls.append(process)
        return process

    monkeypatch.delenv("PARAVIEW_EXEC", raising=False)
    monkeypatch.setattr(paraview_plugin.subprocess, "Popen", fake_popen)

    process = paraview_plugin.run_paraview_with_plugin()
    popen_call = popen_calls[0]

    assert process is popen_call
    assert popen_call.args == [paraview_plugin.paraview_exec]
    assert popen_call.env["PV_PLUGIN_PATH"] == str(
        paraview_plugin.get_ParaView_plugin_path()
    )
    assert popen_call.env["PARAVIEW_LOG_PLUGIN_VERBOSITY"] == "ON"


def test_run_paraview_with_plugin_uses_configured_windows_executable(
    monkeypatch,
):
    """A WSL-mounted ParaView executable enables WSLENV path propagation."""
    popen_calls = []

    def fake_popen(args, env):
        process = SimpleNamespace(args=args, env=env)
        popen_calls.append(process)
        return process

    paraview_exec = "/mnt/c/Program Files/ParaView/bin/paraview.exe"
    monkeypatch.setenv("PARAVIEW_EXEC", paraview_exec)
    monkeypatch.setattr(paraview_plugin.subprocess, "Popen", fake_popen)

    process = paraview_plugin.run_paraview_with_plugin()
    popen_call = popen_calls[0]

    assert process is popen_call
    assert popen_call.args == [paraview_exec]
    assert popen_call.env["PV_PLUGIN_PATH"] == str(
        paraview_plugin.get_ParaView_plugin_path()
    )
    assert popen_call.env["PARAVIEW_LOG_PLUGIN_VERBOSITY"] == "ON"
    assert (
        popen_call.env["WSLENV"] == "PV_PLUGIN_PATH/p:PARAVIEW_LOG_PLUGIN_VERBOSITY/p"
    )
