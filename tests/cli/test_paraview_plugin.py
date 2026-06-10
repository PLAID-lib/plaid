"""Tests for the ParaView plugin CLI helper module."""

import os
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


def test_get_paraview_plugin_path_one_file_writes_bundled_plugin(
    monkeypatch,
    tmp_path,
):
    """The bundled plugin should include helper modules in a temporary file."""
    monkeypatch.setattr(paraview_plugin.tempfile, "mkdtemp", lambda: str(tmp_path))

    plugin_directory = paraview_plugin.get_ParaView_plugin_path_one_file()

    plugin_file = Path(plugin_directory) / "PlaidParaViewPlugin.py"
    content = plugin_file.read_text()
    assert Path(plugin_directory) == tmp_path
    assert plugin_file.exists()
    assert "# ##INCLUDE PLACEHOLDER##" not in content
    assert "def CGNSTreeToVtk" in content
    assert "def cgns_tree_to_json_payload" in content
    assert "from __future__ import annotations" not in content


def test_run_paraview_with_plugin_sets_environment(monkeypatch, tmp_path):
    """Launching ParaView should pass plugin-related environment variables."""
    calls = []

    def fake_popen(args, env):
        calls.append({"args": args, "env": env})
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(
        paraview_plugin,
        "get_ParaView_plugin_path_one_file",
        lambda: tmp_path,
    )
    monkeypatch.setattr(paraview_plugin.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("PARAVIEW_EXEC", "custom-paraview")

    process = paraview_plugin.run_paraview_with_plugin()

    assert process.pid == 1234
    assert calls[0]["args"] == ["custom-paraview"]
    assert calls[0]["env"]["PV_PLUGIN_PATH"] == str(tmp_path)
    assert calls[0]["env"]["PARAVIEW_LOG_PLUGIN_VERBOSITY"] == "ON"
    assert os.environ["PARAVIEW_EXEC"] == "custom-paraview"


def test_run_paraview_with_plugin_sets_wslenv_for_windows_paraview(
    monkeypatch,
    tmp_path,
):
    """WSL launches should propagate plugin variables to Windows ParaView."""
    calls = []

    monkeypatch.setattr(
        paraview_plugin,
        "get_ParaView_plugin_path_one_file",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        paraview_plugin.subprocess,
        "Popen",
        lambda args, env: calls.append({"args": args, "env": env}) or object(),
    )
    monkeypatch.setenv("PARAVIEW_EXEC", "/mnt/c/ParaView/bin/paraview.exe")

    paraview_plugin.run_paraview_with_plugin()

    assert calls[0]["env"]["WSLENV"] == (
        "PV_PLUGIN_PATH/p:PARAVIEW_LOG_PLUGIN_VERBOSITY/p"
    )
