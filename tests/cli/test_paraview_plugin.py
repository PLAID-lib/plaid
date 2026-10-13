"""Tests for the ParaView plugin CLI helper module."""

import ast
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from urllib import error, request

import pytest

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
        return SimpleNamespace(stdout="Z:\\Users\\me\\plugin\r\n")

    monkeypatch.setattr(paraview_plugin.subprocess, "run", fake_run)

    assert (
        paraview_plugin.convert_wsl_to_win("/mnt/z/Users/me/plugin")
        == "Z:\\Users\\me\\plugin"
    )
    assert calls == [
        {
            "args": ["wslpath", "-w", "/mnt/z/Users/me/plugin"],
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
    assert "self.ensure_3D_points = False" in content
    assert "from __future__ import annotations" not in content


def test_get_paraview_plugin_path_one_file_uses_provided_directory(tmp_path):
    """Write the bundled plugin into a caller-provided directory."""
    plugin_directory = paraview_plugin.get_ParaView_plugin_path_one_file(tmp_path)

    assert plugin_directory == str(tmp_path)
    assert (tmp_path / "PlaidParaViewPlugin.py").is_file()


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


def test_process_request_json_returns_response_and_propagates_errors(
    monkeypatch, capsys
):
    """The process request returns JSON or raises the original transport error."""
    plugin_path = paraview_plugin.get_ParaView_plugin_path() / (
        "PlaidParaViewPlugin.py"
    )
    tree = ast.parse(plugin_path.read_text())
    process_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PlaidProcess"
    )
    method = next(
        node
        for node in process_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_request_json"
    )
    namespace = {
        "Optional": Optional,
        "json": json,
        "request": request,
        "error": error,
        "socket": socket,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(plugin_path), "exec"),
        namespace,
    )
    client = SimpleNamespace(host="localhost", port=8001)

    class Response:
        """Provide a JSON HTTP response context manager."""

        def __enter__(self):
            """Return the response."""
            return self

        def __exit__(self, *_args):
            """Close the response context."""

        def read(self):
            """Return a JSON-encoded process result."""
            return b'{"samples": []}'

    monkeypatch.setattr(request, "urlopen", lambda *_args, **_kwargs: Response())
    assert namespace["_request_json"](client, "/process", {}) == {"samples": []}

    failures = (
        (error.HTTPError("/process", 503, "Unavailable", {}, None), "HTTP error: 503"),
        (error.URLError(ConnectionRefusedError()), "Connection refused"),
        (socket.timeout(), "Connection timed out"),
    )
    for failure, message in failures:

        def fail(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(request, "urlopen", fail)
        with pytest.raises(type(failure)) as raised:
            namespace["_request_json"](client, "/process", {})
        assert raised.value is failure
        assert message in capsys.readouterr().out
