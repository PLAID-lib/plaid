"""Tests for the viewer artifact cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from plaid.viewer import cache as cache_mod
from plaid.viewer.cache import CacheRoot, _process_is_alive, sweep_orphans


def test_ephemeral_cache_is_cleaned_up_on_close(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    cache = CacheRoot(install_signal_handlers=False, run_orphan_sweep=False)
    path = cache.path
    assert path.exists()
    assert cache.is_ephemeral is True
    cache.close()
    assert not path.exists()


def test_persistent_cache_is_preserved(tmp_path: Path) -> None:
    target = tmp_path / "persistent"
    cache = CacheRoot(persistent_dir=target, install_signal_handlers=False)
    assert cache.path == target
    assert cache.is_ephemeral is False
    cache.close()
    assert target.exists()


def test_context_manager_removes_ephemeral_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with CacheRoot(install_signal_handlers=False, run_orphan_sweep=False) as cache:
        path = cache.path
        assert path.exists()
    assert not path.exists()


def test_sweep_orphans_removes_dead_pid_dir(tmp_path: Path) -> None:
    victim = tmp_path / "plaid-viewer-999999-deadbeefcafe"
    victim.mkdir()
    removed = sweep_orphans(tmp_path)
    assert victim in removed
    assert not victim.exists()


def test_sweep_orphans_keeps_live_pid_dir(tmp_path: Path) -> None:
    import os

    live = tmp_path / f"plaid-viewer-{os.getpid()}-abc123def456"
    live.mkdir()
    removed = sweep_orphans(tmp_path)
    assert live not in removed
    assert live.exists()


def test_process_is_alive_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _process_is_alive(0) is False

    def missing(_pid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(cache_mod.os, "kill", missing)
    assert _process_is_alive(123) is False

    def denied(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(cache_mod.os, "kill", denied)
    assert _process_is_alive(123) is True

    def other_os_error(_pid: int, _sig: int) -> None:
        raise OSError(5, "other")

    monkeypatch.setattr(cache_mod.os, "kill", other_os_error)
    assert _process_is_alive(123) is True

    def no_such_process(_pid: int, _sig: int) -> None:
        raise OSError(cache_mod.errno.ESRCH, "missing")

    monkeypatch.setattr(cache_mod.os, "kill", no_such_process)
    assert _process_is_alive(123) is False


def test_sweep_orphans_ignores_non_dirs_and_non_matching_names(tmp_path: Path) -> None:
    (tmp_path / "plain-file").write_text("x")
    keep = tmp_path / "not-plaid-viewer"
    keep.mkdir()
    assert sweep_orphans(tmp_path / "missing") == []
    assert sweep_orphans(tmp_path) == []
    assert keep.exists()


def test_cache_runs_orphan_sweep_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    victim = tmp_path / "plaid-viewer-999999-deadbeef"
    victim.mkdir()
    cache = CacheRoot(install_signal_handlers=False, run_orphan_sweep=True)
    assert not victim.exists()
    path = cache.path
    cache.close()
    cache.close()
    assert not path.exists()


def test_cache_signal_handler_cleans_then_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    calls: list[tuple[str, object]] = []
    handlers: dict[int, object] = {}

    def previous(signum, _frame):
        calls.append(("previous", signum))

    def fake_getsignal(_sig):
        return previous

    def fake_signal(sig, handler):
        handlers[sig] = handler
        calls.append(("signal", sig))

    def fake_kill(_pid, sig):
        calls.append(("kill", sig))

    monkeypatch.setattr(cache_mod.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(cache_mod.signal, "signal", fake_signal)
    monkeypatch.setattr(cache_mod.os, "kill", fake_kill)

    cache = CacheRoot(install_signal_handlers=True, run_orphan_sweep=False)
    path = cache.path
    handler = handlers[cache_mod.signal.SIGINT]
    handler(cache_mod.signal.SIGINT, None)

    assert not path.exists()
    assert ("previous", cache_mod.signal.SIGINT) in calls
    assert ("kill", cache_mod.signal.SIGINT) in calls


def test_sweep_orphans_logs_rmtree_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    victim = tmp_path / "plaid-viewer-999999-deadbeefcafe"
    victim.mkdir()

    def broken_rmtree(_path: Path, ignore_errors: bool = False) -> None:  # noqa: ARG001, FBT001, FBT002
        raise OSError("boom")

    monkeypatch.setattr(cache_mod.shutil, "rmtree", broken_rmtree)
    removed = sweep_orphans(tmp_path)
    assert removed == []
    assert "Could not remove orphan viewer cache" in caplog.text


def test_cache_safe_cleanup_logs_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cache_mod.tempfile, "gettempdir", lambda: str(tmp_path))
    cache = CacheRoot(install_signal_handlers=False, run_orphan_sweep=False)

    def broken_rmtree(_path: Path, ignore_errors: bool = False) -> None:  # noqa: ARG001, FBT001, FBT002
        raise RuntimeError("boom")

    monkeypatch.setattr(cache_mod.shutil, "rmtree", broken_rmtree)
    cache.close()
    assert "Failed to clean viewer cache" in caplog.text


def test_cache_signal_handler_install_ignores_signal_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_mod.tempfile, "gettempdir", lambda: str(tmp_path))

    calls = {"getsignal": 0, "signal": 0}

    def flaky_getsignal(_sig):
        calls["getsignal"] += 1
        if calls["getsignal"] == 1:
            raise ValueError("not main thread")
        return cache_mod.signal.SIG_IGN

    def broken_signal(_sig, _handler):
        calls["signal"] += 1
        raise OSError("not main thread")

    monkeypatch.setattr(cache_mod.signal, "getsignal", flaky_getsignal)
    monkeypatch.setattr(cache_mod.signal, "signal", broken_signal)
    cache = CacheRoot(install_signal_handlers=True, run_orphan_sweep=False)
    try:
        assert calls == {"getsignal": 2, "signal": 1}
    finally:
        cache.close()
