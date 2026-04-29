"""Tests for the viewer artifact cache."""

from __future__ import annotations

from pathlib import Path

from plaid.viewer.cache import CacheRoot, sweep_orphans


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
