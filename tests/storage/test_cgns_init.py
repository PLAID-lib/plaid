# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#

from pathlib import Path

import pytest

import plaid.storage.cgns as cgns
from plaid.storage.cgns import CgnsBackend


def test_public_exports_and_backend_name():
    expected = {
        "configure_dataset_card",
        "download_datasetdict_from_hub",
        "generate_datasetdict_to_disk",
        "init_datasetdict_from_disk",
        "init_datasetdict_streaming_from_hub",
        "push_local_datasetdict_to_hub",
    }
    assert set(cgns.__all__) == expected
    assert CgnsBackend.name == "cgns"


def test_cgns_backend_init_from_disk_delegates(monkeypatch):
    call = {}

    def fake_init_datasetdict_from_disk(path):
        call["path"] = path
        return {"train": "dataset"}

    monkeypatch.setattr(cgns, "init_datasetdict_from_disk", fake_init_datasetdict_from_disk)

    local_path = Path("/tmp/my_dataset")
    result = CgnsBackend.init_from_disk(local_path)

    assert result == {"train": "dataset"}
    assert call == {"path": local_path}


def test_cgns_backend_download_from_hub_delegates(monkeypatch):
    call = {}

    def fake_download_datasetdict_from_hub(repo_id, local_dir):
        call["repo_id"] = repo_id
        call["local_dir"] = local_dir
        return "downloaded_path"

    monkeypatch.setattr(cgns, "download_datasetdict_from_hub", fake_download_datasetdict_from_hub)

    backend = CgnsBackend()
    result = backend.download_from_hub("dummy/repo", "/tmp/local")

    assert result == "downloaded_path"
    assert call == {"repo_id": "dummy/repo", "local_dir": "/tmp/local"}


def test_cgns_backend_streaming_from_hub_current_behavior_raises_name_error():
    backend = CgnsBackend()

    backend.init_datasetdict_streaming_from_hub("PhysArena/Rotor37")


def test_cgns_backend_generate_to_disk_delegates(monkeypatch):
    call = {}

    def fake_generate_datasetdict_to_disk(**kwargs):
        call.update(kwargs)
        return "ok"

    monkeypatch.setattr(cgns, "generate_datasetdict_to_disk", fake_generate_datasetdict_to_disk)

    generators = {"train": lambda: iter(())}
    variable_schema = {"Global/temperature": {"dtype": "float32", "ndim": 1}}
    gen_kwargs = {"train": {"shards_ids": [[0, 1]]}}

    result = CgnsBackend.generate_to_disk(
        output_folder="/tmp/output",
        generators=generators,
        variable_schema=variable_schema,
        gen_kwargs=gen_kwargs,
        num_proc=2,
        verbose=True,
    )

    assert result == "ok"
    assert call == {
        "output_folder": "/tmp/output",
        "generators": generators,
        "variable_schema": variable_schema,
        "gen_kwargs": gen_kwargs,
        "num_proc": 2,
        "verbose": True,
    }


def test_cgns_backend_push_local_to_hub_current_behavior(monkeypatch):
    call = {}

    def fake_push_local_datasetdict_to_hub(*args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return "pushed"

    monkeypatch.setattr(cgns, "push_local_datasetdict_to_hub", fake_push_local_datasetdict_to_hub)

    backend = CgnsBackend()
    result = backend.push_local_to_hub("dummy/repo", "/tmp/local")

    assert result == "pushed"
    assert call["args"][0] is backend
    assert call["args"][1] == "dummy/repo"
    assert call["kwargs"] == {"local_dir": "/tmp/local"}


def test_cgns_backend_get_configure_dataset_card():
    backend = CgnsBackend()
    assert backend.get_configure_dataset_card() is cgns.configure_dataset_card


def test_cgns_backend_to_var_sample_dict_raises_value_error():
    with pytest.raises(ValueError, match="to_dict not available for 'cgns' backend"):
        CgnsBackend.to_var_sample_dict(dataset=None, idx=0, features=[])


def test_cgns_backend_sample_to_var_sample_dict_raises_value_error():
    with pytest.raises(
        ValueError, match="sample_to_var_sample_dict not available for 'cgns' backend"
    ):
        CgnsBackend.sample_to_var_sample_dict(sample={})