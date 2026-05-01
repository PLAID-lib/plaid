#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#

from pathlib import Path

import plaid.storage.zarr as zarr
from plaid.storage.zarr import ZarrBackend


def test_public_exports_and_backend_name():
    expected = {
        "configure_dataset_card",
        "download_datasetdict_from_hub",
        "generate_datasetdict_to_disk",
        "init_datasetdict_from_disk",
        "init_datasetdict_streaming_from_hub",
        "push_local_datasetdict_to_hub",
        "sample_to_var_sample_dict",
        "to_var_sample_dict",
    }
    assert set(zarr.__all__) == expected
    assert ZarrBackend.name == "zarr"


def test_zarr_backend_init_from_disk_delegates(monkeypatch):
    call = {}

    def fake_init_datasetdict_from_disk(path):
        call["path"] = path
        return {"train": "dataset"}

    monkeypatch.setattr(zarr, "init_datasetdict_from_disk", fake_init_datasetdict_from_disk)

    local_path = Path("/tmp/my_dataset")
    result = ZarrBackend.init_from_disk(local_path)

    assert result == {"train": "dataset"}
    assert call == {"path": local_path}


def test_zarr_backend_download_from_hub_delegates(monkeypatch):
    call = {}

    def fake_download_datasetdict_from_hub(repo_id, local_dir):
        call["repo_id"] = repo_id
        call["local_dir"] = local_dir
        return "downloaded_path"

    monkeypatch.setattr(zarr, "download_datasetdict_from_hub", fake_download_datasetdict_from_hub)

    backend = ZarrBackend()
    result = backend.download_from_hub("dummy/repo", "/tmp/local")

    assert result == "downloaded_path"
    assert call == {"repo_id": "dummy/repo", "local_dir": "/tmp/local"}


def test_zarr_backend_streaming_from_hub_delegates(monkeypatch):
    call = {}

    def fake_init_datasetdict_streaming_from_hub(repo_id):
        call["repo_id"] = repo_id
        return {"train": "streaming_dataset"}

    monkeypatch.setattr(
        zarr,
        "init_datasetdict_streaming_from_hub",
        fake_init_datasetdict_streaming_from_hub,
    )

    backend = ZarrBackend()
    result = backend.init_datasetdict_streaming_from_hub("PhysArena/Rotor37")

    assert result == {"train": "streaming_dataset"}
    assert call == {"repo_id": "PhysArena/Rotor37"}


def test_zarr_backend_generate_to_disk_delegates(monkeypatch):
    call = {}

    def fake_generate_datasetdict_to_disk(**kwargs):
        call.update(kwargs)
        return "ok"

    monkeypatch.setattr(zarr, "generate_datasetdict_to_disk", fake_generate_datasetdict_to_disk)

    generators = {"train": lambda: iter(())}
    variable_schema = {"Global/temperature": {"dtype": "float32", "ndim": 1}}
    gen_kwargs = {"train": {"shards_ids": [[0, 1]]}}

    result = ZarrBackend.generate_to_disk(
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


def test_zarr_backend_push_local_datasetdict_to_hub_delegates(monkeypatch):
    call = {}

    def fake_push_local_datasetdict_to_hub(repo_id, local_dir):
        call["repo_id"] = repo_id
        call["local_dir"] = local_dir
        return "pushed"

    monkeypatch.setattr(
        zarr,
        "push_local_datasetdict_to_hub",
        fake_push_local_datasetdict_to_hub,
    )

    backend = ZarrBackend()
    result = backend.push_local_datasetdict_to_hub("dummy/repo", "/tmp/local")

    assert result == "pushed"
    assert call == {"repo_id": "dummy/repo", "local_dir": "/tmp/local"}


def test_zarr_backend_configure_dataset_card_delegates(monkeypatch):
    call = {}

    def fake_configure_dataset_card(repo_id, infos):
        call["repo_id"] = repo_id
        call["infos"] = infos
        return "configured"

    monkeypatch.setattr(zarr, "configure_dataset_card", fake_configure_dataset_card)

    backend = ZarrBackend()
    infos = {"legal": {"owner": "owner", "license": "cc-by-4.0"}}
    result = backend.configure_dataset_card("dummy/repo", infos)

    assert result == "configured"
    assert call == {"repo_id": "dummy/repo", "infos": infos}


def test_zarr_backend_to_var_sample_dict_delegates(monkeypatch):
    call = {}

    def fake_to_var_sample_dict(zarr_dataset, idx, features):
        call["zarr_dataset"] = zarr_dataset
        call["idx"] = idx
        call["features"] = features
        return {"field": [1, 2, 3]}

    monkeypatch.setattr(zarr, "to_var_sample_dict", fake_to_var_sample_dict)

    dataset = object()
    features = ["Base/Zone/Field"]
    result = ZarrBackend.to_var_sample_dict(dataset=dataset, idx=3, features=features)

    assert result == {"field": [1, 2, 3]}
    assert call == {"zarr_dataset": dataset, "idx": 3, "features": features}


def test_zarr_backend_sample_to_var_sample_dict_delegates(monkeypatch):
    call = {}

    def fake_sample_to_var_sample_dict(zarr_sample):
        call["zarr_sample"] = zarr_sample
        return {"field": [4, 5]}

    monkeypatch.setattr(zarr, "sample_to_var_sample_dict", fake_sample_to_var_sample_dict)

    sample = {"Base": {"Zone": {}}}
    result = ZarrBackend.sample_to_var_sample_dict(sample)

    assert result == {"field": [4, 5]}
    assert call == {"zarr_sample": sample}