#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#

from pathlib import Path

import plaid.storage.hf_datasets as hf_datasets
from plaid.storage.hf_datasets import HFBackend


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
    assert set(hf_datasets.__all__) == expected
    assert HFBackend.name == "hf_datasets"


def test_hf_backend_init_from_disk_delegates(monkeypatch):
    call = {}

    def fake_init_datasetdict_from_disk(path):
        call["path"] = path
        return {"train": "dataset"}

    monkeypatch.setattr(hf_datasets, "init_datasetdict_from_disk", fake_init_datasetdict_from_disk)

    local_path = Path("/tmp/my_dataset")
    result = HFBackend.init_from_disk(local_path)

    assert result == {"train": "dataset"}
    assert call == {"path": local_path}


def test_hf_backend_download_from_hub_delegates(monkeypatch):
    call = {}

    def fake_download_datasetdict_from_hub(repo_id, local_dir, split_ids, features, overwrite):
        call["repo_id"] = repo_id
        call["local_dir"] = local_dir
        call["split_ids"] = split_ids
        call["features"] = features
        call["overwrite"] = overwrite
        return "downloaded_path"

    monkeypatch.setattr(hf_datasets, "download_datasetdict_from_hub", fake_download_datasetdict_from_hub)

    result = HFBackend.download_from_hub(
        repo_id="dummy/repo",
        local_dir="/tmp/local",
        split_ids={"train": 0},
        features=["path/to/feature"],
        overwrite=True,
    )

    assert result == "downloaded_path"
    assert call == {
        "repo_id": "dummy/repo",
        "local_dir": "/tmp/local",
        "split_ids": {"train": 0},
        "features": ["path/to/feature"],
        "overwrite": True,
    }


def test_hf_backend_streaming_from_hub_delegates(monkeypatch):
    call = {}

    def fake_init_datasetdict_streaming_from_hub(repo_id, split_ids, features):
        call["repo_id"] = repo_id
        call["split_ids"] = split_ids
        call["features"] = features
        return {"train": "streaming_dataset"}

    monkeypatch.setattr(
        hf_datasets,
        "init_datasetdict_streaming_from_hub",
        fake_init_datasetdict_streaming_from_hub,
    )

    result = HFBackend.init_datasetdict_streaming_from_hub(
        repo_id="PhysArena/Rotor37",
        split_ids={"train": [0, 1]},
        features=["Base/Zone/Field"],
    )

    assert result == {"train": "streaming_dataset"}
    assert call == {
        "repo_id": "PhysArena/Rotor37",
        "split_ids": {"train": [0, 1]},
        "features": ["Base/Zone/Field"],
    }


def test_hf_backend_streaming_from_hub_default_args(monkeypatch):
    call = {}

    def fake_init_datasetdict_streaming_from_hub(repo_id, split_ids=None, features=None):
        call["repo_id"] = repo_id
        call["split_ids"] = split_ids
        call["features"] = features
        return {"train": "streaming_dataset"}

    monkeypatch.setattr(
        hf_datasets,
        "init_datasetdict_streaming_from_hub",
        fake_init_datasetdict_streaming_from_hub,
    )

    result = HFBackend.init_datasetdict_streaming_from_hub("PhysArena/Rotor37")

    assert result == {"train": "streaming_dataset"}
    assert call == {
        "repo_id": "PhysArena/Rotor37",
        "split_ids": None,
        "features": None,
    }


def test_hf_backend_generate_to_disk_delegates(monkeypatch):
    call = {}

    def fake_generate_datasetdict_to_disk(**kwargs):
        call.update(kwargs)
        return "ok"

    monkeypatch.setattr(hf_datasets, "generate_datasetdict_to_disk", fake_generate_datasetdict_to_disk)

    generators = {"train": lambda: iter(())}
    variable_schema = {"Global/temperature": {"dtype": "float32", "ndim": 1}}
    gen_kwargs = {"train": {"shards_ids": [[0, 1]]}}

    result = HFBackend.generate_to_disk(
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


def test_hf_backend_push_local_to_hub_delegates(monkeypatch):
    call = {}

    def fake_push_local_datasetdict_to_hub(repo_id, local_dir, num_workers=1):
        call["repo_id"] = repo_id
        call["local_dir"] = local_dir
        call["num_workers"] = num_workers
        return "pushed"

    monkeypatch.setattr(
        hf_datasets,
        "push_local_datasetdict_to_hub",
        fake_push_local_datasetdict_to_hub,
    )

    result = HFBackend.push_local_to_hub("dummy/repo", "/tmp/local")

    assert result == "pushed"
    assert call == {
        "repo_id": "dummy/repo",
        "local_dir": "/tmp/local",
        "num_workers": 1,
    }


def test_hf_backend_configure_dataset_card_delegates(monkeypatch):
    call = {}

    def fake_configure_dataset_card(
        repo_id,
        infos,
        local_dir=None,
        viewer=False,
        pretty_name=None,
        dataset_long_description=None,
        illustration_urls=None,
        arxiv_paper_urls=None,
    ):
        call["repo_id"] = repo_id
        call["infos"] = infos
        call["local_dir"] = local_dir
        return "configured"

    monkeypatch.setattr(hf_datasets, "configure_dataset_card", fake_configure_dataset_card)

    infos = {"legal": {"owner": "owner", "license": "cc-by-4.0"}}
    result = HFBackend.configure_dataset_card("dummy/repo", infos)

    assert result == "configured"
    assert call == {"repo_id": "dummy/repo", "infos": infos, "local_dir": None}


def test_hf_backend_to_var_sample_dict_delegates(monkeypatch):
    call = {}

    def fake_to_var_sample_dict(ds, i, features):
        call["ds"] = ds
        call["i"] = i
        call["features"] = features
        return {"field": [1, 2, 3]}

    monkeypatch.setattr(hf_datasets, "to_var_sample_dict", fake_to_var_sample_dict)

    dataset = object()
    features = ["Base/Zone/Field"]
    result = HFBackend.to_var_sample_dict(dataset=dataset, idx=3, features=features)

    assert result == {"field": [1, 2, 3]}
    assert call == {"ds": dataset, "i": 3, "features": features}


def test_hf_backend_sample_to_var_sample_dict_delegates(monkeypatch):
    call = {}

    def fake_sample_to_var_sample_dict(hf_sample):
        call["hf_sample"] = hf_sample
        return {"field": [4, 5]}

    monkeypatch.setattr(hf_datasets, "sample_to_var_sample_dict", fake_sample_to_var_sample_dict)

    sample = {"Base": {"Zone": {}}}
    result = HFBackend.sample_to_var_sample_dict(sample)

    assert result == {"field": [4, 5]}
    assert call == {"hf_sample": sample}