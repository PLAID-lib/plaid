from pathlib import Path
from unittest.mock import Mock

import pytest

import plaid.storage.common.reader as common_reader
import plaid.storage.hf_datasets.reader as hf_datasets_reader


@pytest.mark.parametrize(
    "arguments,expected",
    [
        pytest.param(
            {
                "repo_id": "dummy/repo",
                "local_dir": "/tmp/local",
            },
            {
                "output_folder": Path("/tmp/local"),
            },
            id="Absolute path",
        ),
        pytest.param(
            {
                "repo_id": "dummy/repo",
                "local_dir": "~/local",
            },
            {
                "output_folder": Path("~/local").expanduser(),
            },
            id="Path contains user dir",
        ),
    ],
)
def test_hf_datasets_download_datasetdict_split_download(
    monkeypatch,
    arguments,
    expected,
):
    fake__split_download = Mock()

    monkeypatch.setattr(
        common_reader,
        "_prepare_local_folder_on_disk",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        hf_datasets_reader,
        "snapshot_download",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        hf_datasets_reader,
        "_split_download",
        fake__split_download,
    )

    result = hf_datasets_reader.download_datasetdict_from_hub(**arguments)

    fake__split_download.assert_called_once()
    call_kwargs = fake__split_download.call_args.kwargs

    for key, value in expected.items():
        assert call_kwargs[key] == value

    assert result == str(expected["output_folder"])
