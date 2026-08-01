from unittest.mock import MagicMock

import pytest
from datasets.utils import logging as datasets_logging

from plaid.storage.hf_datasets import writer


def test_save_datasetdict_uses_in_process_progress(tmp_path):
    datasetdict = MagicMock()
    dataset = MagicMock()
    dataset.__len__.return_value = 10
    dataset.data.nbytes = 1
    datasetdict.items.return_value = [("train", dataset)]

    writer.save_datasetdict_to_disk(tmp_path, datasetdict, num_proc=4)

    datasetdict.save_to_disk.assert_called_once_with(
        str(tmp_path / "data"), num_shards={"train": 1}, num_proc=None
    )


@pytest.mark.parametrize("verbose", [False, True])
def test_generate_controls_and_restores_hf_progress(monkeypatch, tmp_path, verbose):
    datasetdict = MagicMock()
    observed = []

    def fake_generate(*_args, **_kwargs):
        observed.append(datasets_logging.is_progress_bar_enabled())
        return datasetdict

    monkeypatch.setattr(writer, "generator_to_datasetdict", fake_generate)
    monkeypatch.setattr(writer, "save_datasetdict_to_disk", MagicMock())

    initially_enabled = datasets_logging.is_progress_bar_enabled()
    writer.generate_datasetdict_to_disk(
        tmp_path,
        generators={"train": MagicMock()},
        variable_schema={},
        verbose=verbose,
    )

    assert observed == [verbose]
    assert datasets_logging.is_progress_bar_enabled() is initially_enabled
