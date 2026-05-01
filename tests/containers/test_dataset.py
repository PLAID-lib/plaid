# %% Imports

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.version import Version

import plaid
import plaid.containers.dataset as dataset_module
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample

# %% Fixtures


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


# %% Functions


def compare_two_samples(sample_1: Sample, sample_2: Sample):
    assert set(sample_1.features.get_all_time_values()) == set(
        sample_2.features.get_all_time_values()
    )
    assert set(sample_1.get_global_names()) == set(sample_2.get_global_names())
    assert set(sample_1.get_field_names()) == set(sample_2.get_field_names())
    assert np.array_equal(sample_1.get_nodes(), sample_2.get_nodes())
    assert set(sample_1.features.get_base_names()) == set(
        sample_2.features.get_base_names()
    )
    for base_name in sample_1.features.get_base_names():
        assert set(sample_1.features.get_zone_names(base_name)) == set(
            sample_2.features.get_zone_names(base_name)
        )
        for zone_name in sample_1.features.get_zone_names(base_name):
            assert sample_1.features.get_zone_type(
                zone_name, base_name
            ) == sample_2.features.get_zone_type(zone_name, base_name)


# %% Tests


class Test_Dataset:
    # -------------------------------------------------------------------------#
    def test___init__(self, dataset):
        assert len(dataset) == 0
        assert dataset.get_samples() == []

    def test___init__load(self, current_directory):
        dataset_path = current_directory / "dataset"
        dataset_already_filled = Dataset(path = dataset_path, split="train")
        assert len(dataset_already_filled) == 10


    def test___init__unknown_directory(self, current_directory):
        dataset_path = current_directory / "dataset_unknown"
        with pytest.raises(FileNotFoundError):
            Dataset(path=dataset_path).load(split="test")

    def test__init__path(self, current_directory):
        dataset_path = current_directory / "dataset"
        Dataset(path=dataset_path, split="train")


    # -------------------------------------------------------------------------#
    def test_get_samples(self, dataset_with_samples, nb_samples):
        dataset_with_samples.get_samples()
        dataset_with_samples.get_samples(np.arange(np.random.randint(2, nb_samples)))


    def test_add_sample(self, dataset, sample):
        assert dataset.get_backend().add_sample(sample) == 0
        assert len(dataset) == 1

    def test_del_sample_classical(self, dataset, sample):
        for i in range(10):
            assert dataset.get_backend().add_sample(sample) == i









    def test_on_error_del_sample(self, dataset, sample):
        for i in range(3):
            assert dataset.get_backend().add_sample(sample) == i



    def test_add_sample_and_id(self, dataset, sample):
        dataset.get_backend().add_sample(sample, 10)
        assert len(dataset) == 1
        with pytest.raises(ValueError):
            dataset.get_backend().add_sample(sample, -5)

    def test_add_sample_not_a_sample(self, dataset):
        with pytest.raises(TypeError):
            dataset.get_backend().add_sample("not_a_sample")
        with pytest.raises(TypeError):
            dataset.get_backend().add_sample(1)

    def test_add_sample_empty(self, empty_dataset):
        with pytest.raises(TypeError):
            empty_dataset.get_backend().add_sample([], 1)

    def test_add_sample_empty_with_ids(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            empty_dataset.get_backend().add_sample([sample], [1, 2, 3])

    def test_add_sample_bad_number_ids_inf(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            samples = [sample, sample, sample]
            empty_dataset.get_backend().add_sample(samples, [1, 2])

    def test_add_sample_bad_number_ids_supp(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            samples = [sample, sample, sample]
            empty_dataset.get_backend().add_sample(samples, [1, 2, 3, 4])

    def test_add_sample_with_same_ids(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            samples = [sample, sample, sample]
            empty_dataset.get_backend().add_sample(samples, [1, 1, 1])

    def test_add_sample_with_ids_good(self, dataset, sample):
        samples = [sample, sample, sample]
        size_before = len(dataset)
        assert dataset.get_backend().add_sample(samples, [1, 2, 3]) == [1, 2, 3]
        assert size_before + len([1, 2, 3]) == len(dataset)

    def test_add_sample(self, dataset_with_samples, other_samples):
        size_before = len(dataset_with_samples)
        assert (
            dataset_with_samples.get_backend().add_sample(other_samples)
            == np.arange(size_before, size_before + len(other_samples))
        ).all()

    def test_add_sample_not_a_list_of_samples(self, dataset, sample):
        with pytest.raises(TypeError):
            dataset.get_backend().add_sample({0: sample})
        with pytest.raises(TypeError):
            dataset.get_backend().add_sample(["not_a_sample"])


    def test_get_sample_ids(self, dataset):
        dataset.get_sample_ids()

    def test_get_sample_ids_from_disk(self, current_directory):
        dataset_path = current_directory / "dataset" /  "data" / "test"
        assert plaid.get_number_of_samples(dataset_path) == 10


    def test___init___samples_and_path(self, samples, tmp_path):
        # Expects an error since path and samples are provided
        fname = tmp_path / "test.plaid"
        with pytest.raises(ValueError):
            Dataset(samples=samples, path=str(fname))

    # -------------------------------------------------------------------------#

    # -------------------------------------------------------------------------#
    def test___len__empty(self, dataset):
        assert len(dataset) == 0

    def test___len__(self, dataset_with_samples, nb_samples):
        assert len(dataset_with_samples) == nb_samples

    # def test___iter__empty(self, dataset):
    #     for _ in dataset:

    def test___iter__(self, dataset_with_samples):
        sub_dataset = dataset_with_samples[range(0, len(dataset_with_samples), 2)]
        length = len(sub_dataset)
        count = 0
        for sample in sub_dataset:
            count += 1
            assert isinstance(sample, Sample)
        assert count == length


    def test___repr__(self, dataset, dataset_with_samples):
        print(dataset)
        print(dataset_with_samples)

    def test_setattr_path_split_immutability(self, current_directory):
        dataset_path = current_directory / "dataset"
        dataset = Dataset(path=dataset_path, split="train")

        # same value should be accepted (early return branch)
        dataset.path = dataset_path
        dataset.split = "train"

        with pytest.raises(AttributeError):
            dataset.path = current_directory

        with pytest.raises(AttributeError):
            dataset.split = "test"

    def test_from_path(self, current_directory, monkeypatch):
        calls = []

        def fake_load(self, path=None, split=None):
            calls.append((path, split))

        monkeypatch.setattr(Dataset, "load", fake_load)

        dataset_path = current_directory / "dataset"
        pb_def = dataset_module.ProblemDefinition(
            train_split={"train": "all"},
            test_split={"test": "all"},
        )
        ds = Dataset.from_path(path=dataset_path, problem_definition=pb_def)

        assert isinstance(ds, Dataset)
        assert ds.path == dataset_path
        assert calls == [(dataset_path, None)]

    def test_set_infos_validation_and_metadata(self, dataset):
        infos = {
            "legal": {"owner": "PLAID2", "license": "BSD-3"},
            "data_production": {"type": "simulation", "simulator": "Z-set"},
        }
        dataset.set_infos(infos, warn=False)
        assert "plaid" in dataset.infos
        assert isinstance(dataset.infos["plaid"]["version"], Version)

    def test_set_infos_invalid_category(self, dataset):
        with pytest.raises(KeyError):
            dataset.set_infos({"unknown_category": {"owner": "x"}}, warn=False)

    def test_set_infos_invalid_info_key(self, dataset):
        with pytest.raises(KeyError):
            dataset.set_infos({"legal": {"not_allowed": "x"}}, warn=False)

    def test_set_infos_warn_branch(self, dataset, monkeypatch):
        warnings = []

        class _Logger:
            def warning(self, msg):
                warnings.append(msg)

        monkeypatch.setattr(dataset_module, "logger", _Logger(), raising=False)
        dataset.infos = {"legal": {"owner": "old", "license": "old"}}
        dataset.set_infos({"legal": {"owner": "new", "license": "new"}}, warn=True)
        assert warnings == ["infos not empty, replacing it anyway"]

    def test_load_without_path_raises(self, dataset):
        with pytest.raises(RuntimeError):
            dataset.load()

    def test_load_from_file_branch(self, tmp_path, monkeypatch):
        archive = tmp_path / "dataset.tar"
        archive.write_text("dummy")

        extracted = tmp_path / "extracted"
        extracted.mkdir()
        registered = []

        monkeypatch.setattr("tempfile.mkdtemp", lambda prefix: str(extracted))
        monkeypatch.setattr("subprocess.call", lambda _args: 0)
        monkeypatch.setattr("atexit.register", lambda *args: registered.append(args))

        sample = Sample()
        monkeypatch.setattr(
            "plaid.storage.init_from_disk",
            lambda _path: (
                {"train": [sample, sample]},
                {"train": "converter"},
            ),
        )

        ds = Dataset()
        ds.load(path=archive)

        assert (ds.indices == np.arange(2)).all()
        assert len(registered) == 1

    @pytest.mark.parametrize(
        "train_split, expected_indices, should_raise",
        [
            ({"train": "all"}, "all", False),
            ({"train": [0, 2]}, np.array([0, 2]), False),
            ({"train": None}, None, True),
        ],
    )
    def test_from_train_split_indices_branches(
        self, tmp_path, monkeypatch, train_split, expected_indices, should_raise
    ):
        monkeypatch.setattr(Dataset, "load", lambda self, path=None, split=None: None)
        fake_pb_def = SimpleNamespace(train_split=train_split)
        monkeypatch.setattr(
            dataset_module.ProblemDefinition,
            "from_path",
            lambda path, name: fake_pb_def,
        )

        if should_raise:
            with pytest.raises(TypeError):
                Dataset.from_train_split(path=tmp_path)
            return

        ds = Dataset.from_train_split(path=tmp_path)
        assert ds.stage == "training"
        assert ds.split == "train"
        if isinstance(expected_indices, np.ndarray):
            assert np.array_equal(ds.indices, expected_indices)
        else:
            assert ds.indices == expected_indices

    def test_len_invalid_indices_string(self, dataset):
        object.__setattr__(dataset, "indices", "invalid")
        with pytest.raises(RuntimeError):
            len(dataset)

    def test_get_samples_and_ids_with_explicit_indices(self, dataset_with_samples):
        object.__setattr__(dataset_with_samples, "indices", [0, 2])
        samples = dataset_with_samples.get_samples()
        assert len(samples) == 2
        assert dataset_with_samples.get_sample_ids() == [0, 2]

    def test_save_to_dir(self, dataset_with_samples, tmp_path, monkeypatch):
        saved_calls = []

        def fake_save_to_disk(**kwargs):
            saved_calls.append(kwargs)

        monkeypatch.setattr("plaid.storage.save_to_disk", fake_save_to_disk)

        dataset_with_samples.split = "train"
        dataset_with_samples.save_to_dir(tmp_path, verbose=True)

        dataset_without_split = Dataset()
        dataset_without_split.get_backend().add_sample(Sample())
        dataset_without_split.save_to_dir(tmp_path / "default_split")

        assert len(saved_calls) == 2
        assert "train" in saved_calls[0]["ids"]
        assert "train" in saved_calls[1]["ids"]

