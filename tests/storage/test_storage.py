# %% Imports

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from plaid.storage import (
    init_from_disk,
    load_problem_definitions_from_disk,
    save_to_disk,
)
from plaid.storage.cgns.writer import (
    generate_datasetdict_to_disk as cgns_generate_datasetdict_to_disk,
)
from plaid.storage.writer import (
    _build_gen_kwargs,
    _SampleFuncGenerator,
    _split_list,
)
from plaid.storage.zarr.writer import (
    generate_datasetdict_to_disk as zarr_generate_datasetdict_to_disk,
)


class _PicklableSampleLookup:
    """Module-level picklable callable that returns pre-built samples by index."""

    def __init__(self, samples: list):
        self._samples = samples

    def __call__(self, idx):
        return self._samples[idx]


def test_load_metadata_from_hub_materializes_memmaps(tmp_path, monkeypatch):
    """Hub metadata loader must return arrays independent from temp files."""
    from plaid.storage.common import reader as common_reader

    repo_root = tmp_path / "fake_hub_repo"
    constants_dir = repo_root / "constants" / "train"
    constants_dir.mkdir(parents=True)

    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    with open(constants_dir / "data.mmap", "wb") as f:
        f.write(data.tobytes(order="C"))

    with open(constants_dir / "layout.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "Global/cst_numeric": {
                    "offset": 0,
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                }
            },
            f,
        )

    with open(constants_dir / "constant_schema.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"Global/cst_numeric": {"dtype": str(data.dtype), "ndim": 2}}, f)

    with open(repo_root / "variable_schema.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"Global/var": {"dtype": "float32", "ndim": 1}}, f)

    with open(repo_root / "cgns_types.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"Global": "DataArray_t"}, f)

    def _fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        shutil.copytree(
            repo_root / "constants", local_dir / "constants", dirs_exist_ok=True
        )
        return str(local_dir)

    def _fake_hf_hub_download(**kwargs):
        return str(repo_root / kwargs["filename"])

    monkeypatch.setattr(common_reader, "snapshot_download", _fake_snapshot_download)
    monkeypatch.setattr(common_reader, "hf_hub_download", _fake_hf_hub_download)

    flat_cst, variable_schema, constant_schema, cgns_types = (
        common_reader.load_metadata_from_hub("dummy/repo")
    )

    loaded = flat_cst["train"]["Global/cst_numeric"]
    assert isinstance(loaded, np.ndarray)
    assert not isinstance(loaded, np.memmap)
    assert np.array_equal(loaded, data)
    assert variable_schema["Global/var"]["dtype"] == "float32"
    assert "Global/cst_numeric" in constant_schema["train"]
    assert cgns_types["Global"] == "DataArray_t"


def test_load_metadata_from_disk_keeps_memmaps(tmp_path):
    """Local metadata loader keeps memmap-backed numeric constants."""
    from plaid.storage.common import reader as common_reader

    dataset_root = tmp_path / "dataset"
    constants_dir = dataset_root / "constants" / "train"
    constants_dir.mkdir(parents=True)

    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    with open(constants_dir / "data.mmap", "wb") as f:
        f.write(data.tobytes(order="C"))

    with open(constants_dir / "layout.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "Global/cst_numeric": {
                    "offset": 0,
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                }
            },
            f,
        )

    with open(constants_dir / "constant_schema.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"Global/cst_numeric": {"dtype": str(data.dtype), "ndim": 2}}, f)

    with open(dataset_root / "variable_schema.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"Global/var": {"dtype": "float32", "ndim": 1}}, f)

    with open(dataset_root / "cgns_types.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"Global": "DataArray_t"}, f)

    flat_cst, variable_schema, constant_schema, cgns_types = (
        common_reader.load_metadata_from_disk(dataset_root)
    )

    loaded = flat_cst["train"]["Global/cst_numeric"]
    assert isinstance(loaded, np.memmap)
    assert np.array_equal(np.asarray(loaded), data)
    assert variable_schema["Global/var"]["dtype"] == "float32"
    assert "Global/cst_numeric" in constant_schema["train"]
    assert cgns_types["Global"] == "DataArray_t"


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


# %% Fixtures
@pytest.fixture()
def dataset(samples) -> Dataset:
    samples_ = []
    for i, sample in enumerate(samples):
        sample_ = deepcopy(sample)
        if i == 0 or i == 2:
            sample_.add_global("toto", 1.0)
        samples_.append(sample_)

    dataset = Dataset()
    dataset.get_backend().add_sample(sample=samples_)
    return dataset


@pytest.fixture()
def main_splits() -> dict:
    return {"train": [0, 2], "test": [1, 3]}


@pytest.fixture()
def problem_definition(main_splits) -> ProblemDefinition:
    problem_definition = ProblemDefinition()
    problem_definition.task = "regression"
    problem_definition.add_in_features_identifiers(["feature_name_1", "feature_name_2"])
    problem_definition.train_split = {"train": main_splits["train"]}
    problem_definition.test_split = {"test": main_splits["test"]}
    return problem_definition


@pytest.fixture()
def sample_constructor(dataset):
    """A simple function that takes an id and returns a Sample."""

    def _sample_constructor(id):
        return dataset[id]

    return _sample_constructor


@pytest.fixture()
def split_ids(problem_definition) -> dict:
    return {
        "train": problem_definition.get_train_split_indices(),
        "test": problem_definition.get_test_split_indices(),
    }


class Test_Storage:
    def assert_sample(self, sample):
        assert isinstance(sample, Sample)
        sorted_names = sorted(sample.get_global_names())
        for i in range(4):
            assert sorted_names[i] == f"global_{i}"
        assert "test_field_same_size" in sample.get_field_names()
        assert sample.get_field("test_field_same_size").shape[0] == 17

    # ------------------------------------------------------------------------------
    #     HUGGING FACE BRIDGE (with tree flattening and pyarrow tables)
    # ------------------------------------------------------------------------------

    def test_hf_datasets(
        self,
        tmp_path,
        sample_constructor,
        split_ids,
        infos,
        problem_definition,
    ):
        import plaid.storage.hf_datasets.bridge as hf_bridge

        test_dir = tmp_path / "test_hf"

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
        )

        with pytest.raises(ValueError):
            save_to_disk(
                output_folder=test_dir,
                sample_constructor=sample_constructor,
                ids=split_ids,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                overwrite=False,
            )

        with pytest.raises(ValueError):
            problem_definition.name = None
            save_to_disk(
                output_folder=test_dir,
                sample_constructor=sample_constructor,
                ids=split_ids,
                backend="hf_datasets",
                infos=infos,
                pb_defs=problem_definition,
                overwrite=True,
            )

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        load_problem_definitions_from_disk(test_dir)
        with pytest.raises(ValueError):
            load_problem_definitions_from_disk("dummy")

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        hf_dataset = datasetdict["train"]
        converter = converterdict["train"]

        print(converter)

        plaid_sample = converter.to_plaid(
            hf_dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        plaid_sample = converter.to_plaid(hf_dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(hf_dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        hf_bridge.to_var_sample_dict(hf_dataset, 0, enforce_shapes=False)

        converter.to_dict(hf_dataset, 0)
        converter.sample_to_dict(hf_dataset[0])

        converter.to_dict(
            hf_dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        converter.to_dict(
            hf_dataset,
            0,
            features=["Base_Name/Zone_Name/VertexFields/test_field_same_size"],
        )
        converter.to_dict(hf_dataset, 0, features=["Global/global_0"])
        with pytest.raises(KeyError):
            converter.to_dict(hf_dataset, 0, features=["dummy"])

    def test_zarr(
        self, tmp_path, sample_constructor, split_ids, infos, problem_definition
    ):
        test_dir = tmp_path / "test_zarr"

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        zarr_dataset = datasetdict["train"]
        converter = converterdict["train"]

        plaid_sample = converter.to_plaid(
            zarr_dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        plaid_sample = converter.to_plaid(zarr_dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(zarr_dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        # coverage of ZarrDataset classe
        for sample in zarr_dataset:
            sample
        len(zarr_dataset)
        zarr_dataset.zarr_group
        zarr_dataset.toto = 1.0
        print(zarr_dataset)

        converter.to_dict(zarr_dataset, 0)
        converter.sample_to_dict(zarr_dataset[0])

        converter.to_dict(
            zarr_dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        with pytest.raises(KeyError):
            converter.to_dict(zarr_dataset, 0, features=["dummy"])

    def test_hf_datasets_indexers(
        self,
        tmp_path,
        sample_constructor,
        split_ids,
        infos,
        problem_definition,
    ):
        test_dir = tmp_path / "test_hf_indexers"

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir)
        hf_dataset = datasetdict["train"]
        converter = converterdict["train"]

        field_path = "Base_Name/Zone_Name/VertexFields/test_field_same_size"
        selected_idx = [1, 3, 7, 11]

        sampled = converter.to_plaid(
            hf_dataset,
            0,
            features=[field_path],
            indexers={field_path: selected_idx},
        )
        full = converter.to_plaid(hf_dataset, 0, features=[field_path])

        expected = full.get_field("test_field_same_size")[selected_idx]
        got = sampled.get_field("test_field_same_size")
        assert np.array_equal(got, expected)

        with pytest.raises(KeyError):
            converter.to_dict(
                hf_dataset,
                0,
                features=[field_path],
                indexers={"dummy": selected_idx},
            )

        # Valid variable feature key, but not among requested features
        other_variable_feature = next(
            f
            for f in converter.variable_features
            if f != field_path and not f.endswith("_times")
        )
        with pytest.raises(KeyError):
            converter.to_dict(
                hf_dataset,
                0,
                features=[field_path],
                indexers={other_variable_feature: [0]},
            )

    def test_zarr_indexers(
        self,
        tmp_path,
        sample_constructor,
        split_ids,
        infos,
        problem_definition,
    ):
        test_dir = tmp_path / "test_zarr_indexers"

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir)
        zarr_dataset = datasetdict["train"]
        converter = converterdict["train"]

        field_path = "Base_Name/Zone_Name/VertexFields/test_field_same_size"
        selected_idx = [0, 2, 4, 8, 16]

        sampled = converter.to_plaid(
            zarr_dataset,
            0,
            features=[field_path],
            indexers={field_path: selected_idx},
        )
        full = converter.to_plaid(zarr_dataset, 0, features=[field_path])

        expected = full.get_field("test_field_same_size")[selected_idx]
        got = sampled.get_field("test_field_same_size")
        assert np.array_equal(got, expected)

        with pytest.raises(IndexError):
            converter.to_dict(
                zarr_dataset,
                0,
                features=[field_path],
                indexers={field_path: [999]},
            )

    def test_zarr_bridge_indexer_branches(
        self,
        tmp_path,
        sample_constructor,
        split_ids,
        infos,
        problem_definition,
    ):
        import plaid.storage.zarr.bridge as zarr_bridge

        test_dir = tmp_path / "test_zarr_bridge_indexers"
        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )
        datasetdict, _ = init_from_disk(test_dir)
        zarr_dataset = datasetdict["train"]

        # cover `continue` on missing feature key
        out = zarr_bridge.to_var_sample_dict(
            zarr_dataset, 0, features=["missing/feature/path"]
        )
        assert out == {}

        # cover slice branch
        arr = np.arange(10)
        sliced = zarr_bridge._apply_indexer(arr, slice(1, 6, 2), "feat")
        assert np.array_equal(sliced, np.array([1, 3, 5]))

        # cover scalar and invalid-shape indexer branches
        with pytest.raises(ValueError):
            zarr_bridge._apply_indexer(np.array(1), [0], "feat")
        with pytest.raises(ValueError):
            zarr_bridge._apply_indexer(np.arange(5), [[0, 1]], "feat")

    def test_hf_bridge_indexer_branches(
        self,
        tmp_path,
        sample_constructor,
        split_ids,
        infos,
        problem_definition,
    ):
        import pyarrow as pa

        import plaid.storage.hf_datasets.bridge as hf_bridge

        test_dir = tmp_path / "test_hf_bridge_indexers"
        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )
        datasetdict, _ = init_from_disk(test_dir)
        hf_dataset = datasetdict["train"]

        field_path = "Base_Name/Zone_Name/VertexFields/test_field_same_size"

        # cover enforce_shapes=False + indexed branch
        out = hf_bridge.to_var_sample_dict(
            hf_dataset,
            0,
            features=[field_path],
            indexers={field_path: [0, 2, 4]},
            enforce_shapes=False,
        )
        assert out[field_path].shape == (3,)

        # cover sample_to_var_sample_dict None branch
        assert hf_bridge.sample_to_var_sample_dict({"a": None}) == {"a": None}

        # cover _extract_indexed_arrow slice / invalid ndim / oob
        primitive = pa.array([0, 1, 2, 3, 4], type=pa.int64())
        assert np.array_equal(
            hf_bridge._extract_indexed_arrow(primitive, slice(1, 4), "f"),
            np.array([1, 2, 3]),
        )
        with pytest.raises(ValueError):
            hf_bridge._extract_indexed_arrow(primitive, [[0, 1]], "f")
        with pytest.raises(IndexError):
            hf_bridge._extract_indexed_arrow(primitive, [99], "f")

        # cover fallback branch (ListArray -> _to_numpy_arrow + _apply_indexer)
        list_arr = pa.array([[1, 2, 3], [4, 5, 6]], type=pa.list_(pa.int64()))
        fallback = hf_bridge._extract_indexed_arrow(list_arr, [0, 2], "f")
        assert np.array_equal(fallback, np.array([[1, 3], [4, 6]]))

        # cover _to_numpy_arrow default and _apply_indexer scalar guard
        assert np.array_equal(
            hf_bridge._to_numpy_arrow(primitive), np.array([0, 1, 2, 3, 4])
        )
        with pytest.raises(ValueError):
            hf_bridge._apply_indexer(np.array(1), [0], "f")

    def test_cgns(
        self, tmp_path, sample_constructor, split_ids, infos, problem_definition
    ):
        test_dir = tmp_path / "test_cgns"

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="cgns",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        cgns_dataset = datasetdict["train"]
        converter = converterdict["train"]

        # coverage of CGNSDataset classe
        for sample in cgns_dataset:
            sample
        len(cgns_dataset)
        cgns_dataset.ids
        cgns_dataset.toto = 1.0
        print(cgns_dataset)

        plaid_sample = converter.to_plaid(cgns_dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(cgns_dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        with pytest.raises(ValueError):
            converter.to_dict(cgns_dataset, 0)
        with pytest.raises(ValueError):
            converter.sample_to_dict(cgns_dataset[0])

    def test_registry(self):
        from plaid.storage import registry

        backends = registry.available_backends()
        assert "hf_datasets" in backends
        assert "zarr" in backends
        assert "cgns" in backends

        hf_module = registry.get_backend("hf_datasets")
        assert hf_module is not None

        zarr_module = registry.get_backend("zarr")
        assert zarr_module is not None

        cgns_module = registry.get_backend("cgns")
        assert cgns_module is not None

        with pytest.raises(ValueError):
            _ = registry.get_backend("non_existent_backend")

    def test_split_list_single_split(self):
        """Cover _split_list early return path (`n_splits <= 1`)."""
        assert _split_list([0, 1, 2], 1) == [[0, 1, 2]]

    # --------------------------------------------------------------------------
    #     New sample_constructor + ids API tests
    # --------------------------------------------------------------------------

    def test_build_gen_kwargs(self):
        """_build_gen_kwargs shards ids for each split."""
        ids = {
            "train": [0, 1, 2, 3],
            "test": [10, 11],
        }

        gen_kwargs = _build_gen_kwargs(ids, num_proc=2)

        # Check gen_kwargs structure
        assert "train" in gen_kwargs
        assert "test" in gen_kwargs
        assert "shards_ids" in gen_kwargs["train"]
        assert "shards_ids" in gen_kwargs["test"]

        # Train should be split into 2 shards
        train_shards = gen_kwargs["train"]["shards_ids"]
        assert len(train_shards) == 2
        all_train_ids = [i for shard in train_shards for i in shard]
        assert sorted(all_train_ids) == [0, 1, 2, 3]

        # Test should be split into 2 shards (2 items, 2 procs)
        test_shards = gen_kwargs["test"]["shards_ids"]
        assert len(test_shards) == 2
        all_test_ids = [i for shard in test_shards for i in shard]
        assert sorted(all_test_ids) == [10, 11]

    def test_sample_constructor_generator(self):
        """_SampleFuncGenerator wraps a function into a generator."""
        collected = []

        def my_func(id_):
            collected.append(id_)
            return id_

        gen = _SampleFuncGenerator(my_func)

        # Test with shards_ids
        results = list(gen(shards_ids=[[0, 1], [2, 3]]))
        assert results == [0, 1, 2, 3]
        assert collected == [0, 1, 2, 3]

    def test_sample_constructor_generator_default_none(self):
        """_SampleFuncGenerator.__call__ with shards_ids=None uses default [[]] path."""
        collected = []

        def my_func(id_):
            collected.append(id_)
            return id_

        gen = _SampleFuncGenerator(my_func)
        # Call with no arguments — shards_ids defaults to None
        results = list(gen())
        assert results == []
        assert collected == []

    def test_save_to_disk_with_sample_constructor(
        self,
        tmp_path,
        sample_constructor,
        split_ids,
        infos,
        problem_definition,
    ):
        """The new sample_constructor + ids API works with sequential execution."""
        test_dir = tmp_path / "test_hf_sample_constructor"

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            num_proc=1,
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir)
        assert "train" in datasetdict
        assert "test" in datasetdict

        # Verify samples are readable
        converter = converterdict["train"]
        plaid_sample = converter.to_plaid(datasetdict["train"], 0)
        self.assert_sample(plaid_sample)

    def test_save_to_disk_raises_on_non_sliceable_ids(
        self,
        tmp_path,
        infos,
        problem_definition,
    ):
        """save_to_disk raises TypeError when ids are not sliceable."""

        def my_func(id_):
            return id_

        with pytest.raises(TypeError, match="must be a sliceable sequence"):
            save_to_disk(
                output_folder=tmp_path / "test_non_sliceable",
                sample_constructor=my_func,
                ids={"train": iter([1, 2, 3])},  # iterator is not sliceable
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                overwrite=True,
            )

    def test_save_to_disk_with_string_ids(
        self,
        dataset,
        tmp_path,
        infos,
        problem_definition,
    ):
        """save_to_disk works with non-integer ids (strings mapped to indices)."""
        id_map = {"sample_a": 0, "sample_b": 2, "sample_c": 1, "sample_d": 3}

        def sample_constructor(str_id):
            return dataset[id_map[str_id]]

        save_to_disk(
            output_folder=tmp_path / "test_string_ids",
            sample_constructor=sample_constructor,
            ids={
                "train": ["sample_a", "sample_b"],
                "test": ["sample_c", "sample_d"],
            },
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        datasetdict, converterdict = init_from_disk(tmp_path / "test_string_ids")
        assert "train" in datasetdict
        assert "test" in datasetdict

    def test_save_to_disk_parallel_auto_sharding(
        self,
        tmp_path,
        sample_constructor,
        split_ids,
        infos,
        problem_definition,
        monkeypatch,
    ):
        """save_to_disk with num_proc > 1 triggers auto-sharding."""
        from unittest.mock import MagicMock

        import plaid.storage.writer as writer_mod

        # Track whether _build_gen_kwargs was called
        original_build = writer_mod._build_gen_kwargs
        build_called = []

        def tracked_build(ids, num_proc):
            build_called.append(True)
            return original_build(ids, num_proc)

        monkeypatch.setattr(writer_mod, "_build_gen_kwargs", tracked_build)

        # Mock preprocess and backend to avoid multiprocessing pickling issues
        monkeypatch.setattr(
            writer_mod,
            "preprocess",
            MagicMock(
                return_value=(
                    {},  # flat_cst
                    {},  # variable_schema
                    {},  # constant_schema
                    {"train": 2, "test": 2},  # num_samples
                    {},  # cgns_types
                )
            ),
        )
        monkeypatch.setattr(writer_mod, "save_metadata_to_disk", MagicMock())
        monkeypatch.setattr(writer_mod, "save_infos_to_disk", MagicMock())
        monkeypatch.setattr(writer_mod, "save_problem_definitions_to_disk", MagicMock())

        backend_mock = MagicMock()
        monkeypatch.setattr(writer_mod, "get_backend", lambda _name: backend_mock)

        test_dir = tmp_path / "test_hf_parallel_auto_shard"

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            num_proc=2,
            overwrite=True,
        )

        # Verify auto-sharding was triggered
        assert len(build_called) == 1

    def test_cgns_generate_no_gen_kwargs(self, tmp_path, dataset):
        """Cover cgns writer else branch: gen_func() called without batch_ids_list."""
        test_dir = tmp_path / "test_cgns_no_kwargs"
        samples_to_yield = [dataset[0], dataset[1]]

        def my_generator():
            yield from samples_to_yield

        generators = {"train": my_generator}

        cgns_generate_datasetdict_to_disk(
            output_folder=test_dir,
            generators=generators,
            gen_kwargs=None,
        )

        data_dir = test_dir / "data" / "train"
        assert data_dir.exists()
        written = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
        assert len(written) == 2
        assert written[0] == "sample_000000000"
        assert written[1] == "sample_000000001"

    def test_zarr_generate_no_gen_kwargs(
        self, tmp_path, sample_constructor, split_ids, infos, problem_definition
    ):
        """Cover zarr writer else branch: gen_func() called without batch_ids_list."""
        # First, save a dataset normally to get the variable_schema
        ref_dir = tmp_path / "ref_zarr"
        save_to_disk(
            output_folder=ref_dir,
            sample_constructor=sample_constructor,
            ids=split_ids,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        # Load the variable_schema from the saved metadata
        variable_schema_path = ref_dir / "variable_schema.yaml"
        with open(variable_schema_path) as f:
            variable_schema = yaml.safe_load(f)

        # Now call zarr generate_datasetdict_to_disk directly without gen_kwargs
        test_dir = tmp_path / "test_zarr_no_kwargs"
        samples_to_yield = [
            sample_constructor(split_ids["train"][0]),
            sample_constructor(split_ids["train"][1]),
        ]

        def my_generator():
            yield from samples_to_yield

        generators = {"train": my_generator}

        zarr_generate_datasetdict_to_disk(
            output_folder=test_dir,
            generators=generators,
            variable_schema=variable_schema,
            gen_kwargs=None,
        )

        import zarr

        data_dir = test_dir / "data" / "train"
        root = zarr.open_group(str(data_dir), mode="r")
        sample_groups = sorted(root.keys())
        assert len(sample_groups) == 2
        assert sample_groups[0] == "sample_000000000"
        assert sample_groups[1] == "sample_000000001"

    def test_cgns_generate_parallel(
        self, tmp_path, dataset, split_ids, infos, problem_definition
    ):
        """Cover cgns writer parallel branch with num_proc=2."""
        test_dir = tmp_path / "test_cgns_parallel"

        # Pre-build samples list so the picklable generator can index into it
        all_samples = [dataset[i] for i in range(len(dataset))]
        gen = _PicklableSampleLookup(all_samples)

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=gen,
            ids=split_ids,
            backend="cgns",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            num_proc=2,
            overwrite=True,
        )

        data_dir = test_dir / "data" / "train"
        assert data_dir.exists()
        written = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
        assert len(written) == 2

    def test_zarr_generate_parallel(
        self, tmp_path, dataset, split_ids, infos, problem_definition
    ):
        """Cover zarr writer parallel branch with num_proc=2."""
        test_dir = tmp_path / "test_zarr_parallel"

        all_samples = [dataset[i] for i in range(len(dataset))]
        gen = _PicklableSampleLookup(all_samples)

        save_to_disk(
            output_folder=test_dir,
            sample_constructor=gen,
            ids=split_ids,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            num_proc=2,
            overwrite=True,
        )

        import zarr

        data_dir = test_dir / "data" / "train"
        root = zarr.open_group(str(data_dir), mode="r")
        sample_groups = sorted(root.keys())
        assert len(sample_groups) == 2
