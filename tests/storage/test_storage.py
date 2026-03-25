# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import json
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable

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
from plaid.storage.hf_datasets.bridge import (
    to_var_sample_dict,
)
from plaid.storage.writer import (
    _build_gen_kwargs_from_partials,
    _extract_ids_from_partial,
    _ShardGenerator,
    _split_list,
)


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
def dataset(samples, infos) -> Dataset:
    samples_ = []
    for i, sample in enumerate(samples):
        sample_ = deepcopy(sample)
        if i == 0 or i == 2:
            sample_.add_scalar("toto", 1.0)
        samples_.append(sample_)

    dataset = Dataset(samples=samples_)
    dataset.set_infos(infos)
    return dataset


@pytest.fixture()
def main_splits() -> dict:
    return {"train": [0, 2], "test": [1, 3]}


@pytest.fixture()
def problem_definition(main_splits) -> ProblemDefinition:
    problem_definition = ProblemDefinition()
    problem_definition.set_task("regression")
    problem_definition.add_input_scalars_names(["feature_name_1", "feature_name_2"])
    problem_definition.set_split(main_splits)
    return problem_definition


@pytest.fixture()
def generator_split(dataset, problem_definition) -> dict[str, Callable]:
    generators_ = {}

    main_splits = problem_definition.get_split()

    for split_name, ids in main_splits.items():

        def generator_(ids):
            for id in ids:
                yield dataset[id]

        generators_[split_name] = partial(generator_, ids)

    return generators_


class Test_Storage:
    def assert_sample(self, sample):
        assert isinstance(sample, Sample)
        sorted_names = sorted(sample.get_scalar_names())
        for i in range(4):
            assert sorted_names[i] == f"global_{i}"
        assert "test_field_same_size" in sample.get_field_names()
        assert sample.get_field("test_field_same_size").shape[0] == 17

    # ------------------------------------------------------------------------------
    #     HUGGING FACE BRIDGE (with tree flattening and pyarrow tables)
    # ------------------------------------------------------------------------------

    def test_hf_datasets(
        self,
        dataset,
        tmp_path,
        generator_split,
        infos,
        problem_definition,
    ):
        test_dir = tmp_path / "test_hf"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
        )

        with pytest.raises(ValueError):
            save_to_disk(
                output_folder=test_dir,
                generators=generator_split,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                overwrite=False,
            )

        with pytest.raises(ValueError):
            problem_definition.set_name(None)
            save_to_disk(
                output_folder=test_dir,
                generators=generator_split,
                backend="hf_datasets",
                infos=infos,
                pb_defs=problem_definition,
                overwrite=True,
            )

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
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

        dataset = datasetdict["train"]
        converter = converterdict["train"]

        print(converter)

        plaid_sample = converter.to_plaid(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        plaid_sample = converter.to_plaid(dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        to_var_sample_dict(dataset, 0, enforce_shapes=False)

        for t in plaid_sample.get_all_time_values():
            for path in problem_definition.get_in_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)
            for path in problem_definition.get_out_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)

        converter.to_dict(dataset, 0)
        converter.sample_to_dict(dataset[0])

        converter.to_dict(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        converter.to_dict(
            dataset,
            0,
            features=["Base_Name/Zone_Name/VertexFields/test_field_same_size"],
        )
        converter.to_dict(dataset, 0, features=["Global/global_0"])
        with pytest.raises(KeyError):
            converter.to_dict(dataset, 0, features=["dummy"])

    def test_zarr(self, tmp_path, generator_split, infos, problem_definition):
        test_dir = tmp_path / "test_zarr"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="zarr",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        dataset = datasetdict["train"]
        converter = converterdict["train"]

        plaid_sample = converter.to_plaid(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        plaid_sample = converter.to_plaid(dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        # coverage of ZarrDataset classe
        for sample in dataset:
            sample
        len(dataset)
        dataset.zarr_group
        dataset.toto = 1.0
        print(dataset)

        for t in plaid_sample.get_all_time_values():
            for path in problem_definition.get_in_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)
            for path in problem_definition.get_out_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)

        converter.to_dict(dataset, 0)
        converter.sample_to_dict(dataset[0])

        converter.to_dict(
            dataset,
            0,
            features=[
                "Base_Name/Zone_Name/VertexFields/test_field_same_size",
                "Global/global_0",
            ],
        )
        with pytest.raises(KeyError):
            converter.to_dict(dataset, 0, features=["dummy"])

    def test_cgns(self, tmp_path, generator_split, infos, problem_definition):
        test_dir = tmp_path / "test_cgns"

        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
            backend="cgns",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
            verbose=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir, splits=["train"])
        datasetdict, converterdict = init_from_disk(test_dir)

        dataset = datasetdict["train"]
        converter = converterdict["train"]

        # coverage of CGNSDataset classe
        for sample in dataset:
            sample
        len(dataset)
        dataset.ids
        dataset.toto = 1.0
        print(dataset)

        plaid_sample = converter.to_plaid(dataset, 0)
        self.assert_sample(plaid_sample)
        plaid_sample = converter.sample_to_plaid(dataset[0])
        self.assert_sample(plaid_sample)

        converter.plaid_to_dict(plaid_sample)

        for t in plaid_sample.get_all_time_values():
            for path in problem_definition.get_in_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)
            for path in problem_definition.get_out_features_identifiers():
                plaid_sample.get_feature_by_path(path=path, time=t)

        with pytest.raises(ValueError):
            converter.to_dict(dataset, 0)
        with pytest.raises(ValueError):
            converter.sample_to_dict(dataset[0])

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
    #     Partial-based API tests
    # --------------------------------------------------------------------------

    def test_build_gen_kwargs_from_partials(self):
        """_build_gen_kwargs_from_partials shards data and creates shard generators."""
        collected = []

        def my_gen(ids):
            for i in ids:
                collected.append(i)
                yield i

        generators = {
            "train": partial(my_gen, [0, 1, 2, 3]),
            "test": partial(my_gen, [10, 11]),
        }

        wrapped_gens, gen_kwargs = _build_gen_kwargs_from_partials(
            generators, num_proc=2
        )

        # Check gen_kwargs structure
        assert "train" in gen_kwargs
        assert "test" in gen_kwargs
        assert "shards_ids" in gen_kwargs["train"]
        assert "shards_ids" in gen_kwargs["test"]

        # Train should be split into 2 shards
        train_shards = gen_kwargs["train"]["shards_ids"]
        assert len(train_shards) == 2
        # All original ids should be present across shards
        all_train_ids = [i for shard in train_shards for i in shard]
        assert sorted(all_train_ids) == [0, 1, 2, 3]

        # Test should be split into 2 shards (2 items, 2 procs)
        test_shards = gen_kwargs["test"]["shards_ids"]
        assert len(test_shards) == 2
        all_test_ids = [i for shard in test_shards for i in shard]
        assert sorted(all_test_ids) == [10, 11]

        # Wrapped generators should be callable with shards_ids kwarg
        collected.clear()
        results = list(wrapped_gens["train"](shards_ids=train_shards))
        assert sorted(results) == [0, 1, 2, 3]
        assert sorted(collected) == [0, 1, 2, 3]

    def test_build_gen_kwargs_from_partials_raises_on_non_partial(self):
        """_build_gen_kwargs_from_partials raises TypeError for non-partial generators."""

        def plain_gen(ids):
            yield from ids

        generators = {"train": plain_gen}

        with pytest.raises(TypeError, match="must be a functools.partial"):
            _build_gen_kwargs_from_partials(generators, num_proc=2)

    def test_hf_datasets_sequential_with_partial_generators(
        self,
        tmp_path,
        generator_split,
        infos,
        problem_definition,
    ):
        """The new partial-based API works with sequential execution (num_proc=1)."""
        test_dir = tmp_path / "test_hf_partial_sequential"

        # Sequential: no auto-sharding triggered, partial generators used directly
        save_to_disk(
            output_folder=test_dir,
            generators=generator_split,
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

    def test_save_to_disk_raises_on_non_partial(
        self,
        tmp_path,
        infos,
        problem_definition,
    ):
        """save_to_disk raises TypeError when generators are not functools.partial."""

        def plain_gen(ids):
            yield from ids

        generators = {
            "train": plain_gen,
            "test": plain_gen,
        }

        with pytest.raises(TypeError, match="must be a functools.partial"):
            save_to_disk(
                output_folder=tmp_path / "test_non_partial",
                generators=generators,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                num_proc=1,
                overwrite=True,
            )

    def test_save_to_disk_raises_on_partial_no_args(
        self,
        tmp_path,
        infos,
        problem_definition,
    ):
        """save_to_disk raises TypeError when partial has no identifiable ids."""
        generators = {
            "train": partial(lambda: None),
        }

        with pytest.raises(TypeError, match="no identifiable ids argument"):
            save_to_disk(
                output_folder=tmp_path / "test_partial_no_args",
                generators=generators,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                overwrite=True,
            )

    def test_save_to_disk_raises_on_non_sliceable_first_arg(
        self,
        tmp_path,
        infos,
        problem_definition,
    ):
        """save_to_disk raises TypeError when first arg of partial is not sliceable."""
        generators = {
            "train": partial(lambda _x: None, 42),  # int is not sliceable
        }

        with pytest.raises(TypeError, match="must be a sliceable sequence"):
            save_to_disk(
                output_folder=tmp_path / "test_non_sliceable",
                generators=generators,
                backend="hf_datasets",
                infos=infos,
                pb_defs={"pb_def": problem_definition},
                overwrite=True,
            )

    # --------------------------------------------------------------------------
    #     Keyword-arg partial support
    # --------------------------------------------------------------------------

    def test_extract_ids_from_partial_positional(self):
        """_extract_ids_from_partial extracts ids from positional arg."""
        data = [1, 2, 3]

        def my_gen(ids):
            yield from ids

        gen = partial(my_gen, data)
        result_data, base_func, extra_args, extra_kwargs = _extract_ids_from_partial(
            gen, "train"
        )
        assert result_data == data
        assert base_func is my_gen
        assert extra_args == ()
        assert extra_kwargs == {}

    def test_extract_ids_from_partial_keyword(self):
        """_extract_ids_from_partial extracts ids from keyword arg matching first param."""
        data = [1, 2, 3]

        def my_gen(ids):
            yield from ids

        gen = partial(my_gen, ids=data)
        result_data, base_func, extra_args, extra_kwargs = _extract_ids_from_partial(
            gen, "train"
        )
        assert result_data == data
        assert base_func is my_gen
        assert extra_args == ()
        assert extra_kwargs == {}

    def test_extract_ids_from_partial_keyword_with_extra_kwargs(self):
        """_extract_ids_from_partial strips ids from kwargs and keeps the rest."""
        data = [1, 2, 3]

        def my_gen(ids, _extra_param=10):
            yield from ids

        gen = partial(my_gen, ids=data, _extra_param=42)
        result_data, base_func, extra_args, extra_kwargs = _extract_ids_from_partial(
            gen, "train"
        )
        assert result_data == data
        assert base_func is my_gen
        assert extra_args == ()
        assert extra_kwargs == {"_extra_param": 42}

    def test_extract_ids_from_partial_keyword_non_sliceable(self):
        """_extract_ids_from_partial raises TypeError when keyword arg is not sliceable.

        Covers line 107 of writer.py.
        """

        def my_gen(ids):
            yield from ids

        # Pass an int as keyword arg — has no __getitem__/__len__
        gen = partial(my_gen, ids=42)
        with pytest.raises(TypeError, match="must be a sliceable sequence"):
            _extract_ids_from_partial(gen, "train")

    def test_shard_generator_default_shards_ids_none(self):
        """_ShardGenerator.__call__ with shards_ids=None uses default [[]] path.

        Covers line 151 of writer.py.
        """
        collected = []

        def my_gen(ids):
            for i in ids:
                collected.append(i)
                yield i

        shard_gen = _ShardGenerator(my_gen)
        # Call with no arguments — shards_ids defaults to None, triggering line 151
        results = list(shard_gen())
        # With shards_ids=[[]], the base_func is called once with an empty list
        assert results == []
        assert collected == []

    def test_save_to_disk_parallel_auto_sharding(
        self,
        tmp_path,
        generator_split,
        infos,
        problem_definition,
        monkeypatch,
    ):
        """save_to_disk with num_proc > 1 triggers auto-sharding from partials.

        Covers line 313 of writer.py.
        """
        from unittest.mock import MagicMock

        import plaid.storage.writer as writer_mod

        # Track whether _build_gen_kwargs_from_partials was called
        original_build = writer_mod._build_gen_kwargs_from_partials
        build_called = []

        def tracked_build(generators, num_proc):
            build_called.append(True)
            return original_build(generators, num_proc)

        monkeypatch.setattr(
            writer_mod, "_build_gen_kwargs_from_partials", tracked_build
        )

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
            generators=generator_split,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            num_proc=2,
            overwrite=True,
        )

        # Verify auto-sharding was triggered (line 313)
        assert len(build_called) == 1

    def test_save_to_disk_with_keyword_partial(
        self,
        dataset,
        tmp_path,
        infos,
        problem_definition,
    ):
        """save_to_disk works with keyword-arg partial syntax."""

        def generator_(ids):
            for id in ids:
                yield dataset[id]

        main_splits = problem_definition.get_split()
        generators = {
            split_name: partial(generator_, ids=ids)
            for split_name, ids in main_splits.items()
        }

        test_dir = tmp_path / "test_hf_keyword_partial"
        save_to_disk(
            output_folder=test_dir,
            generators=generators,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            overwrite=True,
        )

        datasetdict, converterdict = init_from_disk(test_dir)
        assert "train" in datasetdict
        assert "test" in datasetdict

        converter = converterdict["train"]
        plaid_sample = converter.to_plaid(datasetdict["train"], 0)
        self.assert_sample(plaid_sample)
