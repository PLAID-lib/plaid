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
from typing import Callable, cast

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
    _extract_partial_data,
    _split_list,
)
from plaid.types import IndexType


def _yield_samples_from_shards_ids(dataset: Dataset, shards_ids):
    for ids in shards_ids:
        if isinstance(ids, int):
            ids = [ids]
        for id_ in ids:
            yield dataset[id_]


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
def generator(dataset) -> Callable:
    def generator_():
        for sample in dataset:
            yield sample

    return generator_


@pytest.fixture()
def gen_kwargs(problem_definition) -> dict[str, dict]:
    gen_kwargs = {}
    for split_name, ids in problem_definition.get_split().items():
        mid = len(ids) // 2
        gen_kwargs[split_name] = {"shards_ids": [ids[:mid], ids[mid:]]}
    return gen_kwargs


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


@pytest.fixture()
def generator_split_with_kwargs(dataset, gen_kwargs) -> dict[str, Callable]:
    generators_ = {}

    for split_name in gen_kwargs.keys():
        generators_[split_name] = partial(_yield_samples_from_shards_ids, dataset)

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
        generator_split_with_kwargs,
        infos,
        problem_definition,
        gen_kwargs,
    ):
        test_dir = tmp_path / "test_hf"
        legacy_kwargs_test_dir = tmp_path / "test_hf_legacy_kwargs"

        save_to_disk(
            output_folder=legacy_kwargs_test_dir,
            generators=generator_split_with_kwargs,
            backend="hf_datasets",
            infos=infos,
            pb_defs={"pb_def": problem_definition},
            gen_kwargs=gen_kwargs,
            num_proc=2,
            overwrite=True,
            verbose=True,
        )

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
        ids = cast(list[IndexType], [0, 1, 2])
        assert _split_list(ids, 1) == [[0, 1, 2]]

    # --------------------------------------------------------------------------
    #     New partial-based API tests
    # --------------------------------------------------------------------------

    def test_extract_partial_data_with_partial(self):
        """_extract_partial_data returns the first arg of a functools.partial."""
        data = [1, 2, 3]
        gen = partial(lambda ids: (x for x in ids), data)
        result = _extract_partial_data(gen)
        assert result == data

    def test_extract_partial_data_with_non_partial(self):
        """_extract_partial_data returns None for a plain function."""

        def gen(ids):
            yield from ids

        result = _extract_partial_data(gen)
        assert result is None

    def test_extract_partial_data_with_partial_no_args(self):
        """_extract_partial_data returns None for a partial with no positional args."""
        gen = partial(lambda: None)
        result = _extract_partial_data(gen)
        assert result is None

    def test_extract_partial_data_with_non_sliceable(self):
        """_extract_partial_data returns None when first arg is not sliceable."""
        gen = partial(lambda _x: None, 42)  # int has no __getitem__
        result = _extract_partial_data(gen)
        assert result is None

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
        """save_to_disk raises TypeError when partial has no positional arguments."""
        generators = {
            "train": partial(lambda: None),
        }

        with pytest.raises(TypeError, match="no positional arguments"):
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
