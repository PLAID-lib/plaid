# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from pathlib import Path

import numpy as np
import pytest

import plaid
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.utils.base import ShapeError

# %% Fixtures


@pytest.fixture()
def nb_scalars():
    return 5


@pytest.fixture()
def tabular(nb_samples, nb_scalars):
    return np.random.randn(nb_samples, nb_scalars)


@pytest.fixture()
def scalar_names(nb_scalars):
    return [f"test_scalar_{np.random.randint(1e8, 1e9)}" for _ in range(nb_scalars)]


@pytest.fixture()
def sample(zone_name, base_name):
    sample = Sample()
    sample.init_base(3, 3, base_name)
    sample.init_zone(np.array([0, 0, 0]), zone_name=zone_name, base_name=base_name)
    sample.add_scalar("test_scalar", np.random.randn())
    sample.add_field("test_field_same_size", np.random.randn(17), zone_name, base_name)
    sample.add_field(
        f"test_field_{np.random.randint(1e8, 1e9)}",
        np.random.randn(np.random.randint(10, 20)),
        zone_name,
        base_name,
    )
    return sample


@pytest.fixture
def empty_sample():
    return Sample()


@pytest.fixture()
def empty_dataset():
    return Dataset()


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


@pytest.fixture()
def dataset_with_samples(dataset, samples, infos):
    dataset.add_samples(samples)
    dataset.set_infos(infos)
    return dataset


@pytest.fixture()
def other_dataset_with_samples(other_samples):
    other_dataset = Dataset()
    other_dataset.add_samples(other_samples)
    return other_dataset


def compare_two_samples(sample_1: Sample, sample_2: Sample):
    assert set(sample_1.get_all_mesh_times()) == set(sample_2.get_all_mesh_times())
    assert set(sample_1.get_scalar_names()) == set(sample_2.get_scalar_names())
    assert set(sample_1.get_field_names()) == set(sample_2.get_field_names())
    assert set(sample_1.get_time_series_names()) == set(
        sample_2.get_time_series_names()
    )
    assert np.array_equal(sample_1.get_nodes(), sample_2.get_nodes())
    assert set(sample_1.get_base_names()) == set(sample_2.get_base_names())
    for base_name in sample_1.get_base_names():
        assert set(sample_1.get_zone_names(base_name)) == set(
            sample_2.get_zone_names(base_name)
        )
        for zone_name in sample_1.get_zone_names(base_name):
            assert sample_1.get_zone_type(
                zone_name, base_name
            ) == sample_2.get_zone_type(zone_name, base_name)


# %% Tests


class Test_Dataset:
    # -------------------------------------------------------------------------#
    def test___init__(self, dataset):
        assert len(dataset) == 0
        assert dataset.get_samples() == {}
        with pytest.raises(TypeError):
            dataset.add_sample(None)

    def test___init__load(self, current_directory):
        dataset_path = current_directory / "dataset"
        dataset_already_filled = Dataset(dataset_path)
        assert len(dataset_already_filled) == 3

    def test__init__load_with_str(self, current_directory):
        dataset_path = current_directory / "dataset"
        dataset_already_filled = Dataset(str(dataset_path))
        assert len(dataset_already_filled) == 3

    def test___init__unknown_directory(self, current_directory):
        dataset_path = current_directory / "dataset_unknown"
        with pytest.raises(FileNotFoundError):
            Dataset(dataset_path)

    def test___init__file_provided(self, current_directory):
        dataset_path = current_directory / "bad_dataset_test"
        with pytest.raises(FileNotFoundError):
            Dataset(dataset_path)

    # -------------------------------------------------------------------------#
    def test_get_samples(self, dataset_with_samples, nb_samples):
        dataset_with_samples.get_samples()
        dataset_with_samples.get_samples(np.arange(np.random.randint(2, nb_samples)))
        dataset_with_samples.get_samples(as_list=True)
        dataset_with_samples.get_samples(
            np.arange(np.random.randint(2, nb_samples)), as_list=True
        )

    def test_add_sample(self, dataset, sample):
        assert dataset.add_sample(sample) == 0
        assert len(dataset) == 1

    def test_del_sample_classical(self, dataset, sample):
        for i in range(10):
            assert dataset.add_sample(sample) == i

        assert isinstance(dataset.del_sample(9), Sample)
        assert len(dataset) == 9, dataset._samples.keys()
        assert dataset.get_sample_ids() == list(np.arange(0, 9))

        assert isinstance(dataset.del_sample(3), Sample)
        assert len(dataset) == 8
        assert dataset.get_sample_ids() == list(np.arange(0, 8))

        assert isinstance(dataset.del_sample(0), Sample)
        assert len(dataset) == 7
        assert dataset.get_sample_ids() == list(np.arange(0, 7))

    def test_full_del_sample(self, dataset, sample):
        for i in range(3):
            assert dataset.add_sample(sample) == i

        assert isinstance(dataset.del_sample(2), Sample)
        assert len(dataset) == 2, dataset._samples.keys()
        assert dataset.get_sample_ids() == list(np.arange(0, 2))

        assert isinstance(dataset.del_sample(0), Sample)
        assert len(dataset) == 1
        assert dataset.get_sample_ids() == list(np.arange(0, 1))

        assert isinstance(dataset.del_sample(0), Sample)
        assert len(dataset) == 0
        assert dataset.get_sample_ids() == []

        with pytest.raises(ValueError):
            dataset.del_sample(0)

    def test_on_error_del_sample(self, dataset, sample):
        for i in range(3):
            assert dataset.add_sample(sample) == i

        with pytest.raises(ValueError):
            dataset.del_sample(-1)
        with pytest.raises(ValueError):
            dataset.del_sample(10)

    def test_add_sample_and_id(self, dataset, sample):
        dataset.add_sample(sample, 10)
        assert len(dataset) == 1
        assert list(dataset._samples.keys()) == [10]
        with pytest.raises(ValueError):
            dataset.add_sample(sample, -5)

    def test_add_sample_not_a_sample(self, dataset):
        with pytest.raises(TypeError):
            dataset.add_sample("not_a_sample")
        with pytest.raises(TypeError):
            dataset.add_sample(1)

    def test_add_samples_empty(self, empty_dataset):
        with pytest.raises(ValueError):
            empty_dataset.add_samples([])
        with pytest.raises(ValueError):
            empty_dataset.add_samples([], 1)

    def test_add_samples_empty_with_ids(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            empty_dataset.add_samples([sample], [1, 2, 3])

    def test_add_samples_bad_number_ids_inf(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            samples = [sample, sample, sample]
            empty_dataset.add_samples(samples, [1, 2])

    def test_add_samples_bad_number_ids_supp(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            samples = [sample, sample, sample]
            empty_dataset.add_samples(samples, [1, 2, 3, 4])

    def test_add_samples_with_same_ids(self, empty_dataset, sample):
        with pytest.raises(ValueError):
            samples = [sample, sample, sample]
            empty_dataset.add_samples(samples, [1, 1, 1])

    def test_add_samples_with_ids_good(self, dataset, sample):
        samples = [sample, sample, sample]
        size_before = len(dataset)
        assert dataset.add_samples(samples, [1, 2, 3]) == [1, 2, 3]
        assert size_before + len([1, 2, 3]) == len(dataset)

    def test_add_samples(self, dataset_with_samples, other_samples):
        size_before = len(dataset_with_samples)
        assert (
            dataset_with_samples.add_samples(other_samples)
            == np.arange(size_before, size_before + len(other_samples))
        ).all()

    def test_add_samples_not_a_list_of_samples(self, dataset, sample):
        with pytest.raises(TypeError):
            dataset.add_samples({0: sample})
        with pytest.raises(TypeError):
            dataset.add_samples(["not_a_sample"])

    def test_del_samples_classical(self, dataset, sample):
        for i in range(10):
            assert dataset.add_sample(sample) == i

        assert isinstance(dataset.del_samples([9])[0], Sample)

        list_deleted = dataset.del_samples([0, 2, 4, 6])
        for s in list_deleted:
            assert isinstance(s, Sample)
        assert len(dataset) == 5, dataset._samples.keys()
        assert dataset.get_sample_ids() == list(np.arange(0, 5))

        list_deleted = dataset.del_samples([1, 2, 3])
        for s in list_deleted:
            assert isinstance(s, Sample)
        assert len(dataset) == 2, dataset._samples.keys()
        assert dataset.get_sample_ids() == list(np.arange(0, 2))

        assert isinstance(dataset.del_sample(1), Sample)
        assert len(dataset) == 1, dataset._samples.keys()
        assert dataset.get_sample_ids() == [0]

        list_deleted = dataset.del_samples([0])
        for s in list_deleted:
            assert isinstance(s, Sample)
        assert len(dataset) == 0, dataset._samples.keys()
        assert dataset.get_sample_ids() == []

    def test_full_del_samples(self, dataset, sample):
        for i in range(3):
            assert dataset.add_sample(sample) == i

        list_deleted = dataset.del_samples([0, 1, 2])
        for s in list_deleted:
            assert isinstance(s, Sample)
        assert len(dataset) == 0, dataset._samples.keys()
        assert dataset.get_sample_ids() == []

        with pytest.raises(ValueError):
            dataset.del_samples([0])

    def test_on_error_del_samples(self, dataset, sample):
        for i in range(3):
            assert dataset.add_sample(sample) == i

        with pytest.raises(TypeError):
            dataset.del_samples(1)
        with pytest.raises(ValueError):
            dataset.del_samples([])
        with pytest.raises(ValueError):
            dataset.del_samples([-1, 0, 1])
        with pytest.raises(ValueError):
            dataset.del_samples([0, 1, 10])
        with pytest.raises(ValueError):
            dataset.del_samples([0, 0, 1])

    def test_get_sample_ids(self, dataset):
        dataset.get_sample_ids()

    def test_get_sample_ids_from_disk(self, current_directory):
        dataset_path = current_directory / "dataset"
        assert plaid.get_number_of_samples(dataset_path) == 3

    # -------------------------------------------------------------------------#
    def test_get_scalar_names(self, dataset_with_samples, nb_samples):
        dataset_with_samples.get_scalar_names()
        dataset_with_samples.get_scalar_names(np.random.randint(2, nb_samples, size=2))

    def test_get_scalar_names_same_ids(self, dataset_with_samples):
        dataset_with_samples.get_scalar_names()
        dataset_with_samples.get_scalar_names([0, 0])

    def test_get_field_names(self, dataset_with_samples, nb_samples):
        dataset_with_samples.get_field_names()
        dataset_with_samples.get_field_names(np.random.randint(2, nb_samples, size=2))

    # -------------------------------------------------------------------------#
    def test_add_tabular_scalars(self, dataset, tabular, scalar_names, nb_samples):
        dataset.add_tabular_scalars(tabular, scalar_names)
        assert len(dataset) == nb_samples

    def test_add_tabular_scalars_no_names(self, dataset, tabular, nb_samples):
        dataset.add_tabular_scalars(tabular)
        assert len(dataset) == nb_samples

    def test_add_tabular_scalars_bad_ndim(self, dataset, tabular, scalar_names):
        with pytest.raises(ShapeError):
            dataset.add_tabular_scalars(tabular.reshape((-1)), scalar_names)

    def test_add_tabular_scalars_bad_shape(self, dataset, tabular, scalar_names):
        tabular = np.concatenate((tabular, np.zeros((len(tabular), 1))), axis=1)
        with pytest.raises(ShapeError):
            dataset.add_tabular_scalars(tabular, scalar_names)

    def test_get_scalars_to_tabular(self, dataset, tabular, scalar_names):
        assert len(dataset.get_scalars_to_tabular()) == 0
        assert dataset.get_scalars_to_tabular() == {}
        dataset.add_tabular_scalars(tabular, scalar_names)
        assert dataset.get_scalars_to_tabular(as_nparray=True).shape == (
            len(tabular),
            len(scalar_names),
        )
        dict_tabular = dataset.get_scalars_to_tabular()
        for i_s, sname in enumerate(scalar_names):
            assert np.all(dict_tabular[sname] == tabular[:, i_s])

    def test_get_scalars_to_tabular_same_scalars_name(
        self, dataset, tabular, scalar_names
    ):
        dataset.add_tabular_scalars(tabular, scalar_names)
        assert dataset.get_scalars_to_tabular(as_nparray=True).shape == (
            len(tabular),
            len(scalar_names),
        )
        dataset.get_scalars_to_tabular(sample_ids=[0, 0])
        dataset.get_scalars_to_tabular(scalar_names=["test", "test"])

    # -------------------------------------------------------------------------#
    def test_add_info(self, dataset):
        dataset.add_info("legal", "owner", "PLAID")
        with pytest.raises(KeyError):
            dataset.add_info("illegal_category_key", "owner", "PLAID")
        with pytest.raises(KeyError):
            dataset.add_info("legal", "illegal_info_key", "PLAID")
        dataset.add_info("legal", "owner", "PLAID2")

    def test_add_infos(self, dataset):
        infos = {"legal": {"owner": "CompX", "license": "li_X"}}
        dataset.set_infos(infos)
        new_info = {"type": "simulation", "simulator": "Z-set"}
        dataset.add_infos("data_production", new_info)
        dataset.add_infos("legal", {"owner": "CompY", "license": "li_Y"})
        with pytest.raises(KeyError):
            dataset.add_infos("illegal_category_key", new_info)
        with pytest.raises(KeyError):
            illegal_info = {"z": "simulation", "e": "Z-set"}
            dataset.add_infos("data_production", illegal_info)
        dataset.add_info("legal", "owner", "PLAID2")

    def test_set_infos(self, dataset, infos):
        dataset.set_infos(infos)
        dataset.set_infos(infos)
        with pytest.raises(KeyError):
            dataset.set_infos(
                {"illegal_category_key": {"owner": "PLAID2", "license": "BSD-3"}}
            )
        with pytest.raises(KeyError):
            dataset.set_infos(
                {"legal": {"illegal_info_key": "PLAID2", "license": "BSD-3"}}
            )

    def test_get_infos(self, dataset):
        assert dataset.get_infos() == {}

    def test_print_infos(self, dataset, infos):
        dataset.set_infos(infos)
        dataset.print_infos()

    # -------------------------------------------------------------------------#
    def test_merge_dataset(self, dataset_with_samples, other_dataset_with_samples):
        init_len = len(dataset_with_samples)
        dataset_with_samples.merge_dataset(other_dataset_with_samples)
        assert len(dataset_with_samples) == init_len + len(other_dataset_with_samples)

    def test_merge_dataset_with_None(self, dataset_with_samples):
        dataset_with_samples.merge_dataset(None)

    def test_merge_dataset_with_bad_type(self, dataset_with_samples):
        with pytest.raises(ValueError):
            dataset_with_samples.merge_dataset(3)

    # -------------------------------------------------------------------------#

    def test_from_list_of_samples(self, samples):
        loaded_dataset = Dataset.from_list_of_samples(samples)
        assert len(loaded_dataset) == len(samples)

    # -------------------------------------------------------------------------#

    def test_save(self, dataset_with_samples, tmp_path):
        fname = tmp_path / "test.plaid"
        dataset_with_samples.save(fname)
        assert fname.is_file()

    def test_load(self, dataset_with_samples, tmp_path):
        fname = tmp_path / "test.plaid"
        dataset_with_samples.save(fname)
        new_dataset = Dataset()
        new_dataset.load(fname)
        assert len(new_dataset) == len(dataset_with_samples)
        for sample_1, sample_2 in zip(dataset_with_samples, new_dataset):
            compare_two_samples(sample_1, sample_2)

        n_dataset = Dataset(str(fname))
        assert len(n_dataset) == len(dataset_with_samples)
        for sample_1, sample_2 in zip(dataset_with_samples, n_dataset):
            compare_two_samples(sample_1, sample_2)

    def test_load_multiprocessing(self, dataset_with_samples, tmp_path):
        fname = tmp_path / "test.plaid"
        dataset_with_samples.save(fname)

        new_dataset = Dataset()
        new_dataset.load(fname)
        multi_process_new_dataset = Dataset()
        multi_process_new_dataset.load(fname, processes_number=3)
        assert len(new_dataset) == len(multi_process_new_dataset)
        for sample_1, sample_2 in zip(multi_process_new_dataset, new_dataset):
            compare_two_samples(sample_1, sample_2)

        n_dataset = Dataset(str(fname))
        multi_process_n_dataset = Dataset(str(fname), processes_number=0)
        assert len(n_dataset) == len(multi_process_n_dataset)
        for sample_1, sample_2 in zip(multi_process_n_dataset, n_dataset):
            compare_two_samples(sample_1, sample_2)

        loaded_dataset = Dataset.load_from_file(fname)
        multi_process_loaded_dataset = Dataset.load_from_file(
            fname, processes_number=-1
        )
        assert len(loaded_dataset) == len(multi_process_loaded_dataset)
        for sample_1, sample_2 in zip(multi_process_loaded_dataset, loaded_dataset):
            compare_two_samples(sample_1, sample_2)

    def test_load_process_eror(self, dataset_with_samples, tmp_path):
        fname = tmp_path / "test.plaid"
        dataset_with_samples.save(fname)
        with pytest.raises(ValueError):
            Dataset(str(fname), processes_number=-3)

    def test_load_from_file(self, dataset_with_samples, tmp_path):
        fname = tmp_path / "test_fname.plaid"
        dataset_with_samples.save(fname)
        loaded_dataset = Dataset.load_from_file(fname)
        assert len(loaded_dataset) == len(dataset_with_samples)

    def test_load_from_dir(self, dataset_with_samples, tmp_path):
        dname = tmp_path / "test_dname"
        dataset_with_samples._save_to_dir_(dname)
        loaded_dataset = Dataset.load_from_dir(dname)
        assert len(loaded_dataset) == len(dataset_with_samples)

    # -------------------------------------------------------------------------#
    def test__save_to_dir_(self, dataset_with_samples, tmp_path):
        savedir = tmp_path / "testdir"
        dataset_with_samples._save_to_dir_(savedir)
        assert plaid.get_number_of_samples(savedir) == len(dataset_with_samples)
        assert savedir.is_dir()
        assert (savedir / "infos.yaml").is_file()

    def test__load_from_dir_(self, dataset_with_samples, infos, tmp_path):
        savedir = tmp_path / "testdir"
        dataset_with_samples._save_to_dir_(savedir)
        new_dataset = Dataset()
        new_dataset._load_from_dir_(savedir)
        assert len(new_dataset) == len(dataset_with_samples)
        assert new_dataset.get_infos() == infos
        for sample_1, sample_2 in zip(dataset_with_samples, new_dataset):
            compare_two_samples(sample_1, sample_2)

        new_dataset = Dataset()
        new_dataset._load_from_dir_(savedir, [1, 2])
        assert len(new_dataset) == 2

    # -------------------------------------------------------------------------#
    def test_set_samples(self, dataset, samples):
        dataset.set_samples({i: samp for i, samp in enumerate(samples)})
        dataset.set_samples({i: samp for i, samp in enumerate(samples)})

    def test_set_samples_not_a_dict(self, dataset, samples):
        with pytest.raises(TypeError):
            dataset.set_samples([samp for samp in samples])

    def test_set_samples_not_an_int(self, dataset, sample):
        with pytest.raises(TypeError):
            dataset.set_samples({"0": sample})

    def test_set_samples_not_positive(self, dataset, sample):
        with pytest.raises(ValueError):
            dataset.set_samples({-1: sample})

    def test_set_samples_not_a_sample(self, dataset):
        with pytest.raises(TypeError):
            dataset.set_samples({0: "not_a_sample"})

    def test_set_sample(self, dataset, sample):
        dataset.set_sample(0, sample)
        dataset.set_sample(0, sample)

    def test_set_sample_not_an_int(self, dataset, sample):
        with pytest.raises(TypeError):
            dataset.set_sample("0", sample)

    def test_set_sample_not_positive(self, dataset, sample):
        with pytest.raises(ValueError):
            dataset.set_sample(-1, sample)

    def test_set_sample_not_a_sample(self, dataset):
        with pytest.raises(TypeError):
            dataset.set_sample(0, "not_a_sample")

    # -------------------------------------------------------------------------#
    def test___len__empty(self, dataset):
        assert len(dataset) == 0

    def test___len__(self, dataset_with_samples, nb_samples):
        assert len(dataset_with_samples) == nb_samples

    def test___getitem__empty(self, dataset):
        with pytest.raises(IndexError):
            dataset[0]

    def test___getitem__(self, dataset_with_samples, nb_samples):
        assert isinstance(dataset_with_samples[np.random.randint(nb_samples)], Sample)
        assert isinstance(dataset_with_samples[1 : nb_samples - 1], Dataset)
        assert isinstance(dataset_with_samples[np.arange(1, nb_samples - 1)], Dataset)
        assert isinstance(dataset_with_samples[list(range(1, nb_samples - 1))], Dataset)

    def test___call__empty(self, dataset):
        with pytest.raises(IndexError):
            dataset(0)

    def test___call__(self, dataset_with_samples, nb_samples):
        dataset_with_samples(np.random.randint(nb_samples))

    def test___repr__(self, dataset):
        print(dataset)

    # #-------------------------------------------------------------------------#
    # #-------------------------------------------------------------------------#
    # #-------------------------------------------------------------------------#
    # def test__init__(self, dataset):
    #     pass

    # #---#
    # def test_get_samples(self, dataset_with_samples, nb_samples):
    #     samples = dataset_with_samples.get_samples()
    #     assert(isinstance(samples,dict))
    #     for samp in samples.values():
    #         assert(isinstance(samp,Sample))
    #     assert(len(samples)==nb_samples)
    # def test_get_samples_with_ids(self, dataset_with_samples, nb_samples):
    #     ids = [np.random.randint(0,nb_samples)]
    #     samples = dataset_with_samples.get_samples(ids)
    #     assert(isinstance(samples,dict))
    #     for samp in samples.values():
    #         assert(isinstance(samp,Sample))
    #     assert(len(samples)==len(ids))

    # def test_set_samples(self, dataset, samples):
    #     dataset.set_samples({id:sample for id,sample in enumerate(samples)})
    # def test_set_samples_fail_type(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.set_samples(1)
    # def test_set_samples_fail_type_input(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.set_samples(1)
    # def test_set_samples_fail_type_key(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.set_samples({'test':'test'})
    # def test_set_samples_fail_negative_key(self, dataset):
    #     with pytest.raises(ValueError):
    #         dataset.set_samples({-1:'test'})
    # def test_set_samples_fail_type_value(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.set_samples({0:'test'})

    # def test_set_sample(self, dataset, sample):
    #     dataset.set_sample(id=0,sample=sample)
    # def test_set_sample_already_present(self, dataset_with_samples, sample):
    #     dataset_with_samples.set_sample(id=0,sample=sample)
    # def test_set_sample_fail_id_type(self, dataset, sample):
    #     with pytest.raises(TypeError):
    #         dataset.set_sample(id='0',sample=sample)
    # def test_set_sample_fail_negative_id(self, dataset, sample):
    #     with pytest.raises(ValueError):
    #         dataset.set_sample(id=-1,sample=sample)
    # def test_set_sample_fail_sample_type(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.set_sample(id=1,sample='sample')

    # def test_add_sample(self, dataset, sample):
    #     dataset.add_sample(sample)
    # def test_add_sample_fail_type(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.add_sample(1)

    # def test_add_samples(self, dataset,samples):
    #     dataset.add_samples(samples)
    # def test_add_samples_fail_type(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.add_samples(1)
    # def test_add_samples_fail_sample_type(self, dataset):
    #     with pytest.raises(TypeError):
    #         dataset.add_samples([1])

    # #---#
    # def test_get_sample_ids(self, dataset):
    #     dataset.get_sample_ids()
    # def test_get_sample_ids_type(self, dataset):
    #     dataset.get_sample_ids(feature_type='scalar')
    # def test_get_sample_ids_name(self, dataset):
    #     dataset.get_sample_ids(feature_name='test_scalar2')

    # #---#
    # def test_get_feature_types(self, dataset):
    #     assert(dataset.get_feature_types()==set())
    #     assert(dataset.get_feature_types(feature_name='missing_name')==set())
    # def test_get_feature_names(self, dataset):
    #     assert(dataset.get_feature_names()==set())
    #     assert(dataset.get_feature_names(feature_type='missing_type')==set())

    # #---#
    # def test_get_scalars_to_tabular(self, dataset_with_samples):
    #     dataset_with_samples.get_scalars_to_tabular(feature_names=['test_scalar'])
    # def test_get_scalars_to_tabular_all(self, dataset_with_samples):
    #     dataset_with_samples.get_scalars_to_tabular()
    # def test_get_scalars_to_tabular_dict(self, dataset_with_samples):
    #     dataset_with_samples.get_scalars_to_tabular(feature_names=['test_scalar'])
    # def test_get_scalars_to_tabular_all_dict(self, dataset_with_samples):
    #     dataset_with_samples.get_scalars_to_tabular()
    # def test_get_scalars_to_tabular_missing_name(self, dataset_with_samples):
    #     with pytest.raises(ValueError):
    #         dataset_with_samples.get_scalars_to_tabular(feature_names=['missing_scalar_name'])

    # def test_merge_dataset(self, dataset_with_samples, other_dataset_with_samples):
    #     dataset_with_samples.merge_dataset(other_dataset_with_samples)

    # # def test_save(self, dataset):
    # #     dataset.save()

    # # def test_load(self, dataset):
    # #     dataset.load()

    # # def test__save_to_dir_(self, dataset):
    # #     dataset._save_to_dir_()

    # # def test__load_from_dir_(self, dataset):
    # #     dataset._load_from_dir_()

    # def test___len__(self, dataset):
    #     assert(len(dataset)==0)
    # def test___len__with_samples(self, dataset_with_samples, nb_samples):
    #     assert(len(dataset_with_samples)==nb_samples)

    # def test___getitem__(self, dataset_with_samples):
    #     dataset_with_samples[0]
    # def test___getitem__no_id(self, dataset_with_samples, nb_samples):
    #     with pytest.raises(IndexError):
    #         dataset_with_samples[nb_samples + 1]

    # def test___call__(self, dataset_with_samples):
    #     dataset_with_samples(0)

    # def test___repr__(self, dataset_with_samples):
    #     print(dataset_with_samples)
