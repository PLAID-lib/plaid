# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import numpy as np
import pytest

from plaid.utils.interpolation import (
    binary_search,
    binary_search_vectorized,
    piece_wise_linear_interpolation,
    piece_wise_linear_interpolation_vectorized,
    piece_wise_linear_interpolation_vectorized_with_map,
    piece_wise_linear_interpolation_with_map,
)


@pytest.fixture()
def time_indices():
    return np.array([0.0, 1.0, 2.5])


@pytest.fixture()
def vectors():
    return np.array([np.ones(5), 2.0 * np.ones(5), 3.0 * np.ones(5)])


@pytest.fixture()
def vectors_map():
    return ["vec1", "vec2", "vec1"]


@pytest.fixture()
def vectors_dict():
    return {"vec1": np.ones(5), "vec2": 2.0 * np.ones(5)}


@pytest.fixture()
def input_values():
    return np.array([-0.1, 2.0, 3.0])


@pytest.fixture()
def time_indices_bis():
    return np.array(
        [
            0.0,
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            1000.0,
            2000.0,
        ]
    )


@pytest.fixture()
def coefficients():
    return np.array(
        [
            2000000.0,
            2200000.0,
            2400000.0,
            2000000.0,
            2400000.0,
            3000000.0,
            2500000.0,
            2400000.0,
            2100000.0,
            2800000.0,
            4000000.0,
            3000000.0,
        ]
    )


@pytest.fixture()
def vals():
    return np.array(
        [
            -10.0,
            0.0,
            100.0,
            150.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            1000.0,
            3000.0,
            701.4752695491923,
        ]
    )


class Test_sinterpolation:
    def test_piece_wise_linear_interpolation_1(self, time_indices, vectors):
        result = piece_wise_linear_interpolation(-1.0, time_indices, vectors)
        np.testing.assert_almost_equal(result, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_piece_wise_linear_interpolation_2(self, time_indices, vectors):
        result = piece_wise_linear_interpolation(1.0, time_indices, vectors)
        np.testing.assert_almost_equal(result, [2.0, 2.0, 2.0, 2.0, 2.0])

    def test_piece_wise_linear_interpolation_3(self, time_indices, vectors):
        result = piece_wise_linear_interpolation(0.4, time_indices, vectors)
        np.testing.assert_almost_equal(result, [1.4, 1.4, 1.4, 1.4, 1.4])

    def test_piece_wise_linear_interpolation_with_map_1(
        self, time_indices, vectors_map, vectors_dict
    ):
        result = piece_wise_linear_interpolation_with_map(
            3.0, time_indices, vectors_dict, vectors_map
        )
        np.testing.assert_almost_equal(result, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_piece_wise_linear_interpolation_with_map_2(
        self, time_indices, vectors_map, vectors_dict
    ):
        result = piece_wise_linear_interpolation_with_map(
            1.0, time_indices, vectors_dict, vectors_map
        )
        np.testing.assert_almost_equal(result, [2.0, 2.0, 2.0, 2.0, 2.0])

    def test_piece_wise_linear_interpolation_with_map_3(
        self, time_indices, vectors_map, vectors_dict
    ):
        result = piece_wise_linear_interpolation_with_map(
            0.6, time_indices, vectors_dict, vectors_map
        )
        np.testing.assert_almost_equal(result, [1.6, 1.6, 1.6, 1.6, 1.6])

    def test_piece_wise_linear_interpolation_vectorized_with_map(
        self, input_values, time_indices, vectors_dict, vectors_map
    ):
        result = piece_wise_linear_interpolation_vectorized_with_map(
            input_values, time_indices, vectors_dict, vectors_map
        )

        expected_result = [
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ]

        np.testing.assert_almost_equal(result, expected_result)

    def test_piece_wise_linear_interpolation_loop(
        self, time_indices_bis, coefficients, vals
    ):
        expected_result = np.array(
            [
                2000000.0,
                2000000.0,
                2200000.0,
                2300000.0,
                2400000.0,
                2000000.0,
                2400000.0,
                3000000.0,
                2500000.0,
                2400000.0,
                2100000.0,
                2800000.0,
                4000000.0,
                3000000.0,
                2395574.19135242,
            ]
        )

        for i in range(vals.shape[0]):
            assert (
                piece_wise_linear_interpolation(vals[i], time_indices_bis, coefficients)
                - expected_result[i]
            ) / expected_result[i] < 1.0e-10

    def test_piece_wise_linear_interpolation_vectorized(
        self, time_indices_bis, coefficients, vals
    ):
        result = piece_wise_linear_interpolation_vectorized(
            np.array(vals), time_indices_bis, coefficients
        )

        expected_result = [
            2000000.0,
            2000000.0,
            2200000.0,
            2300000.0,
            2400000.0,
            2000000.0,
            2400000.0,
            3000000.0,
            2500000.0,
            2400000.0,
            2100000.0,
            2800000.0,
            4000000.0,
            3000000.0,
            2395574.1913524233,
        ]

        np.testing.assert_almost_equal(result, expected_result)

    def test_binary_search(self):
        test_list = np.array([0.0, 1.0, 2.5, 10.0])
        val_list = np.array([-1.0, 11.0, 0.6, 2.0, 2.6, 9.9, 1.0])

        # Apply binary search to find indices for given values within a reference list
        ref = np.array([0, 3, 0, 1, 2, 2, 1], dtype=int)
        result = binary_search_vectorized(test_list, val_list)

        for i, val in enumerate(val_list):
            assert binary_search(test_list, val) == ref[i]
            assert result[i] == ref[i]
