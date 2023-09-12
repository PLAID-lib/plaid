# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np

from plaid.utils.interpolation import (
    binary_search, binary_search_vectorized, piece_wise_linear_interpolation,
    piece_wise_linear_interpolation_vectorized,
    piece_wise_linear_interpolation_vectorized_with_map,
    piece_wise_linear_interpolation_with_map)

# %% Functions


def main():

    time_indices = np.array([0.0, 1.0, 2.5])
    vectors = np.array([np.ones(5), 2.0 * np.ones(5), 3.0 * np.ones(5)])
    vectors_dict = {}
    vectors_dict["vec1"] = np.ones(5)
    vectors_dict["vec2"] = 2.0 * np.ones(5)
    vectors_map = ["vec1", "vec2", "vec1"]

    res = piece_wise_linear_interpolation(-1.0, time_indices, vectors)
    np.testing.assert_almost_equal(res, [1., 1., 1., 1., 1.])

    res = piece_wise_linear_interpolation_with_map(
        3.0, time_indices, vectors_dict, vectors_map)
    np.testing.assert_almost_equal(res, [1., 1., 1., 1., 1.])

    res = piece_wise_linear_interpolation(1.0, time_indices, vectors)
    np.testing.assert_almost_equal(res, [2., 2., 2., 2., 2.])

    res = piece_wise_linear_interpolation_with_map(
        1.0, time_indices, vectors_dict, vectors_map)
    np.testing.assert_almost_equal(res, [2., 2., 2., 2., 2.])

    res = piece_wise_linear_interpolation(0.4, time_indices, vectors)
    np.testing.assert_almost_equal(res, [1.4, 1.4, 1.4, 1.4, 1.4])

    res = piece_wise_linear_interpolation(1.4, time_indices, vectors)
    np.testing.assert_almost_equal(
        res, [6.8 / 3, 6.8 / 3, 6.8 / 3, 6.8 / 3, 6.8 / 3])

    res = piece_wise_linear_interpolation_with_map(
        0.6, time_indices, vectors_dict, vectors_map)
    np.testing.assert_almost_equal(res, [1.6, 1.6, 1.6, 1.6, 1.6])

    res = piece_wise_linear_interpolation_vectorized_with_map(
        np.array([-0.1, 2.0, 3.0]), time_indices, vectors_dict, vectors_map)
    np.testing.assert_almost_equal(res, [np.array([1., 1., 1., 1., 1.]), np.array(
        [1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333]), np.array([1., 1., 1., 1., 1.])])

    time_indices = np.array([0., 100., 200., 300., 400., 500., 600., 700.,
                            800., 900., 1000., 2000.])
    coefficients = np.array([2000000., 2200000., 2400000., 2000000., 2400000.,
                            3000000., 2500000., 2400000., 2100000., 2800000.,
                            4000000., 3000000.])

    vals = np.array([-10., 0., 100., 150., 200., 300., 400., 500., 600., 700.,
                    800., 900., 1000., 3000., 701.4752695491923])

    res = np.array([2000000., 2000000., 2200000., 2300000., 2400000., 2000000.,
                    2400000., 3000000., 2500000., 2400000., 2100000., 2800000.,
                    4000000., 3000000., 2395574.19135242])

    for i in range(vals.shape[0]):
        assert (
            piece_wise_linear_interpolation(
                vals[i],
                time_indices,
                coefficients) - res[i]) / res[i] < 1.e-10

    res = piece_wise_linear_interpolation_vectorized(
        np.array(vals), time_indices, coefficients)
    np.testing.assert_almost_equal(res,
                                   [2000000.0,
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
                                    2395574.1913524233])

    test_list = np.array([0.0, 1.0, 2.5, 10.])
    val_list = np.array([-1., 11., 0.6, 2.0, 2.6, 9.9, 1.0])

    ref = np.array([0, 3, 0, 1, 2, 2, 1], dtype=int)
    res = binary_search_vectorized(test_list, val_list)

    for i, val in enumerate(val_list):
        assert binary_search(test_list, val) == ref[i]
        assert res[i] == ref[i]


# %% Main Script
if __name__ == '__main__':
    main()

    print()
    print("#==============#")
    print("#===# DONE #===#")
    print("#==============#")
