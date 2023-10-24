# %% [markdown]
# # Interpolation Examples
#
# This Jupyter Notebook demonstrates the usage and functionality of interpolation functions in the PLAID library. It includes examples of:
#
# 1. Piece-wise linear interpolation
# 2. Piece-wise linear interpolation with mapping
# 3. Vectorized interpolation with mapping
# 4. Vectorized interpolation
# 5. Binary Search
#
# This function provides comprehensive examples and tests for interpolation functions, including piece-wise linear interpolation, interpolation with mapping, vectorized interpolation, and binary search.

# %%
# Import required libraries
import numpy as np

# %%
# Import necessary libraries and functions
from plaid.utils.interpolation import (
    binary_search, binary_search_vectorized, piece_wise_linear_interpolation,
    piece_wise_linear_interpolation_vectorized,
    piece_wise_linear_interpolation_vectorized_with_map,
    piece_wise_linear_interpolation_with_map)

# %% [markdown]
# ## Section 1: Piece-wise Linear Interpolation

# %%
# Init example variables
time_indices = np.array([0.0, 1.0, 2.5])
vectors = np.array([np.ones(5), 2.0 * np.ones(5), 3.0 * np.ones(5)])

print(f"{time_indices = }")
print(f"{vectors = }")

# %%
# Test piece-wise linear interpolation for various inputs
result = piece_wise_linear_interpolation(-1.0, time_indices, vectors)
print(f"{result = }")

np.testing.assert_almost_equal(result, [1., 1., 1., 1., 1.])

# %%
result = piece_wise_linear_interpolation(1.0, time_indices, vectors)
print(f"{result = }")

np.testing.assert_almost_equal(result, [2., 2., 2., 2., 2.])

# %%
result = piece_wise_linear_interpolation(0.4, time_indices, vectors)
print(f"{result = }")

np.testing.assert_almost_equal(result, [1.4, 1.4, 1.4, 1.4, 1.4])

# %% [markdown]
# ## Section 2: Piece-wise Linear Interpolation with Mapping

# %%
# Init vectors variables
vectors_map = ["vec1", "vec2", "vec1"]
vectors_dict = {
    "vec1": np.ones(5),
    "vec2": 2.0 * np.ones(5)
}

print(f"{vectors_map = }")
print(f"{vectors_dict = }")

# %%
# Test interpolation with mapping to named vectors
result = piece_wise_linear_interpolation_with_map(
    3.0, time_indices, vectors_dict, vectors_map)
print(f"{result = }")

np.testing.assert_almost_equal(result, [1., 1., 1., 1., 1.])

# %%
result = piece_wise_linear_interpolation_with_map(
    1.0, time_indices, vectors_dict, vectors_map)
print(f"{result = }")

np.testing.assert_almost_equal(result, [2., 2., 2., 2., 2.])

# %%
result = piece_wise_linear_interpolation_with_map(
    0.6, time_indices, vectors_dict, vectors_map)
print(f"{result = }")

np.testing.assert_almost_equal(result, [1.6, 1.6, 1.6, 1.6, 1.6])

# %% [markdown]
# ## Section 3: Vectorized Interpolation with Mapping

# %%
# Init input values
input_values = np.array([-0.1, 2.0, 3.0])

# %%
result = piece_wise_linear_interpolation_vectorized_with_map(
    input_values, time_indices, vectors_dict, vectors_map)
print(f"{result = }")

expected_result = [
    np.array([1., 1., 1., 1., 1.]),
    np.array([1.33333333, 1.33333333, 1.33333333, 1.33333333, 1.33333333]),
    np.array([1., 1., 1., 1., 1.])
]

np.testing.assert_almost_equal(result, expected_result)

# %%
"""
Checks the accuracy of a piecewise linear interpolation function
by comparing its output for a set of input values to a set of precomputed
expected values.
"""

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

# %% [markdown]
# ## Section 4: Vectorized Interpolation

# %%
result = piece_wise_linear_interpolation_vectorized(
        np.array(vals), time_indices, coefficients)

expected_result = [2000000.0, 2000000.0, 2200000.0, 2300000.0, 2400000.0, 2000000.0,
                   2400000.0, 3000000.0, 2500000.0, 2400000.0, 2100000.0, 2800000.0,
                   4000000.0, 3000000.0, 2395574.1913524233]

np.testing.assert_almost_equal(result, expected_result)

# %% [markdown]
# ## Section 5: Binary Search

# %%
test_list = np.array([0.0, 1.0, 2.5, 10.])
val_list = np.array([-1., 11., 0.6, 2.0, 2.6, 9.9, 1.0])

# Apply binary search to find indices for given values within a reference list
ref = np.array([0, 3, 0, 1, 2, 2, 1], dtype=int)
result = binary_search_vectorized(test_list, val_list)

for i, val in enumerate(val_list):
    assert binary_search(test_list, val) == ref[i]
    assert result[i] == ref[i]


