# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: plaid-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Downloadable examples
#
# This Jupyter Notebook show how to easily retrieve advanced examples online.
#
# 1. Datasets
# 2. Samples

# %%
import warnings
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

# %% [markdown]
# ## Section 1: Datasets

# %% [markdown]
# Retrieving advanded datasets examples is as easy as:

# %%
from plaid.examples import AVAILABLE_EXAMPLES

print(AVAILABLE_EXAMPLES)

# %%
from plaid.examples import datasets
import time

start = time.perf_counter()
print(datasets.tensile2d)
end = time.perf_counter()
print(f"First dataset retrieval duration: {end - start:.6f} seconds")

# %%
start = time.perf_counter()
print(datasets.tensile2d)
end = time.perf_counter()
print(f"Second dataset retrieval duration: {end - start:.6f} seconds")


# %% [markdown]
# ## Section 2: Samples

# %%
from plaid.examples import samples

start = time.perf_counter()
print(samples.vki_ls59)

end = time.perf_counter()
print(f"First sample retrieval duration: {end - start:.6f} seconds")
assert(len(samples.vki_ls59.get_scalar_names())==8)

# %%
from plaid.examples import samples

start = time.perf_counter()
print(samples.tensile2d)

end = time.perf_counter()
print(f"The tensile2d dataset being already loaded: sample retrieval duration: {end - start:.6f} seconds")
assert(len(samples.tensile2d.get_scalar_names())==10)