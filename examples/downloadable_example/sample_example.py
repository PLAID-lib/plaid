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
# This Jupyter Notebook show how to easily retrieve sample examples online.

# %%
import warnings
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

# %% [markdown]
# Retrieving sample examples is as easy as:

# %%
from plaid.downloadable_examples import AVAILABLE_EXAMPLES
import time

print(AVAILABLE_EXAMPLES)

# %%
from plaid.downloadable_examples import samples

start = time.perf_counter()
print("samples.vki_ls59:", samples.vki_ls59)
end = time.perf_counter()

print(f"First sample retrieval duration: {end - start:.6f} seconds")
assert(len(samples.vki_ls59.get_global_names())==8)

# %%
start = time.perf_counter()
sample = samples.vki_ls59
end = time.perf_counter()

print(f"Second sample retrieval duration: {end - start:.6f} seconds (in cache)")
print("Done")
