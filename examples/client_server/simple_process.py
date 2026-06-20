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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Import required libraries

from typing import Any
import time
import sys
import numpy as np
from matplotlib import pyplot as plt

from plaid import Sample
from plaid.utils.process_client import PlaidClient

# %% [markdown]
# # Connecion to the plaid server

# %%
plaidserver = PlaidClient(host="localhost", port=8000)
if plaidserver.check_connection():
    print("Connection to PLAID server successful.")
else:
    print("Failed to connect to PLAID server.")
    sys.exit()

pb = plaidserver.problem_definition()
print(pb)

infos = plaidserver.infos()
print(infos)

# %% [markdown]
# # Load a sample for modification

# %%
#load a sample from a distant storage
#from plaid.downloadable_examples import samples
#sample = samples.tensile2d
#...

# from a local disk
#from plaid.storage.reader import init_from_disk, load_infos_from_disk
#from plaid.storage.common.reader import load_problem_definitions_from_disk
#datasetdict, converterdict = init_from_disk("../Datasets/Tensile2d/")
#infos = load_infos_from_disk("../Datasets/Tensile2d/")
#pds = load_problem_definitions_from_disk("../Datasets/Tensile2d/")
#input_features = [x for x in pds['PLAID_benchmark'].input_features if x.startswith("Global")]
#output_features = [x for x in pb["output_features"] if x.startswith("Base")]
#sample: Sample = converterdict["test"].to_plaid(datasetdict["test"], 1)

# from the server
sample: Sample = plaidserver.samples(sample_ids=[0], split=pb['training_split'][0])[0]
input_features = [x for x in pb["input_features"] if x.startswith("Global")]
output_features = [x for x in pb["output_features"] if x.startswith("Base")]
print(sample)

# %% [markdown]
# # Select active feature

# %%

#create a const function to encapsulate the process call
print(f"{input_features=}")
active_input_feature = input_features[0]
print(f"{active_input_feature=}")

print(f"{output_features=}")
active_output_feature = output_features[0]
print(f"{active_output_feature=}")

minmax = {}
minmax["Global/P"] = (-49.99 ,-40.01)
minmax["Global/p1"] = (10.01, 19.99)
minmax["Global/p2"] = (300.3, 599.7)
minmax["Global/p3"] = (1001.0, 1999.0)
minmax["Global/p4"] = (1001.0, 1999.0)
minmax["Global/p5"] = (50050.0, 99950.0)

# %% [markdown]
# # Define the const function
# in this case the function return the value of the active output

# %%

def cost_fuction(x: Any) -> float :
    # 1) here we recover the current optimisation point and map it to the sample.
    for f,v in zip([active_input_feature], x):
        sample.update_value_by_path(f,v)

    # sample.show_tree()

    # # 2) Then send the sample for evaluation/process and recover the sample
    # for f in output_features :
    #     if f.startswith("Global"):
    #         global_name = f.strip("Global/")
    #         if global_name in sample.get_global_names() :
    #             sample.del_global(global_name)
    #     else:
    #         sample.del_feature_by_path(f)

    response: Sample = plaidserver.process(sample)

    # 3) evaluate the cost function
    output: float = np.mean(response.get_feature_by_path(active_output_feature))
    return output

# %% [markdown]
# # Evaluate the cost function at one point

# %%
print(cost_fuction([-45]))

# %% [markdown]
# # Call the process for a range of

# %%
stime = time.time()
nb_calls = 50
x = np.empty(nb_calls)
y = np.empty(nb_calls)

for i,v in enumerate(np.linspace(minmax[active_input_feature][0],minmax[active_input_feature][1],nb_calls)):
    x[i] = v
    #sample.del_global(active_output_features)
    y[i] = cost_fuction([v])
print(f"{nb_calls} calls of process in {time.time()-stime} s")

# %% [markdown]
# # Plot output

# %%

plt.scatter(x,y)
plt.xlabel(active_input_feature)
plt.ylabel(active_output_feature)
plt.title(f"{active_input_feature} vs {active_output_feature}")
plt.grid()
plt.show()
# %%
