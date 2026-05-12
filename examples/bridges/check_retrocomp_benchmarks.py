"""This files serves to check if the main retrieval command in the PLAID Benchmarks
is not returning an error."""

from datasets import load_dataset

from plaid import Dataset
from plaid.storage.common.bridge import to_plaid_sample, to_sample_dict
from plaid.storage.common.reader import load_metadata_from_hub, load_problem_definitions_from_hub
from plaid.storage.hf_datasets.bridge import sample_to_var_sample_dict

repo_id = "PhysArena/Tensile2d"
split_name = "train"

hf_dataset = load_dataset(repo_id, split=split_name, num_proc=1)

flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_hub(repo_id)

plaid_dataset = Dataset()
for hf_sample in hf_dataset:
    var_sample_dict = sample_to_var_sample_dict(hf_sample)
    sample_dict = to_sample_dict(var_sample_dict, flat_cst[split_name], cgns_types)
    sample = to_plaid_sample(sample_dict, cgns_types)
    plaid_dataset.get_backend().add_sample(sample)

pb_defs = load_problem_definitions_from_hub(repo_id)
pb_def = next(iter(pb_defs.values()))

ids_train = pb_def.get_train_split_indices()
sample_train_0 = plaid_dataset[0]