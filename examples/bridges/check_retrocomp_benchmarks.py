"""This files serves to check if the main retrieval command in the PLAID Benchmarks
is not returning an error."""

from plaid.bridges import huggingface_bridge

hf_dataset = huggingface_bridge.load_dataset_from_hub(
    f"PLAID-datasets/Tensile2d", split="all_samples[:5]", num_proc=1
)

plaid_dataset, pb_def = huggingface_bridge.huggingface_dataset_to_plaid(
    hf_dataset, processes_number=1, verbose=True
)

ids_train = pb_def.get_split('train_500')
sample_train_0 = plaid_dataset[ids_train[0]]