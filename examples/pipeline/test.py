from datasets import load_from_disk
from plaid.containers.sample import Sample
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition



path = "Z:\\Users\\d582428\\Downloads\\Tensile2d"


hf_dataset = load_from_disk(path)
ds, pbd =  huggingface_dataset_to_plaid(hf_dataset)
for sample in ds:
    print(sample)