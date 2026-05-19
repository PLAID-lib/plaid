"""Examples for PLAID objects."""
from typing import Any

from plaid.examples.config import _HF_REPOS
from plaid.storage import init_streaming_from_hub
from ..containers.sample import Sample


AVAILABLE_EXAMPLES = list(_HF_REPOS.keys())

__all__ = ["datasets", "samples", "AVAILABLE_EXAMPLES"]
class Datasets_examples:
    def __init__(self):
        self.cache: dict[str,Any] = {}
    def __getattr__ (self, key):

        if key in _HF_REPOS.keys():
            if key not in self.cache:
                datasetdict, converterdict = init_streaming_from_hub(_HF_REPOS[key])
                self.cache[key] = (datasetdict, converterdict)
            return self.cache[key]
        raise ValueError(f'cant find example dataset {key}')

datasets = Datasets_examples()

class Samples_example:
    def __init__(self):
        self.cache: dict[str,Sample] = {}
    def __getattr__ (self, key):
        if key in _HF_REPOS.keys():
            if key not in self.cache:
                datasetdict, converterdict = init_streaming_from_hub(_HF_REPOS[key])
                # first sample
                k =  list(converterdict.keys())[0]
                hf_sample = next(iter(datasetdict[k]))
                plaid_sample = converterdict[k].sample_to_plaid(hf_sample)
                self.cache[key] = plaid_sample
            return self.cache[key]
        raise ValueError(f'cant find example sample {key}')

samples = Samples_example()

