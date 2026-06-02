"""Examples for PLAID objects."""

import multiprocessing as mp

from plaid.downloadable_examples.config import _HF_REPOS
from plaid.storage import init_streaming_from_hub

from ..containers.sample import Sample

AVAILABLE_EXAMPLES = list(_HF_REPOS.keys())

__all__ = ["samples", "AVAILABLE_EXAMPLES"]


def _load_first_sample(repo_id: str, queue: mp.Queue) -> None:
    """Load one sample in a worker to avoid HF streaming shutdown hangs."""
    datasetdict, converterdict = init_streaming_from_hub(repo_id)
    split = next(iter(converterdict))
    hf_sample = next(iter(datasetdict[split]))
    queue.put(converterdict[split].sample_to_plaid(hf_sample))


def _download_sample(repo_id: str) -> Sample:
    """Download the first example sample and return it as a PLAID sample."""
    queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_load_first_sample, args=(repo_id, queue))
    process.start()

    try:
        return queue.get(timeout=120)
    finally:
        process.terminate()
        process.join()


class Samples_example:
    def __init__(self):
        self.cache: dict[str, Sample] = {}

    def __getattr__(self, key):
        if key in _HF_REPOS.keys():
            if key not in self.cache:
                self.cache[key] = _download_sample(_HF_REPOS[key])
            return self.cache[key]
        raise ValueError(f"cant find example sample {key}")


samples = Samples_example()
