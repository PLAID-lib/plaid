import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from tqdm import tqdm

import zarr
from plaid import Sample
from plaid.storage.common import build_sample_dict, preprocess
from plaid.types import IndexType


def flatten_path(key: str) -> str:
    return key.replace("/", "__")


def plaid_generator_to_datasetdict(
    output_folder: str,
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    processes_number: int = 1,
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    # Preprocess to get feature keys
    flat_cst, key_mappings, var_features_types = preprocess(
        generators, gen_kwargs, processes_number, verbose
    )
    all_features_keys = list(var_features_types.keys())

    def worker_shard(
        split_root_path, gen_func, all_features_keys, shard, start_index, queue
    ):
        """Process a single shard and write samples to Zarr."""
        split_root = zarr.open_group(split_root_path, mode="a")
        sample_counter = start_index

        for sample in gen_func([shard]):  # client _generator([shard]) interface
            sample_dict, _, _ = build_sample_dict(sample)
            sample_data = {
                path: sample_dict.get(path, None) for path in all_features_keys
            }

            g = split_root.create_group(f"sample_{sample_counter:09d}")
            for key, value in sample_data.items():
                g.create_array(flatten_path(key), data=value)

            sample_counter += 1
            queue.put(1)  # notify tqdm

    def tqdm_updater(total, queue, desc="Processing"):
        """Tqdm process that listens to the queue to update progress."""
        with tqdm(total=total, desc=desc) as pbar:
            finished = 0
            while finished < total:
                finished += queue.get()
                pbar.update(1)

    for split_name, gen_func in generators.items():
        split_root_path = str(output_folder / split_name)
        split_root = zarr.open_group(split_root_path, mode="w")  # ensure group exists

        gen_kwargs_ = gen_kwargs or {sn: {} for sn in generators.keys()}
        shard_ids_list = gen_kwargs_.get(split_name, {}).get("shards_ids", [])

        total_samples = sum(len(shard) for shard in shard_ids_list)
        sample_counter = 0  # for sequential writes

        if processes_number > 1 and shard_ids_list:
            # Parallel execution
            queue = mp.Queue()
            tqdm_proc = mp.Process(
                target=tqdm_updater,
                args=(total_samples, queue, f"Writing {split_name} split"),
            )
            tqdm_proc.start()

            processes = []
            start_index = 0
            for shard in shard_ids_list:
                p = mp.Process(
                    target=worker_shard,
                    args=(
                        split_root_path,
                        gen_func,
                        all_features_keys,
                        shard,
                        start_index,
                        queue,
                    ),
                )
                p.start()
                processes.append(p)
                start_index += len(shard)

            for p in processes:
                p.join()

            tqdm_proc.join()

        else:
            # Sequential execution
            with tqdm(total=total_samples, desc=f"Writing {split_name} split") as pbar:
                for sample in gen_func():
                    sample_dict, _, _ = build_sample_dict(sample)
                    sample_data = {
                        path: sample_dict.get(path, None) for path in all_features_keys
                    }

                    g = split_root.create_group(f"sample_{sample_counter:09d}")
                    for key, value in sample_data.items():
                        g.create_array(flatten_path(key), data=value)

                    sample_counter += 1
                    pbar.update(1)

    return flat_cst, key_mappings
