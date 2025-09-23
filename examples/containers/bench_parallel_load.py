import argparse
import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rich
from tqdm import tqdm

from plaid import Dataset
from plaid import Sample

if __name__ == "__main__":
    # ---# Input parameters
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--number_of_samples",
        # dest='number_of_samples',
        type=int,
        default=100,
        help="number of samples to test parallel load",
    )
    parser.add_argument(
        "-t",
        "--number_of_tests",
        # dest='number_of_tests',
        type=int,
        default=5,
        help="number of tests to perform to compute statistics",
    )
    parser.add_argument(
        "-c",
        "--number_of_cores_max",
        # dest='number_of_cores_max',
        type=int,
        default=None,
        help="max number of cores to test on",
    )
    args = parser.parse_args()
    print()
    rich.print(args)
    print()

    NB_CORES = (
        2
        if args.number_of_cores_max is None
        else (
            os.cpu_count()
            if args.number_of_cores_max < 0
            else min(os.cpu_count(), args.number_of_cores_max)
        )
    )

    rich.print(f"Will run with max {NB_CORES=}")

    with tempfile.TemporaryDirectory() as out_dir:
        rich.print(f"{out_dir=}")
        print()

        t0 = time.perf_counter()
        # ---# Build a fake dataset
        dset = Dataset()
        for i in tqdm(range(args.number_of_samples), desc="Generate samples"):
            # ---# Read a sample provided in tests
            sample_path = Path("../tests/containers/dataset/samples/sample_000000000")
            if not (sample_path.is_dir()):
                sample_path = Path(
                    "../../tests/containers/dataset/samples/sample_000000000"
                )
            tmpsmp = Sample(path=sample_path)
            smp = tmpsmp

            # ---# Add some random data
            smp.add_scalar("id", i)
            smp.add_scalar("s0", np.random.randn())
            smp.add_field("f0", np.random.randn(100))

            dset.add_sample(smp)

        dset._save_to_dir_(out_dir, verbose=True)
        rich.print(f"create and save dataset took: {time.perf_counter() - t0:.3f} s")
        print()

        # ---# Measure loading durations depending on number of cores
        all_durations = []
        for nb_cores in tqdm(range(NB_CORES + 1), desc="Loop on nb_cores"):
            durations = []
            for _ in tqdm(range(args.number_of_tests), desc="   Loop on nb_tests"):
                new_dset = Dataset()
                t0 = time.perf_counter()
                new_dset._load_from_dir_(out_dir, processes_number=nb_cores)
                t1 = time.perf_counter()
                durations.append(t1 - t0)

            all_durations.append(durations)
            # rich.print(f'<{nb_cores=:2d}> load took: {np.min(durations):.3f} s | {np.mean(durations):.3f} s | {np.max(durations):.3f} s | {np.std(durations):.3f} s')

        all_min = [np.min(d) for d in all_durations]
        all_mean = [np.mean(d) for d in all_durations]
        all_max = [np.max(d) for d in all_durations]
        all_std = [np.std(d) for d in all_durations]

        plt.figure()
        plt.plot([0, NB_CORES], [0, 0], "k--")
        plt.plot(all_min, label="min")
        plt.plot(all_mean, label="mean")
        plt.plot(all_max, label="max")
        plt.plot(all_std, label="std")
        plt.legend()
        plt.xlabel("nb cores")
        plt.ylabel("duration")
        plt.tight_layout()
        plt.savefig(
            f"bench_{args.number_of_samples}_{args.number_of_tests}_{NB_CORES}.png"
        )

    os.remove("bench_100_5_2.png")