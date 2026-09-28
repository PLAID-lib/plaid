"""Print per-feature minimum and maximum values for each Tensile2d split."""

from pathlib import Path

import numpy as np
from datasets import load_from_disk


DATASET_DIR = Path("/home/fbw/repos/Safran/maestro/data/Tensile2d/data")


def get_min_max(
    values: list[object],
) -> tuple[float | int, float | int] | None:
    """Return minimum and maximum finite values from nested feature data.

    Args:
        Values for one dataset feature across a split. Individual samples may
        have different array shapes.

    Returns:
        The global minimum and maximum values, or ``None`` when every sample
        has a missing or non-finite value for the feature.
    """
    minimum: float | int | None = None
    maximum: float | int | None = None

    for value in values:
        if value is None:
            continue

        array = np.asarray(value).reshape(-1)

        if np.issubdtype(array.dtype, np.floating):
            array = array[np.isfinite(array)]

        if array.size == 0:
            continue

        sample_minimum = array.min().item()
        sample_maximum = array.max().item()
        minimum = sample_minimum if minimum is None else min(minimum, sample_minimum)
        maximum = sample_maximum if maximum is None else max(maximum, sample_maximum)

    if minimum is None or maximum is None:
        return None

    return minimum, maximum


dataset_dict = load_from_disk(DATASET_DIR)

for split_name, dataset in dataset_dict.items():
    print(f"\n{'=' * 80}")
    print(f"Split: {split_name} ({len(dataset)} samples)")
    print(f"{'=' * 80}")

    for feature_name in dataset.column_names:
        if not feature_name.startswith("Global"):
            continue

        min_max = get_min_max(dataset[feature_name])

        if min_max is None:
            print(f"{feature_name:<65} min=n/a  max=n/a (missing)")
            continue

        minimum, maximum = min_max
        print(f"{feature_name:<65} min={minimum: .8e}  max={maximum: .8e}")
