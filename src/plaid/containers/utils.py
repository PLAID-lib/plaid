"""Utility functions for PLAID containers."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from pathlib import Path
from typing import Union

# %% Functions


def get_sample_ids(savedir: Union[str, Path]) -> list[int]:
    """Return list of sample ids in a dataset on disk.

    Args:
        savedir (Union[str,Path]): The path to the directory where sample files are stored.

    Returns:
        list[int]: List of sample ids.
    """
    savedir = Path(savedir)
    return sorted(
        [
            int(d.stem.split("_")[-1])
            for d in (savedir / "samples").glob("sample_*")
            if d.is_dir()
        ]
    )


def get_number_of_samples(savedir: Union[str, Path]) -> int:
    """Return number of samples in a dataset on disk.

    Args:
        savedir (Union[str,Path]): The path to the directory where sample files are stored.

    Returns:
        int: number of samples.
    """
    return len(get_sample_ids(savedir))
