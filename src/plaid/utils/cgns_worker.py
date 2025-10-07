"""Utility function to save a pickled CGNS tree in a subprocess."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import logging
import os
import pickle
import sys

import CGNS.MAP as CGM

logger = logging.getLogger(__name__)

if (
    __name__ == "__main__"
):  # pragma: no cover (run in subprocess, not picked up by coverage routine)
    tmpfile, outfname = sys.argv[1], sys.argv[2]
    with open(tmpfile, "rb") as f:
        tree = pickle.load(f)
    status = CGM.save(outfname, tree)
    logger.debug(f"save -> {status=}")
    os.remove(tmpfile)
