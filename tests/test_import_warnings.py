"""Importing plaid must not emit beartype deprecation warnings."""

import subprocess
import sys

# beartype classes cannot go through ``python -W`` (resolved before import), so filter in code.
_SCRIPT = """
import warnings
from beartype.roar import BeartypeDecorHintPepDeprecationWarning
warnings.simplefilter("error", BeartypeDecorHintPepDeprecationWarning)
import plaid
import plaid.utils.cgns_helper
import plaid.utils.cgns_vtk
import plaid.viewer.global_plots
import plaid.viewer.services.plaid_dataset_service
"""


def test_import_plaid_emits_no_beartype_deprecation_warning():
    """Every consumer sees import-time warnings, so turn them into errors."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
