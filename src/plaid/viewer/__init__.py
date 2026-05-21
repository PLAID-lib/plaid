"""Dataset viewer for PLAID.

This package hosts the raw PLAID dataset viewer: a FastAPI backend plus an
embedded trame/ParaView visualization server. PLAID owns the UI shell and
the page; PLAID owns data loading, sample interpretation, and CGNS export;
ParaView/trame owns the scientific visualization.
"""

from plaid.viewer.models import ParaviewArtifact, SampleRef

__all__ = ["ParaviewArtifact", "SampleRef"]
