"""
Utility Tools for PLAID MCP Server

Provides utility functions and helper tools.
"""

from pathlib import Path
from typing import Any, Dict

from plaid.storage.registry import available_backends

PLAID_SKILLS_PATH = Path("/home/sagemaker-user/softs/plaid-skills")
PATTERNS_PATH = PLAID_SKILLS_PATH / "skills/plaid-conversion/examples/patterns"


class UtilityTools:
    """Utility tools for PLAID operations."""

    def __init__(self, session_manager):
        self.session = session_manager

    async def list_available_backends(self) -> Dict[str, Any]:
        """List available storage backends for PLAID datasets."""
        backends = available_backends()

        backend_info = {
            "cgns": {
                "name": "cgns",
                "description": "CGNS (CFD General Notation System) format - native PLAID storage",
                "suitable_for": "CFD and mesh-based simulations",
            },
            "hf_datasets": {
                "name": "hf_datasets",
                "description": "HuggingFace Datasets format - cloud-friendly, streaming capable",
                "suitable_for": "Sharing on HuggingFace Hub, large datasets, ML workflows",
            },
            "zarr": {
                "name": "zarr",
                "description": "Zarr format - chunked, compressed array storage",
                "suitable_for": "Large arrays, parallel I/O, cloud storage",
            },
        }

        return {
            "num_backends": len(backends),
            "backends": backends,
            "backend_details": {name: backend_info.get(name, {}) for name in backends},
            "default_recommended": "hf_datasets",
            "message": "Choose backend based on your use case and target platform",
        }

    async def get_conversion_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """Get documentation about a conversion pattern."""
        pattern_file = PATTERNS_PATH / f"{pattern_name}.md"

        if not pattern_file.exists():
            # List available patterns
            available = [f.stem for f in PATTERNS_PATH.glob("*.md")]
            return {
                "error": f"Pattern not found: {pattern_name}",
                "available_patterns": available,
            }

        try:
            with open(pattern_file, "r") as f:
                content = f.read()

            return {
                "success": True,
                "pattern_name": pattern_name,
                "file_path": str(pattern_file),
                "content": content,
                "message": "Pattern documentation retrieved successfully",
            }
        except Exception as e:
            return {"error": str(e)}
