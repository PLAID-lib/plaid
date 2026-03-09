"""
Problem Definition Tools for PLAID MCP Server

Provides tools for loading and managing PLAID problem definitions.
"""

from typing import Any, Dict

from plaid.storage import (
    load_problem_definitions_from_disk,
    load_problem_definitions_from_hub,
)


class ProblemTools:
    """Tools for PLAID problem definition operations."""

    def __init__(self, session_manager):
        self.session = session_manager

    async def load_problem_definitions(
        self, source: str, source_type: str
    ) -> Dict[str, Any]:
        """Load problem definitions from disk or hub."""
        try:
            if source_type == "disk":
                pb_defs = load_problem_definitions_from_disk(source)
            elif source_type == "hub":
                pb_defs = load_problem_definitions_from_hub(source)
            else:
                return {
                    "error": f"Invalid source_type: {source_type}. Must be 'disk' or 'hub'"
                }

            if pb_defs is None:
                return {
                    "success": False,
                    "message": "No problem definitions found at source",
                }

            # Store problem definitions
            pd_ids = {}
            if isinstance(pb_defs, dict):
                for name, pb_def in pb_defs.items():
                    pd_id = self.session.add_problem_definition(pb_def, name=name)
                    pd_ids[name] = pd_id
            else:
                pd_id = self.session.add_problem_definition(pb_defs)
                pd_ids["default"] = pd_id

            return {
                "success": True,
                "source": source,
                "source_type": source_type,
                "problem_definition_ids": pd_ids,
                "message": f"Loaded {len(pd_ids)} problem definition(s)",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to load problem definitions from {source}",
            }

    async def get_problem_definition_info(self, problem_def_id: str) -> Dict[str, Any]:
        """Get information about a loaded problem definition."""
        pb_def = self.session.get_problem_definition(problem_def_id)
        if not pb_def:
            return {"error": f"Problem definition not found: {problem_def_id}"}

        try:
            return {
                "success": True,
                "problem_def_id": problem_def_id,
                "name": pb_def.get_name(),
                "task": pb_def.get_task(),
                "score_function": pb_def.get_score_function(),
                "input_features": [
                    str(f) for f in pb_def.get_in_features_identifiers()
                ],
                "output_features": [
                    str(f) for f in pb_def.get_out_features_identifiers()
                ],
                "constant_features": pb_def.get_constant_features_identifiers(),
                "split_info": pb_def.get_split()
                if hasattr(pb_def, "get_split")
                else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
