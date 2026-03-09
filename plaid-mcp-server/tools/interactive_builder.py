"""
Interactive Builder Tools for PLAID MCP Server

Provides tools for building conversion scripts interactively step-by-step.
"""

from typing import Any, Dict


class InteractiveBuilder:
    """Tools for interactive conversion script building."""
    
    def __init__(self, session_manager):
        self.session = session_manager
        
    async def start_conversion_session(self, dataset_name: str) -> Dict[str, Any]:
        """Start an interactive conversion session."""
        session_id = self.session.create_conversion_session(dataset_name)
        
        return {
            "success": True,
            "session_id": session_id,
            "dataset_name": dataset_name,
            "message": f"Conversion session started with ID: {session_id}",
            "next_step": "Add components using add_script_component tool",
            "available_components": ["generator", "problem_definition", "metadata", "infos"]
        }
    
    async def add_script_component(
        self,
        session_id: str,
        component_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add a component to the conversion session."""
        session = self.session.get_conversion_session(session_id)
        if not session:
            return {"error": f"Session not found: {session_id}"}
        
        # Validate component type
        valid_types = ["generator", "problem_definition", "metadata", "infos"]
        if component_type not in valid_types:
            return {
                "error": f"Invalid component_type: {component_type}",
                "valid_types": valid_types
            }
        
        # Add component
        success = self.session.add_session_component(session_id, component_type, config)
        
        if success:
            return {
                "success": True,
                "session_id": session_id,
                "component_type": component_type,
                "message": f"Added {component_type} component to session",
                "current_components": list(session.components.keys())
            }
        
        return {"error": "Failed to add component"}
    
    async def preview_conversion_script(self, session_id: str) -> Dict[str, Any]:
        """Preview the current state of the conversion script."""
        session = self.session.get_conversion_session(session_id)
        if not session:
            return {"error": f"Session not found: {session_id}"}
        
        # Generate preview script
        script_preview = self._generate_script_from_session(session)
        
        return {
            "success": True,
            "session_id": session_id,
            "dataset_name": session.dataset_name,
            "components": list(session.components.keys()),
            "script_preview": script_preview,
            "is_complete": self._is_session_complete(session),
            "missing_components": self._get_missing_components(session)
        }
    
    async def finalize_conversion_script(
        self,
        session_id: str,
        output_path: str
    ) -> Dict[str, Any]:
        """Finalize and save the conversion script."""
        session = self.session.get_conversion_session(session_id)
        if not session:
            return {"error": f"Session not found: {session_id}"}
        
        if not self._is_session_complete(session):
            return {
                "error": "Session incomplete",
                "missing_components": self._get_missing_components(session)
            }
        
        try:
            from pathlib import Path
            
            # Generate final script
            final_script = self._generate_script_from_session(session)
            
            # Save to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(final_script)
            
            # Clean up session
            self.session.remove_conversion_session(session_id)
            
            return {
                "success": True,
                "session_id": session_id,
                "output_path": str(output_file),
                "dataset_name": session.dataset_name,
                "message": f"Conversion script saved to {output_path}",
                "session_closed": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_script_from_session(self, session) -> str:
        """Generate script content from session components."""
        components = session.components
        
        script = f'''"""
PLAID Conversion Script: {session.dataset_name}
Generated interactively via PLAID MCP Server
Created: {session.created_at}
"""

from pathlib import Path
import numpy as np
from plaid import Sample, ProblemDefinition
from plaid.storage import save_to_disk, push_to_hub

'''
        
        # Add infos
        if "infos" in components:
            infos_config = components["infos"]
            script += f'''
# Dataset Information
infos = {infos_config}

'''
        
        # Add problem definition
        if "problem_definition" in components:
            pb_config = components["problem_definition"]
            script += f'''
# Problem Definition
pb_def = ProblemDefinition()
pb_def.add_in_features_identifiers({pb_config.get("input_features", [])})
pb_def.add_out_features_identifiers({pb_config.get("output_features", [])})
pb_def.add_constant_features_identifiers({pb_config.get("constant_features", [])})
pb_def.set_task("{pb_config.get("task", "regression")}")
pb_def.set_name("{pb_config.get("name", "default")}")

'''
        
        # Add generator
        if "generator" in components:
            gen_config = components["generator"]
            script += f'''
# Generator Function
def generator_fn():
    """Generate PLAID samples from raw data."""
    # TODO: Implement data loading logic
    for i in range({gen_config.get("num_samples", 10)}):
        sample = Sample()
        # TODO: Add tree, fields, scalars to sample
        yield sample

generators = {{"train": generator_fn}}

'''
        
        # Add execution
        script += '''
# Execution
if __name__ == "__main__":
    save_to_disk(
        output_folder="output_dataset",
        generators=generators,
        backend="hf_datasets",
        infos=infos,
        pb_defs=pb_def,
        overwrite=True,
        verbose=True
    )
'''
        
        return script
    
    def _is_session_complete(self, session) -> bool:
        """Check if session has all required components."""
        required = ["generator", "problem_definition", "infos"]
        return all(comp in session.components for comp in required)
    
    def _get_missing_components(self, session) -> list:
        """Get list of missing required components."""
        required = ["generator", "problem_definition", "infos"]
        return [comp for comp in required if comp not in session.components]
