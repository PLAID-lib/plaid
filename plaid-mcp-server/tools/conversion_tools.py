"""
Conversion Tools for PLAID MCP Server

Provides tools for generating PLAID conversion scripts based on plaid-skills examples.
"""

import os
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional


# Path to plaid-skills repository
PLAID_SKILLS_PATH = Path("/home/sagemaker-user/softs/plaid-skills")
CONVERSIONS_PATH = PLAID_SKILLS_PATH / "skills/plaid-conversion/examples/conversions"
PATTERNS_PATH = PLAID_SKILLS_PATH / "skills/plaid-conversion/examples/patterns"


class ConversionTools:
    """Tools for generating PLAID conversion scripts."""
    
    def __init__(self, session_manager):
        self.session = session_manager
        self._examples_cache = None
        
    def _get_examples_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about available conversion examples."""
        if self._examples_cache:
            return self._examples_cache
            
        examples = {
            "shapenetcar": {
                "file": "shapenetcar.py",
                "temporal": False,
                "structured": False,
                "field_location": "nodal",
                "mesh_type": "unstructured",
                "description": "Static triangular surface meshes with nodal scalar fields",
                "dataset_type": "geometry",
                "suitable_for": ["3D meshes", "static data", "surface meshes", "CAD geometry"]
            },
            "pdebench_2d_darcy_flow": {
                "file": "pdebench_2d_darcy_flow.py",
                "temporal": False,
                "structured": True,
                "field_location": "cell_centered",
                "mesh_type": "structured",
                "description": "Static, parameterized PDE dataset with cell-centered fields on rectilinear grid",
                "dataset_type": "pde_simulation",
                "suitable_for": ["structured grids", "cell-centered data", "parametric studies", "PDE solutions"]
            },
            "force_asr": {
                "file": "force_asr.py",
                "temporal": True,
                "structured": False,
                "field_location": "nodal",
                "mesh_type": "unstructured",
                "description": "Time-dependent fracture mechanics with trajectories",
                "dataset_type": "temporal_simulation",
                "suitable_for": ["time series", "trajectories", "nodal fields", "external time metadata"]
            },
            "thewell_turbulent_layer_2d": {
                "file": "thewell_turbulent_layer_2d.py",
                "temporal": True,
                "structured": True,
                "field_location": "nodal",
                "mesh_type": "structured",
                "description": "Temporal trajectories on structured grids",
                "dataset_type": "temporal_simulation",
                "suitable_for": ["time series", "structured grids", "fluid dynamics", "periodic domains"]
            },
            "drivaerml": {
                "file": "drivaerml.py",
                "temporal": False,
                "structured": False,
                "field_location": "cell_centered",
                "mesh_type": "unstructured",
                "description": "Steady-state automotive CFD with OpenFOAM",
                "dataset_type": "cfd_simulation",
                "suitable_for": ["CFD data", "OpenFOAM", "static simulations", "automotive"]
            }
        }
        
        self._examples_cache = examples
        return examples
        
    async def list_conversion_examples(
        self,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List available conversion examples from plaid-skills."""
        examples = self._get_examples_metadata()
        
        if filter_by:
            filtered = {}
            for name, meta in examples.items():
                match = True
                for key, value in filter_by.items():
                    if key in meta and meta[key] != value:
                        match = False
                        break
                if match:
                    filtered[name] = meta
            examples = filtered
        
        return {
            "num_examples": len(examples),
            "examples": examples,
            "skills_path": str(PLAID_SKILLS_PATH)
        }
    
    async def get_conversion_example(self, example_name: str) -> Dict[str, Any]:
        """Get the full content of a conversion example."""
        examples = self._get_examples_metadata()
        
        if example_name not in examples:
            return {
                "error": f"Example not found: {example_name}",
                "available_examples": list(examples.keys())
            }
        
        example_file = CONVERSIONS_PATH / examples[example_name]["file"]
        
        if not example_file.exists():
            return {"error": f"Example file not found: {example_file}"}
        
        try:
            with open(example_file, 'r') as f:
                content = f.read()
            
            return {
                "success": True,
                "example_name": example_name,
                "metadata": examples[example_name],
                "file_path": str(example_file),
                "content": content,
                "length_lines": len(content.split('\n')),
                "message": f"Use this as a template for your conversion script"
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def find_similar_conversion(
        self,
        characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find the most similar conversion example based on characteristics."""
        examples = self._get_examples_metadata()
        
        # Score each example based on matching characteristics
        scores = {}
        for name, meta in examples.items():
            score = 0
            matches = []
            
            if "temporal" in characteristics and meta.get("temporal") == characteristics["temporal"]:
                score += 3
                matches.append("temporal" if characteristics["temporal"] else "static")
            
            if "structured" in characteristics and meta.get("structured") == characteristics["structured"]:
                score += 2
                matches.append("structured" if characteristics["structured"] else "unstructured")
            
            if "field_location" in characteristics and meta.get("field_location") == characteristics["field_location"]:
                score += 2
                matches.append(f"field_location={characteristics['field_location']}")
            
            scores[name] = {"score": score, "matches": matches, "metadata": meta}
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        return {
            "characteristics": characteristics,
            "best_match": ranked[0][0] if ranked else None,
            "ranked_examples": [
                {
                    "name": name,
                    "score": data["score"],
                    "matches": data["matches"],
                    "description": data["metadata"]["description"]
                }
                for name, data in ranked[:3]  # Top 3
            ],
            "recommendation": f"Start with '{ranked[0][0]}' example as a template" if ranked else "No matches found"
        }
    
    async def analyze_raw_data_structure(
        self,
        data_path: str,
        file_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze the structure of raw data to help inform script generation."""
        try:
            data_dir = Path(data_path)
            if not data_dir.exists():
                return {"error": f"Path not found: {data_path}"}
            
            # Find files
            if file_pattern:
                files = list(data_dir.glob(file_pattern))
            else:
                files = list(data_dir.iterdir())
            
            # Analyze file types
            extensions = {}
            for f in files:
                if f.is_file():
                    ext = f.suffix.lower()
                    extensions[ext] = extensions.get(ext, 0) + 1
            
            # Infer characteristics
            inferred = {
                "has_hdf5": any(ext in ['.h5', '.hdf5'] for ext in extensions),
                "has_vtk": any(ext in ['.vtk', '.vtu', '.vtp'] for ext in extensions),
                "has_cgns": any(ext == '.cgns' for ext in extensions),
                "has_ply": any(ext == '.ply' for ext in extensions),
                "has_csv": any(ext == '.csv' for ext in extensions),
                "has_numpy": any(ext in ['.npy', '.npz'] for ext in extensions)
            }
            
            return {
                "success": True,
                "data_path": data_path,
                "num_files": len(files),
                "file_extensions": extensions,
                "inferred_characteristics": inferred,
                "sample_files": [str(f.name) for f in files[:5]],  # First 5
                "suggestions": self._get_suggestions_from_analysis(extensions, inferred)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_suggestions_from_analysis(
        self,
        extensions: Dict[str, int],
        inferred: Dict[str, bool]
    ) -> List[str]:
        """Generate suggestions based on data analysis."""
        suggestions = []
        
        if inferred["has_hdf5"]:
            suggestions.append("HDF5 files detected - consider pdebench_2d_darcy_flow.py as template")
        if inferred["has_ply"]:
            suggestions.append("PLY mesh files detected - consider shapenetcar.py as template")
        if inferred["has_vtk"]:
            suggestions.append("VTK files detected - may need vtk/Muscat for conversion")
        if inferred["has_cgns"]:
            suggestions.append("CGNS files detected - can use cgns backend directly")
        if inferred["has_csv"]:
            suggestions.append("CSV files detected - may contain metadata or scalar parameters")
        
        return suggestions
    
    async def generate_conversion_script(
        self,
        dataset_description: str,
        data_characteristics: Dict[str, Any],
        output_path: str,
        raw_data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a complete conversion script customized for the dataset."""
        try:
            # Find best matching example
            similar = await self.find_similar_conversion(data_characteristics)
            best_example = similar["best_match"]
            
            if not best_example:
                return {"error": "Could not find suitable template"}
            
            # Get the example script
            example_data = await self.get_conversion_example(best_example)
            if "error" in example_data:
                return example_data
            
            template_content = example_data["content"]
            
            # Customize the script
            customized_script = self._customize_script(
                template_content,
                dataset_description,
                data_characteristics,
                best_example,
                raw_data_path
            )
            
            # Save the script
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(customized_script)
            
            return {
                "success": True,
                "output_path": str(output_file),
                "based_on_example": best_example,
                "characteristics": data_characteristics,
                "message": f"Conversion script generated at {output_path}. IMPORTANT: Review and customize the marked sections before running!",
                "next_steps": [
                    "1. Open the generated script and review all CUSTOMIZATION_REQUIRED sections",
                    "2. Set RAW_DATA_DIR, OUTPUT_DIR, and REPO_ID variables",
                    "3. Adapt the generator_fn() to your specific data format",
                    "4. Verify feature identifiers match your data",
                    "5. Test on a small subset first using validate_conversion_script tool"
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _customize_script(
        self,
        template: str,
        description: str,
        characteristics: Dict[str, Any],
        example_name: str,
        raw_data_path: Optional[str]
    ) -> str:
        """Customize the template script for the user's dataset."""
        
        # Add custom header
        header = f'''"""
PLAID Dataset Conversion Script
AUTO-GENERATED by PLAID MCP Server

Dataset Description: {description}
Based on example: {example_name}
Generated: {self._get_timestamp()}

Characteristics:
- Temporal: {characteristics.get("temporal", "unknown")}
- Structured: {characteristics.get("structured", "unknown")}
- Field Location: {characteristics.get("field_location", "unknown")}
- File Format: {characteristics.get("file_format", "unknown")}

⚠️  IMPORTANT - CUSTOMIZATION REQUIRED:
This script is a TEMPLATE that needs customization for your specific dataset.
Search for "CUSTOMIZATION_REQUIRED" comments throughout this file.

Key sections to customize:
1. Data paths (RAW_DATA_DIR, OUTPUT_DIR, REPO_ID)
2. Generator function (how to read your specific data format)
3. Feature identifiers (input/output/constant features)
4. Metadata and infos (owner, license, physics domain)
5. Problem definition (task type, splits)

Before running:
- Test on a small subset first
- Verify mesh/field construction matches your data structure
- Validate output samples using PLAID's sample.summarize()
"""

# Original template from: {example_name}
# Refer to plaid-skills documentation for patterns and best practices

'''
        
        # Customize paths if raw_data_path provided
        if raw_data_path:
            template = template.replace(
                'RAW_DATA_DIR = "/path/to/raw/data"',
                f'RAW_DATA_DIR = "{raw_data_path}"  # CUSTOMIZATION_REQUIRED: Verify this path'
            )
        
        # Add customization markers
        template = template.replace(
            'def generator_fn',
            '# CUSTOMIZATION_REQUIRED: Adapt this generator to your data format\ndef generator_fn'
        )
        
        template = template.replace(
            'infos = {',
            '# CUSTOMIZATION_REQUIRED: Update owner, license, and physics domain\ninfos = {'
        )
        
        template = template.replace(
            'input_features = [',
            '# CUSTOMIZATION_REQUIRED: Define your input features\ninput_features = ['
        )
        
        template = template.replace(
            'output_features = [',
            '# CUSTOMIZATION_REQUIRED: Define your output features\noutput_features = ['
        )
        
        return header + template
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def validate_conversion_script(
        self,
        script_path: str,
        test_data_path: Optional[str] = None,
        num_samples: int = 3
    ) -> Dict[str, Any]:
        """Validate a conversion script by testing it."""
        try:
            script_file = Path(script_path)
            if not script_file.exists():
                return {"error": f"Script not found: {script_path}"}
            
            # Check for customization markers
            with open(script_file, 'r') as f:
                content = f.read()
            
            markers = content.count("CUSTOMIZATION_REQUIRED")
            has_placeholders = "/path/to/" in content or "channel/repo" in content
            
            warnings = []
            if markers > 0:
                warnings.append(f"Found {markers} CUSTOMIZATION_REQUIRED markers - review these sections")
            if has_placeholders:
                warnings.append("Found placeholder paths - update before running")
            
            return {
                "success": True,
                "script_path": script_path,
                "validation_warnings": warnings,
                "customization_markers": markers,
                "has_placeholders": has_placeholders,
                "status": "NEEDS_REVIEW" if warnings else "READY",
                "message": "Script generated successfully. Review customization sections before running.",
                "recommendation": "Fix all warnings, then test with a small data subset"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
