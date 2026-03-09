#!/usr/bin/env python3
"""
PLAID MCP Server

This MCP server provides tools for working with PLAID datasets, including:
- Storage operations (load/save datasets)
- Sample operations (access and manipulate samples)
- Problem definition management
- Conversion script generation (main feature)
- Interactive script building

The server leverages plaid-skills examples to generate customized conversion scripts.
"""

import asyncio
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from session_manager import SessionManager
from tools.storage_tools import StorageTools
from tools.sample_tools import SampleTools
from tools.problem_tools import ProblemTools
from tools.conversion_tools import ConversionTools
from tools.interactive_builder import InteractiveBuilder
from tools.utility_tools import UtilityTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plaid-mcp-server")

# Initialize server
app = Server("plaid-mcp-server")

# Initialize session manager and tool modules
session_manager = SessionManager()
storage_tools = StorageTools(session_manager)
sample_tools = SampleTools(session_manager)
problem_tools = ProblemTools(session_manager)
conversion_tools = ConversionTools(session_manager)
interactive_builder = InteractiveBuilder(session_manager)
utility_tools = UtilityTools(session_manager)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    tools = []
    
    # Storage Operations
    tools.extend([
        Tool(
            name="init_from_disk",
            description="Load a PLAID dataset from local disk with converters for accessing samples",
            inputSchema={
                "type": "object",
                "properties": {
                    "local_dir": {
                        "type": "string",
                        "description": "Path to the local directory containing the saved dataset"
                    },
                    "splits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of split names to load. If not provided, all splits are loaded"
                    }
                },
                "required": ["local_dir"]
            }
        ),
        Tool(
            name="download_from_hub",
            description="Download a PLAID dataset from HuggingFace Hub to local disk",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": "HuggingFace repository ID (e.g., 'username/dataset-name')"
                    },
                    "local_dir": {
                        "type": "string",
                        "description": "Local directory where the dataset will be downloaded"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite existing directory",
                        "default": False
                    }
                },
                "required": ["repo_id", "local_dir"]
            }
        ),
        Tool(
            name="get_dataset_info",
            description="Get information about a loaded dataset including splits, features, and metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the loaded dataset"
                    }
                },
                "required": ["dataset_id"]
            }
        ),
        Tool(
            name="list_loaded_datasets",
            description="List all currently loaded datasets in the session",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
    ])
    
    # Sample Operations
    tools.extend([
        Tool(
            name="get_sample",
            description="Retrieve a sample from a loaded dataset and convert it to PLAID format",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the loaded dataset"
                    },
                    "split_name": {
                        "type": "string",
                        "description": "Name of the split (e.g., 'train', 'test')"
                    },
                    "sample_idx": {
                        "type": "integer",
                        "description": "Index of the sample to retrieve"
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of feature names to include"
                    }
                },
                "required": ["dataset_id", "split_name", "sample_idx"]
            }
        ),
        Tool(
            name="get_sample_info",
            description="Get metadata and structure information about a sample without loading all data",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "ID of the loaded dataset"
                    },
                    "split_name": {
                        "type": "string",
                        "description": "Name of the split"
                    },
                    "sample_idx": {
                        "type": "integer",
                        "description": "Index of the sample"
                    }
                },
                "required": ["dataset_id", "split_name", "sample_idx"]
            }
        ),
    ])
    
    # Problem Definition Tools
    tools.extend([
        Tool(
            name="load_problem_definitions",
            description="Load problem definitions from a local directory or HuggingFace Hub",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to local directory or HuggingFace repo ID"
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["disk", "hub"],
                        "description": "Type of source: 'disk' for local or 'hub' for HuggingFace"
                    }
                },
                "required": ["source", "source_type"]
            }
        ),
        Tool(
            name="get_problem_definition_info",
            description="Get information about a loaded problem definition",
            inputSchema={
                "type": "object",
                "properties": {
                    "problem_def_id": {
                        "type": "string",
                        "description": "ID of the problem definition"
                    }
                },
                "required": ["problem_def_id"]
            }
        ),
    ])
    
    # Conversion Script Generation Tools
    tools.extend([
        Tool(
            name="generate_conversion_script",
            description="Generate a complete PLAID conversion script customized for your dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_description": {
                        "type": "string",
                        "description": "Description of your dataset (e.g., 'Time-series CFD data with unstructured meshes')"
                    },
                    "data_characteristics": {
                        "type": "object",
                        "properties": {
                            "temporal": {"type": "boolean"},
                            "structured": {"type": "boolean"},
                            "field_location": {
                                "type": "string",
                                "enum": ["nodal", "cell_centered", "mixed"]
                            },
                            "file_format": {"type": "string"},
                            "num_samples": {"type": "integer"}
                        },
                        "description": "Characteristics of your dataset"
                    },
                    "raw_data_path": {
                        "type": "string",
                        "description": "Path to sample raw data (for analysis)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save the generated script"
                    }
                },
                "required": ["dataset_description", "data_characteristics", "output_path"]
            }
        ),
        Tool(
            name="analyze_raw_data_structure",
            description="Analyze the structure of your raw data to help inform script generation",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to raw data directory"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File pattern to match (e.g., '*.h5', '*.vtk')"
                    }
                },
                "required": ["data_path"]
            }
        ),
        Tool(
            name="list_conversion_examples",
            description="List available conversion examples from plaid-skills repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_by": {
                        "type": "object",
                        "description": "Optional filters (temporal, structured, etc.)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_conversion_example",
            description="Get the full content and explanation of a specific conversion example",
            inputSchema={
                "type": "object",
                "properties": {
                    "example_name": {
                        "type": "string",
                        "description": "Name of the example (e.g., 'shapenetcar', 'pdebench_2d_darcy_flow')"
                    }
                },
                "required": ["example_name"]
            }
        ),
        Tool(
            name="find_similar_conversion",
            description="Find the conversion example most similar to your dataset characteristics",
            inputSchema={
                "type": "object",
                "properties": {
                    "characteristics": {
                        "type": "object",
                        "properties": {
                            "temporal": {"type": "boolean"},
                            "structured": {"type": "boolean"},
                            "field_location": {"type": "string"}
                        },
                        "description": "Your dataset characteristics"
                    }
                },
                "required": ["characteristics"]
            }
        ),
        Tool(
            name="validate_conversion_script",
            description="Validate a conversion script by testing it on sample data",
            inputSchema={
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": "Path to the conversion script"
                    },
                    "test_data_path": {
                        "type": "string",
                        "description": "Path to test data"
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of samples to test",
                        "default": 3
                    }
                },
                "required": ["script_path"]
            }
        ),
    ])
    
    # Interactive Building Tools
    tools.extend([
        Tool(
            name="start_conversion_session",
            description="Start an interactive session for building a conversion script step-by-step",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name for your dataset"
                    }
                },
                "required": ["dataset_name"]
            }
        ),
        Tool(
            name="add_script_component",
            description="Add a component to an active conversion session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "ID of the conversion session"
                    },
                    "component_type": {
                        "type": "string",
                        "enum": ["generator", "problem_definition", "metadata", "infos"],
                        "description": "Type of component to add"
                    },
                    "config": {
                        "type": "object",
                        "description": "Configuration for the component"
                    }
                },
                "required": ["session_id", "component_type", "config"]
            }
        ),
        Tool(
            name="preview_conversion_script",
            description="Preview the current state of a conversion script being built",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "ID of the conversion session"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="finalize_conversion_script",
            description="Finalize and save a conversion script from an interactive session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "ID of the conversion session"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path where to save the final script"
                    }
                },
                "required": ["session_id", "output_path"]
            }
        ),
    ])
    
    # Utility Tools
    tools.extend([
        Tool(
            name="list_available_backends",
            description="List available storage backends for PLAID datasets",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_conversion_pattern",
            description="Get documentation about a specific conversion pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_name": {
                        "type": "string",
                        "description": "Name of the pattern (e.g., 'static_vs_temporal_samples')"
                    }
                },
                "required": ["pattern_name"]
            }
        ),
    ])
    
    return tools


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls by routing to appropriate tool module."""
    try:
        logger.info(f"Tool called: {name} with arguments: {arguments}")
        
        # Storage Tools
        if name == "init_from_disk":
            result = await storage_tools.init_from_disk(**arguments)
        elif name == "download_from_hub":
            result = await storage_tools.download_from_hub(**arguments)
        elif name == "get_dataset_info":
            result = await storage_tools.get_dataset_info(**arguments)
        elif name == "list_loaded_datasets":
            result = await storage_tools.list_loaded_datasets()
            
        # Sample Tools
        elif name == "get_sample":
            result = await sample_tools.get_sample(**arguments)
        elif name == "get_sample_info":
            result = await sample_tools.get_sample_info(**arguments)
            
        # Problem Tools
        elif name == "load_problem_definitions":
            result = await problem_tools.load_problem_definitions(**arguments)
        elif name == "get_problem_definition_info":
            result = await problem_tools.get_problem_definition_info(**arguments)
            
        # Conversion Tools
        elif name == "generate_conversion_script":
            result = await conversion_tools.generate_conversion_script(**arguments)
        elif name == "analyze_raw_data_structure":
            result = await conversion_tools.analyze_raw_data_structure(**arguments)
        elif name == "list_conversion_examples":
            result = await conversion_tools.list_conversion_examples(**arguments)
        elif name == "get_conversion_example":
            result = await conversion_tools.get_conversion_example(**arguments)
        elif name == "find_similar_conversion":
            result = await conversion_tools.find_similar_conversion(**arguments)
        elif name == "validate_conversion_script":
            result = await conversion_tools.validate_conversion_script(**arguments)
            
        # Interactive Builder Tools
        elif name == "start_conversion_session":
            result = await interactive_builder.start_conversion_session(**arguments)
        elif name == "add_script_component":
            result = await interactive_builder.add_script_component(**arguments)
        elif name == "preview_conversion_script":
            result = await interactive_builder.preview_conversion_script(**arguments)
        elif name == "finalize_conversion_script":
            result = await interactive_builder.finalize_conversion_script(**arguments)
            
        # Utility Tools
        elif name == "list_available_backends":
            result = await utility_tools.list_available_backends()
        elif name == "get_conversion_pattern":
            result = await utility_tools.get_conversion_pattern(**arguments)
        else:
            result = f"Unknown tool: {name}"
            
        return [TextContent(type="text", text=str(result))]
        
    except Exception as e:
        logger.error(f"Error executing tool {name}: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
