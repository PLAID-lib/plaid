# PLAID MCP Server

An MCP (Model Context Protocol) server for PLAID (Physics Learning AI Datamodel) that provides tools for dataset management, sample operations, and **automated conversion script generation**.

## Overview

This MCP server helps users work with PLAID datasets and, most importantly, **generates customized conversion scripts** to transform raw scientific data into PLAID format. It leverages the [plaid-skills](https://github.com/PLAID-lib/plaid-skills) repository to provide real-world conversion templates.

## Key Features

### 🎯 Conversion Script Generation (Main Feature)
- **Automated script generation** based on your dataset characteristics
- Analyzes raw data structure and suggests best templates
- Customizes plaid-skills examples for your specific needs
- Adds helpful comments and customization markers
- Interactive and one-shot generation modes

### 📦 Dataset Management
- Load PLAID datasets from local disk or HuggingFace Hub
- Access samples with converters
- Manage multiple backends (CGNS, HF Datasets, Zarr)
- Session-based state management

### 🔬 Sample Operations
- Retrieve samples from loaded datasets
- Get sample metadata and structure
- Extract scalars, fields, and features

### 📝 Problem Definition Management
- Load and query problem definitions
- Access input/output feature specifications
- View task types and splits

## Installation

### Prerequisites

1. **PLAID library** (already installed in parent directory)
2. **plaid-skills repository** (cloned at `/home/sagemaker-user/softs/plaid-skills`)
3. **Python 3.11+**
4. **MCP SDK**

### Install Dependencies

```bash
cd /home/sagemaker-user/softs/plaid/plaid-mcp-server
pip install -e .
pip install mcp
```

## Configuration

### For Claude Desktop or other MCP clients

Add this to your MCP configuration file (e.g., `~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "plaid": {
      "command": "python",
      "args": [
        "/home/sagemaker-user/softs/plaid/plaid-mcp-server/server.py"
      ],
      "env": {
        "PYTHONPATH": "/home/sagemaker-user/softs/plaid:/home/sagemaker-user/softs/plaid-mcp-server"
      }
    }
  }
}
```

## Usage

### 1. Generate a Conversion Script (Main Use Case)

**One-Shot Generation:**

```python
# Tell the server about your dataset
result = generate_conversion_script(
    dataset_description="Time-series CFD simulations with unstructured meshes",
    data_characteristics={
        "temporal": true,
        "structured": false,
        "field_location": "nodal",
        "file_format": "hdf5",
        "num_samples": 100
    },
    raw_data_path="/path/to/my/data",
    output_path="my_conversion.py"
)

# Result includes:
# - Generated script path
# - Template used
# - Next steps for customization
```

The generated script will be a working template based on the most similar plaid-skills example, customized with:
- Your data path
- Appropriate comments
- CUSTOMIZATION_REQUIRED markers
- Instructions for adaptation

**Interactive Generation:**

```python
# Start a session
session = start_conversion_session(dataset_name="MyDataset")

# Add components step by step
add_script_component(
    session_id=session["session_id"],
    component_type="infos",
    config={
        "legal": {"owner": "Me", "license": "MIT"},
        "data_production": {"type": "simulation", "physics": "CFD"}
    }
)

add_script_component(
    session_id=session["session_id"],
    component_type="problem_definition",
    config={
        "input_features": ["velocity", "pressure"],
        "output_features": ["drag", "lift"],
        "task": "regression"
    }
)

# Preview and finalize
preview = preview_conversion_script(session_id=session["session_id"])
final = finalize_conversion_script(
    session_id=session["session_id"],
    output_path="my_conversion.py"
)
```

### 2. Find Similar Examples

```python
# Find which plaid-skills example matches your data
result = find_similar_conversion(
    characteristics={
        "temporal": false,
        "structured": true,
        "field_location": "cell_centered"
    }
)

# Returns ranked examples with explanations
# Best match: "pdebench_2d_darcy_flow"
```

### 3. Analyze Your Raw Data

```python
# Analyze data structure to inform conversion
analysis = analyze_raw_data_structure(
    data_path="/path/to/data",
    file_pattern="*.h5"
)

# Returns:
# - File types detected
# - Inferred characteristics
# - Template suggestions
```

### 4. Load and Query Datasets

```python
# Load a PLAID dataset
dataset = init_from_disk(local_dir="/path/to/dataset")

# Get dataset info
info = get_dataset_info(dataset_id=dataset["dataset_id"])

# Access a sample
sample = get_sample(
    dataset_id=dataset["dataset_id"],
    split_name="train",
    sample_idx=0
)
```

### 5. Work with Problem Definitions

```python
# Load problem definitions
pb_defs = load_problem_definitions(
    source="/path/to/dataset",
    source_type="disk"
)

# Query problem definition
info = get_problem_definition_info(
    problem_def_id="regression_1"
)
```

## Available Tools

### Conversion Script Generation (7 tools)
- `generate_conversion_script` - Generate complete conversion script
- `analyze_raw_data_structure` - Analyze raw data files
- `list_conversion_examples` - List plaid-skills examples
- `get_conversion_example` - Get full example content
- `find_similar_conversion` - Find matching template
- `validate_conversion_script` - Check generated script
- `get_conversion_pattern` - Get pattern documentation

### Storage Operations (4 tools)
- `init_from_disk` - Load dataset from disk
- `download_from_hub` - Download from HuggingFace Hub
- `get_dataset_info` - Get dataset information
- `list_loaded_datasets` - List active datasets

### Sample Operations (2 tools)
- `get_sample` - Retrieve sample with converter
- `get_sample_info` - Get sample metadata

### Problem Definitions (2 tools)
- `load_problem_definitions` - Load problem definitions
- `get_problem_definition_info` - Query problem definition

### Interactive Building (4 tools)
- `start_conversion_session` - Begin interactive session
- `add_script_component` - Add script components
- `preview_conversion_script` - Preview current state
- `finalize_conversion_script` - Save final script

### Utilities (2 tools)
- `list_available_backends` - Show storage backends
- `get_conversion_pattern` - Get pattern docs

**Total: 21 tools**

## Workflow Example

### Converting Your Dataset to PLAID

1. **Analyze your data:**
   ```python
   analysis = analyze_raw_data_structure(
       data_path="/my/data",
       file_pattern="*.h5"
   )
   # Suggestions: "HDF5 files detected - consider pdebench_2d_darcy_flow.py"
   ```

2. **Generate conversion script:**
   ```python
   script = generate_conversion_script(
       dataset_description="2D PDE simulations",
       data_characteristics={
           "temporal": false,
           "structured": true,
           "field_location": "cell_centered"
       },
       output_path="convert_my_data.py"
   )
   ```

3. **Review and customize:**
   - Open `convert_my_data.py`
   - Search for `CUSTOMIZATION_REQUIRED` comments
   - Update paths, generator logic, features

4. **Validate:**
   ```python
   validation = validate_conversion_script(
       script_path="convert_my_data.py"
   )
   # Check for warnings and placeholders
   ```

5. **Test on subset:**
   - Modify script to process only 3-5 samples
   - Run the script
   - Verify output using PLAID's `sample.summarize()`

6. **Full conversion:**
   - Update to full dataset
   - Run conversion
   - Upload to HuggingFace Hub (optional)

## Architecture

```
plaid-mcp-server/
├── server.py                    # Main MCP server
├── session_manager.py           # State management
├── tools/
│   ├── storage_tools.py         # Dataset loading/saving
│   ├── sample_tools.py          # Sample access
│   ├── problem_tools.py         # Problem definitions
│   ├── conversion_tools.py      # Script generation (KEY)
│   ├── interactive_builder.py   # Interactive mode
│   └── utility_tools.py         # Utilities
├── config.json                  # MCP configuration
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

## Design Philosophy

This server follows the [plaid-skills](https://github.com/PLAID-lib/plaid-skills) philosophy:

- ✅ **Guidance over automation** - Generate helpful templates, not black boxes
- ✅ **Explicit over implicit** - Make assumptions visible
- ✅ **Correctness over convenience** - Preserve scientific semantics
- ✅ **Real examples** - Use proven conversion patterns
- ✅ **User validation required** - Scripts need review before running

The server **generates starting points**, not final solutions. Users must review and adapt generated scripts for their specific data.

## Key Differences from Generic Converters

1. **Dataset-specific templates** - Each generated script is based on a real, working example
2. **Scientific semantics preserved** - Temporal structure, field locations, mesh types respected
3. **Explicit customization points** - Clear markers show what needs adaptation
4. **Pattern-based** - Leverages documented conversion patterns
5. **Validation included** - Built-in checks for common issues

## Troubleshooting

### Server won't start
- Check Python path includes PLAID and plaid-skills
- Verify MCP SDK is installed: `pip install mcp`
- Check server.py is executable

### Generated script has errors
- This is expected! Scripts are templates
- Search for `CUSTOMIZATION_REQUIRED` comments
- Update placeholder paths
- Adapt generator function to your data format

### plaid-skills examples not found
- Verify plaid-skills is cloned at `/home/sagemaker-user/softs/plaid-skills`
- Update `PLAID_SKILLS_PATH` in `tools/conversion_tools.py` if different location

### Dataset loading fails
- Ensure dataset was created with PLAID storage API
- Check backend compatibility
- Verify problem definitions exist if loading them

## Contributing

To add new conversion examples or patterns:

1. Add examples to plaid-skills repository
2. Update metadata in `tools/conversion_tools.py`
3. Test generation with new example

## Resources

- **PLAID Library**: https://github.com/PLAID-lib/plaid
- **PLAID Documentation**: https://plaid-lib.readthedocs.io/
- **plaid-skills**: https://github.com/PLAID-lib/plaid-skills
- **MCP Protocol**: https://modelcontextprotocol.io/

## License

BSD 3-Clause License (same as PLAID)

## Support

For issues related to:
- **MCP server**: Open issue in this repository
- **PLAID library**: https://github.com/PLAID-lib/plaid/issues
- **Conversion patterns**: https://github.com/PLAID-lib/plaid-skills/issues
