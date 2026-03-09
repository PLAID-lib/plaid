# PLAID MCP Server - Quick Start Guide

## 5-Minute Setup

### 1. Install MCP SDK

```bash
pip install mcp
```

### 2. Test the Server

```bash
cd /home/sagemaker-user/softs/plaid/plaid-mcp-server
python server.py
```

The server should start and wait for MCP protocol messages on stdio.

### 3. Configure Your MCP Client

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "plaid": {
      "command": "python",
      "args": ["/home/sagemaker-user/softs/plaid/plaid-mcp-server/server.py"],
      "env": {
        "PYTHONPATH": "/home/sagemaker-user/softs/plaid:/home/sagemaker-user/softs/plaid-mcp-server"
      }
    }
  }
}
```

## Quick Usage Examples

### Generate a Conversion Script

Ask your MCP client (e.g., Claude):

> "I have HDF5 files with structured 2D grids and cell-centered data. Can you generate a PLAID conversion script for me?"

The assistant will use the `generate_conversion_script` tool to create a customized template.

### Explore Conversion Examples

> "What conversion examples are available for temporal data with unstructured meshes?"

The assistant will use `find_similar_conversion` to recommend templates.

### Load and Inspect a Dataset

> "Load the PLAID dataset from /path/to/dataset and show me the first sample from the train split"

The assistant will:
1. Use `init_from_disk` to load the dataset
2. Use `get_sample` to retrieve and display sample information

## Key Tools to Know

### Most Important
- **generate_conversion_script** - Creates customized conversion scripts
- **find_similar_conversion** - Finds matching templates
- **analyze_raw_data_structure** - Analyzes your data files

### Dataset Operations
- **init_from_disk** - Load PLAID datasets
- **get_sample** - Access individual samples
- **get_dataset_info** - Query dataset metadata

### Patterns & Examples
- **list_conversion_examples** - Show all available templates
- **get_conversion_example** - View full example code
- **get_conversion_pattern** - Read pattern documentation

## Typical Workflow

1. **Describe your data** to the AI assistant
2. **Analyze** your data structure (optional but helpful)
3. **Generate** a conversion script
4. **Review** the generated script (look for CUSTOMIZATION_REQUIRED markers)
5. **Test** on a small subset
6. **Run** full conversion

## Example Conversation

**You:** "I have a dataset of static CFD simulations in HDF5 format with structured grids. Each file contains velocity and pressure fields at cell centers. Can you help me convert this to PLAID?"

**Assistant:** Uses `find_similar_conversion` and suggests the pdebench example, then uses `generate_conversion_script` to create a starter script.

**You:** "Can you show me the pdebench example so I can see how it's structured?"

**Assistant:** Uses `get_conversion_example` to show the full template.

**You:** "Great! Now analyze my data at /data/cfd_simulations/"

**Assistant:** Uses `analyze_raw_data_structure` to inspect your files and provide suggestions.

## Troubleshooting

**Q: Server won't start**
- Ensure `pip install mcp` is installed
- Check PYTHONPATH includes PLAID library
- Verify Python 3.11+

**Q: Generated script has errors**
- This is normal! Scripts are templates
- Look for `CUSTOMIZATION_REQUIRED` comments
- Adapt the generator function to your data format

**Q: Can't find plaid-skills examples**
- Verify plaid-skills is at `/home/sagemaker-user/softs/plaid-skills`
- Update path in `tools/conversion_tools.py` if different

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore [plaid-skills](https://github.com/PLAID-lib/plaid-skills) for conversion patterns
- Check [PLAID docs](https://plaid-lib.readthedocs.io/) for API reference

## Getting Help

- Tool descriptions available via MCP client
- Each tool returns helpful error messages
- Check PLAID and plaid-skills documentation
- Open issues on respective GitHub repositories
