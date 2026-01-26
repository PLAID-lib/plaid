# Implementation Plan: WebDataset Storage Backend for PLAID

## [Overview]

Implementation of a WebDataset storage backend for the PLAID library to provide tar-based, streaming-friendly dataset storage with seamless HuggingFace Hub integration.

The WebDataset backend will be added as the fourth storage option alongside the existing cgns, hf_datasets, and zarr backends. WebDataset uses tar archives where samples sharing the same basename (stripped of extensions) belong together, making it ideal for streaming large physics datasets. This format aligns well with PLAID's architecture and provides efficient I/O for both local and cloud storage.

The implementation follows the established backend pattern in PLAID: a module directory (`src/plaid/storage/webdataset/`) containing reader, writer, and bridge components, registered in the central registry, with full test coverage and documentation. The backend will support all standard PLAID operations: local disk save/load, HuggingFace Hub push/download, streaming access, and bidirectional conversion between WebDataset format and PLAID Sample objects.

Key design considerations:
- Each PLAID sample becomes a set of files in a tar archive with shared basename (e.g., `sample_000000000.json`, `sample_000000000.npy`)
- Variable features stored as individual .npy files for efficient array storage
- Metadata and constant features stored in JSON format
- Split-based tar sharding for scalability
- Compatible with HuggingFace Hub's tar file hosting
- Streaming support via webdataset library's pipeline architecture

## [Types]

Type system changes to support WebDataset format and integration with PLAID's type system.

**New Type Definitions:**

```python
# In src/plaid/storage/webdataset/reader.py
from typing import Iterator, Iterable, Any, Optional, Union
from pathlib import Path
import webdataset as wds

# WebDataset pipeline type (returned by wds.WebDataset)
WebDatasetPipeline = wds.WebDataset

# Sample dictionary type for WebDataset
WebDatasetSample = dict[str, Any]  # Keys: feature paths, Values: numpy arrays or None
```

**Modified Type Definitions:**

```python
# In src/plaid/storage/registry.py - BackendSpec already supports Optional callables
# No changes needed to BackendSpec dataclass definition
```

**Type Annotations:**

All new functions will use complete type annotations following the existing codebase patterns:
- `Union[str, Path]` for file paths
- `dict[str, Any]` for sample dictionaries
- `Optional[...]` for optional parameters
- `Callable[..., Generator[Sample, None, None]]` for generator functions
- `Iterator[dict[str, Any]]` for WebDataset iteration

## [Files]

File modifications required to implement the WebDataset backend.

**New Files to Create:**

1. `src/plaid/storage/webdataset/__init__.py`
   - Purpose: Package initialization and public API exports
   - Exports: `init_datasetdict_from_disk`, `download_datasetdict_from_hub`, `init_datasetdict_streaming_from_hub`, `generate_datasetdict_to_disk`, `push_local_datasetdict_to_hub`, `configure_dataset_card`, `to_var_sample_dict`, `sample_to_var_sample_dict`

2. `src/plaid/storage/webdataset/reader.py`
   - Purpose: Dataset loading and streaming functionality
   - Key components: `WebDatasetDict` class (wrapper for split-based access), `init_datasetdict_from_disk`, `download_datasetdict_from_hub`, `init_datasetdict_streaming_from_hub`, helper functions for tar file iteration

3. `src/plaid/storage/webdataset/writer.py`
   - Purpose: Dataset generation and Hub upload
   - Key components: `generate_datasetdict_to_disk`, `push_local_datasetdict_to_hub`, `configure_dataset_card`, tar archive creation logic, sample serialization

4. `src/plaid/storage/webdataset/bridge.py`
   - Purpose: Conversion between WebDataset format and PLAID samples
   - Key components: `to_var_sample_dict`, `sample_to_var_sample_dict`, feature extraction helpers

5. `tests/storage/test_webdataset.py`
   - Purpose: Unit tests for WebDataset backend
   - Test coverage: reader/writer functionality, conversion operations, edge cases, integration with registry

6. `docs/source/core_concepts/webdataset_backend.md`
   - Purpose: Documentation for WebDataset backend usage
   - Content: Format specification, usage examples, performance characteristics, comparison with other backends

**Existing Files to Modify:**

1. `src/plaid/storage/registry.py`
   - Line ~70: Add "webdataset" entry to BACKENDS dict
   - Specify all required backend functions following the pattern of zarr/hf_datasets entries

2. `src/plaid/storage/__init__.py`
   - No changes needed (uses registry dynamically)

3. `pyproject.toml`
   - Line ~37 (dependencies section): Add `"webdataset"` to dependencies list

4. `docs/source/tutorials/storage.md`
   - Add WebDataset backend to the list of available backends
   - Include WebDataset in the example loops: `all_backends = ["hf_datasets", "cgns", "zarr", "webdataset"]`

5. `tests/storage/test_storage.py`
   - Add test method: `test_webdataset` following the pattern of `test_zarr`
   - Add webdataset to registry test assertions (line ~230)

**Configuration Files:**

1. `.pre-commit-config.yaml` - No changes needed (linting applies automatically)
2. `ruff.toml` - No changes needed
3. `pyrightconfig.json` - No changes needed

## [Functions]

Function-level implementation details for the WebDataset backend.

**New Functions in `src/plaid/storage/webdataset/reader.py`:**

1. `init_datasetdict_from_disk(path: Union[str, Path]) -> dict[str, WebDatasetDict]`
   - Purpose: Load WebDataset from local tar files
   - Returns: Dictionary mapping split names to WebDatasetDict objects
   - Logic: Scan for tar files in `path/data/`, create WebDatasetDict wrappers

2. `download_datasetdict_from_hub(repo_id: str, local_dir: Union[str, Path], split_ids: Optional[dict[str, list[int]]] = None, features: Optional[list[str]] = None, overwrite: bool = False) -> None`
   - Purpose: Download WebDataset from HuggingFace Hub
   - Uses: `snapshot_download` from huggingface_hub
   - Logic: Download tar files with filtering patterns if split_ids/features specified

3. `init_datasetdict_streaming_from_hub(repo_id: str, split_ids: Optional[dict[str, list[int]]] = None, features: Optional[list[str]] = None) -> dict[str, wds.WebDataset]`
   - Purpose: Create streaming dataset from Hub
   - Returns: Dictionary of split names to streaming WebDataset pipelines
   - Logic: Use wds.WebDataset with Hub URLs, apply filters for split_ids/features

4. `_create_webdataset_pipeline(tar_path: str, features: Optional[list[str]] = None) -> wds.WebDataset`
   - Purpose: Helper to create WebDataset pipeline with decoding
   - Returns: Configured WebDataset pipeline
   - Logic: Set up .decode(), .to_tuple(), .map() operations

**New Functions in `src/plaid/storage/webdataset/writer.py`:**

1. `generate_datasetdict_to_disk(output_folder: Union[str, Path], generators: dict[str, Callable], variable_schema: dict, gen_kwargs: Optional[dict] = None, num_proc: int = 1, verbose: bool = False) -> None`
   - Purpose: Generate and save WebDataset tar files from sample generators
   - Logic: Iterate samples, serialize to numpy/json, write to tar archives
   - Supports: Both sequential and parallel (multiprocess with sharding) modes

2. `push_local_datasetdict_to_hub(repo_id: str, local_dir: Union[str, Path], num_workers: int = 1) -> None`
   - Purpose: Upload local WebDataset to HuggingFace Hub
   - Uses: `HfApi.upload_large_folder`
   - Logic: Upload tar files with appropriate patterns

3. `configure_dataset_card(repo_id: str, infos: dict, local_dir: Optional[Union[str, Path]] = None, viewer: Optional[bool] = None, pretty_name: Optional[str] = None, dataset_long_description: Optional[str] = None, illustration_urls: Optional[list[str]] = None, arxiv_paper_urls: Optional[list[str]] = None) -> None`
   - Purpose: Create and push dataset card to Hub
   - Logic: Generate README.md with metadata, usage examples, format description

4. `_write_sample_to_tar(tar_writer: wds.TarWriter, sample: Sample, var_features_keys: list[str], sample_idx: int) -> None`
   - Purpose: Helper to serialize one sample to tar
   - Logic: Convert sample to dict, write .npy files for arrays, .json for metadata

**New Functions in `src/plaid/storage/webdataset/bridge.py`:**

1. `to_var_sample_dict(wds_sample: dict[str, Any], idx: int, features: Optional[list[str]]) -> dict[str, Any]`
   - Purpose: Extract variable features from WebDataset sample
   - Returns: Dictionary of feature paths to values
   - Logic: Filter and return requested features from sample dict

2. `sample_to_var_sample_dict(wds_sample: dict[str, Any]) -> dict[str, Any]`
   - Purpose: Convert raw WebDataset sample to variable sample dict
   - Returns: Processed sample dictionary
   - Logic: Pass through (WebDataset samples are already in correct format)

3. `_decode_sample(sample: dict[str, bytes]) -> dict[str, Any]`
   - Purpose: Helper to decode WebDataset sample bytes to numpy arrays
   - Logic: Deserialize .npy files, parse .json metadata

**Modified Functions:**

None - all integration is through the registry system, so no existing functions need modification.

## [Classes]

Class definitions and modifications for WebDataset backend support.

**New Classes:**

1. `WebDatasetDict` (in `src/plaid/storage/webdataset/reader.py`)
   - Purpose: Wrapper class for WebDataset tar archives providing dict-like split access
   - Inherits: None (standalone class, similar to ZarrDataset)
   - Attributes:
     - `path: Union[str, Path]` - Path to dataset root
     - `split_tar_paths: dict[str, Path]` - Mapping of split names to tar file paths
     - `_extra_fields: dict[str, Any]` - Additional metadata
   - Methods:
     - `__init__(self, path: Union[str, Path], split_tar_paths: dict[str, Path], **kwargs)`
     - `__getitem__(self, split: str) -> wds.WebDataset` - Return WebDataset for a split
     - `__len__(self) -> int` - Return number of splits
     - `__iter__(self) -> Iterator[tuple[str, wds.WebDataset]]` - Iterate over splits
     - `__getattr__(self, name: str) -> Any` - Access extra fields
     - `__setattr__(self, name: str, value: Any) -> None` - Set extra fields
     - `__repr__(self) -> str` - String representation
   - Purpose: Provides consistent interface matching ZarrDataset and CGNSDataset patterns

2. `WebDatasetWrapper` (in `src/plaid/storage/webdataset/reader.py`)
   - Purpose: Wrapper for individual WebDataset splits with indexing support
   - Inherits: None
   - Attributes:
     - `wds_pipeline: wds.WebDataset` - Underlying WebDataset pipeline
     - `path: Union[str, Path]` - Path to tar file
     - `_cache: Optional[list]` - Optional cache for random access
     - `ids: np.ndarray` - Array of sample IDs
   - Methods:
     - `__init__(self, tar_path: Union[str, Path], cache: bool = False)`
     - `__getitem__(self, idx: int) -> dict[str, Any]` - Get sample by index
     - `__len__(self) -> int` - Return number of samples
     - `__iter__(self) -> Iterator[dict[str, Any]]` - Iterate over samples
   - Purpose: Enable random access to WebDataset samples (required for PLAID's indexing pattern)

**Modified Classes:**

None - WebDataset backend integrates via the BackendSpec dataclass which already exists.

**Class Relationships:**

```
registry.BackendSpec
    └─> Configured with webdataset functions
        ├─> reader.init_datasetdict_from_disk → WebDatasetDict
        ├─> reader.download_datasetdict_from_hub → None
        ├─> reader.init_datasetdict_streaming_from_hub → dict[str, wds.WebDataset]
        ├─> writer.generate_datasetdict_to_disk → None
        ├─> writer.push_local_datasetdict_to_hub → None
        ├─> writer.configure_dataset_card → None
        ├─> bridge.to_var_sample_dict → dict[str, Any]
        └─> bridge.sample_to_var_sample_dict → dict[str, Any]
```

## [Dependencies]

Dependency additions and version requirements for WebDataset backend.

**New Dependencies:**

1. `webdataset` (PyPI package)
   - Version requirement: `>=0.2.0` (stable release with core features)
   - Reason: Core library for WebDataset format handling
   - Features used:
     - `wds.TarWriter` for tar archive creation
     - `wds.WebDataset` for reading and streaming
     - `.decode()`, `.to_tuple()`, `.map()` pipeline operations
   - Add to `pyproject.toml` dependencies list

**Existing Dependencies (no changes):**

- `huggingface_hub` - Already present, used for Hub integration
- `numpy` - Already present, used for array serialization
- `pyyaml` - Already present, used for metadata
- `tqdm` - Already present, used for progress bars

**Installation:**

Add to `pyproject.toml` line ~37:
```python
dependencies = [
    "tqdm",
    "pyyaml",
    "pycgns",
    "zarr",
    "scikit-learn",
    "datasets",
    "numpy",
    "matplotlib",
    "pydantic",
    "webdataset>=0.2.0",  # ADD THIS LINE
]
```

**Compatibility:**

- Python 3.11-3.13: webdataset supports these versions
- No conflicts with existing dependencies
- Optional GPU acceleration not required (webdataset is pure Python for basic operations)

## [Testing]

Comprehensive testing strategy for WebDataset backend implementation.

**New Test Files:**

1. `tests/storage/test_webdataset.py`
   - Purpose: WebDataset-specific unit tests
   - Structure:
     ```python
     class TestWebDataset:
         def test_write_and_read_local(self, tmp_path, generator_split, infos, problem_definition)
         def test_sample_iteration(self, tmp_path, generator_split, infos, problem_definition)
         def test_converter_operations(self, tmp_path, generator_split, infos, problem_definition)
         def test_feature_filtering(self, tmp_path, generator_split, infos, problem_definition)
         def test_webdataset_dict_class(self, tmp_path, generator_split, infos, problem_definition)
         def test_parallel_generation(self, tmp_path, generator_split_with_kwargs, gen_kwargs, infos, problem_definition)
     ```
   - Coverage targets: >90% line coverage for webdataset module

**Existing Test File Modifications:**

1. `tests/storage/test_storage.py`
   - Add method: `test_webdataset(self, tmp_path, generator_split, infos, problem_definition)`
   - Location: After `test_cgns` method (around line 220)
   - Content: Following the pattern of `test_zarr`, test basic operations:
     - save_to_disk with webdataset backend
     - init_from_disk and sample conversion
     - plaid_sample reconstruction and feature access
     - converter.to_dict and converter.sample_to_dict operations
   
2. `tests/storage/test_storage.py` - Registry test
   - Line ~240: Add to registry test:
     ```python
     assert "webdataset" in backends
     webdataset_module = registry.get_backend("webdataset")
     assert webdataset_module is not None
     ```

**Test Fixtures:**

Reuse existing fixtures from `tests/conftest.py` and `tests/storage/test_storage.py`:
- `samples` - Sample PLAID objects
- `infos` - Dataset metadata
- `problem_definition` - Problem definition object
- `dataset` - PLAID Dataset
- `main_splits` - Split configuration
- `generator_split` - Split-based generators
- `generator_split_with_kwargs` - Generators with kwargs for parallel processing
- `gen_kwargs` - Generator arguments for parallel mode

**Test Coverage Requirements:**

1. Core functionality:
   - ✓ Generate dataset to disk (sequential and parallel)
   - ✓ Load dataset from disk
   - ✓ Iterate over samples
   - ✓ Convert samples to PLAID format
   - ✓ Feature extraction and filtering

2. Edge cases:
   - ✓ Empty datasets
   - ✓ None values in features
   - ✓ Large arrays
   - ✓ Unicode strings
   - ✓ Missing features

3. Error handling:
   - ✓ Invalid tar files
   - ✓ Missing split files
   - ✓ Corrupted data
   - ✓ Feature key mismatches

4. Integration:
   - ✓ Registry integration
   - ✓ Converter class operations
   - ✓ Problem definition compatibility

**Validation Strategy:**

Run existing test suite with new backend to ensure no regressions:
```bash
pytest tests/storage/test_storage.py::Test_Storage::test_webdataset -v
pytest tests/storage/test_webdataset.py -v
pytest tests/storage/test_storage.py::Test_Storage::test_registry -v
```

## [Implementation Order]

Step-by-step implementation sequence to minimize conflicts and ensure successful integration.

**Phase 1: Foundation (Dependencies & Structure)**

1. Update `pyproject.toml`
   - Add `webdataset>=0.2.0` to dependencies
   - Rationale: Required before any code can import webdataset

2. Create directory structure
   - Create `src/plaid/storage/webdataset/` directory
   - Create empty `__init__.py`, `reader.py`, `writer.py`, `bridge.py` files
   - Rationale: Establishes module structure for imports

**Phase 2: Core Bridge Layer**

3. Implement `src/plaid/storage/webdataset/bridge.py`
   - Implement `to_var_sample_dict` function
   - Implement `sample_to_var_sample_dict` function
   - Add docstrings and type hints
   - Rationale: Bridge functions are dependencies for reader/writer

4. Implement `src/plaid/storage/common/bridge.py` helpers (if needed)
   - No changes required (existing helpers sufficient)
   - Rationale: Reuse existing flatten_path/unflatten_path utilities

**Phase 3: Writer Implementation**

5. Implement `src/plaid/storage/webdataset/writer.py` - Basic structure
   - Implement `_write_sample_to_tar` helper function
   - Implement `generate_datasetdict_to_disk` (sequential mode only)
   - Rationale: Writing capability needed before testing read operations

6. Implement `src/plaid/storage/webdataset/writer.py` - Advanced features
   - Add parallel processing support to `generate_datasetdict_to_disk`
   - Implement `push_local_datasetdict_to_hub`
   - Implement `configure_dataset_card`
   - Rationale: Complete write functionality before reader

**Phase 4: Reader Implementation**

7. Implement `src/plaid/storage/webdataset/reader.py` - Core classes
   - Implement `WebDatasetWrapper` class
   - Implement `WebDatasetDict` class
   - Rationale: Dataset wrapper classes needed for consistent interface

8. Implement `src/plaid/storage/webdataset/reader.py` - Load functions
   - Implement `init_datasetdict_from_disk`
   - Implement `_create_webdataset_pipeline` helper
   - Rationale: Local loading enables testing without Hub dependency

9. Implement `src/plaid/storage/webdataset/reader.py` - Hub functions
   - Implement `download_datasetdict_from_hub`
   - Implement `init_datasetdict_streaming_from_hub`
   - Rationale: Hub integration completes reader functionality

**Phase 5: Integration**

10. Implement `src/plaid/storage/webdataset/__init__.py`
    - Export all public functions
    - Add module docstring
    - Rationale: Establishes public API

11. Update `src/plaid/storage/registry.py`
    - Add webdataset BackendSpec to BACKENDS dict
    - Import webdataset module
    - Rationale: Register backend for system-wide availability

**Phase 6: Testing**

12. Create `tests/storage/test_webdataset.py`
    - Implement core test class and methods
    - Test write and read operations
    - Test converter operations
    - Rationale: Dedicated tests for WebDataset-specific functionality

13. Update `tests/storage/test_storage.py`
    - Add `test_webdataset` method
    - Update registry test to include webdataset
    - Rationale: Integration tests with existing test infrastructure

14. Run full test suite
    - Execute: `pytest tests/storage/ -v`
    - Verify: No regressions in existing backends
    - Rationale: Ensure system-wide compatibility

**Phase 7: Documentation**

15. Create `docs/source/core_concepts/webdataset_backend.md`
    - Document WebDataset format specification
    - Provide usage examples
    - Compare with other backends
    - Rationale: User-facing documentation

16. Update `docs/source/tutorials/storage.md`
    - Add "webdataset" to all_backends list
    - Add WebDataset examples to tutorial
    - Rationale: Integration into existing documentation

**Phase 8: Code Quality**

17. Run linting and formatting
    - Execute: `ruff check src/plaid/storage/webdataset/`
    - Execute: `ruff format src/plaid/storage/webdataset/`
    - Fix any issues
    - Rationale: Ensure code quality standards

18. Run type checking
    - Execute: `pyright src/plaid/storage/webdataset/`
    - Fix any type errors
    - Rationale: Ensure type safety

19. Final validation
    - Run complete test suite: `pytest tests/ -v`
    - Check test coverage: `pytest tests/storage/ --cov=src/plaid/storage/webdataset`
    - Verify: Coverage >90%
    - Rationale: Final quality gate before completion

**Critical Path Dependencies:**

```
1. pyproject.toml → 2. Directory structure → 3. Bridge layer
                                                    ↓
                                           5-6. Writer implementation
                                                    ↓
                                           7-9. Reader implementation
                                                    ↓
                                           10-11. Integration
                                                    ↓
                                           12-14. Testing
                                                    ↓
                                           15-16. Documentation
                                                    ↓
                                           17-19. Quality checks
```

**Estimated Implementation Time:**

- Phase 1-2: 1 hour (setup)
- Phase 3: 2 hours (bridge layer)
- Phase 4: 4 hours (writer)
- Phase 5: 4 hours (reader)
- Phase 6: 2 hours (integration)
- Phase 7: 4 hours (testing)
- Phase 8: 2 hours (documentation)
- Phase 9: 1 hour (quality)

**Total: ~20 hours**

**Risk Mitigation:**

- Test each phase independently before proceeding
- Use existing zarr/hf_datasets backends as reference implementations
- Create checkpoints after each major phase (commit to version control)
- If Hub integration issues arise, implement local-only first, then add Hub support