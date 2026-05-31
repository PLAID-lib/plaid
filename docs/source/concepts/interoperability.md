---
title: Interoperability
---

# Interoperability

- [CGNS standard](https://cgns.org/): the mesh/field containers align with CGNS conventions for bases, zones, elements and locations.
- Storage backends: PLAID supports backend-based interoperability for persistent `cgns`, `hf_datasets`, and `zarr` datasets through `plaid.storage` converters.
- Hugging Face integration is provided through the storage APIs (`push_to_hub`, `download_from_hub`, `init_streaming_from_hub`).
- The `in_memory` backend is registered for in-process sample storage, but it does not implement disk or Hub persistence.
- `plaid-viewer` uses the same storage/converter layer to browse local and streamed Hub datasets uniformly.


