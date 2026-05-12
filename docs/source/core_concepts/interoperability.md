---
title: Interoperability
---

# Interoperability

- [CGNS standard](https://cgns.org/): the mesh/field containers align with CGNS conventions for bases, zones, elements and locations.
- Storage backends: PLAID supports backend-based interoperability for `cgns`, `hf_datasets`, and `zarr` through `plaid.storage` converters.
- Hugging Face integration is provided through the storage APIs (`push_to_hub`, `download_from_hub`, `init_streaming_from_hub`).


