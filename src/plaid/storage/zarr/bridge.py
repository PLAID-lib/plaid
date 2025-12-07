import zarr

from typing import Any

from plaid.storage.common.bridge import to_sample_dict, to_plaid_sample




# class DictView:
#     def __init__(self, dataset, flat_cst, cgns_types):
#         self.dataset = dataset
#         self.flat_cst = flat_cst
#         self.cgns_types = cgns_types

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         var_sample_dict = self.dataset[idx]
#         return to_sample_dict(
#             var_sample_dict,
#             self.flat_cst,
#             self.cgns_types
#         )
# class PLAIDView:
#     def __init__(self, dataset, flat_cst, cgns_types):
#         self.dataset = dataset
#         self.flat_cst = flat_cst
#         self.cgns_types = cgns_types

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         var_sample_dict = self.dataset[idx]
#         sample_dict = to_sample_dict(
#             var_sample_dict,
#             self.flat_cst,
#             self.cgns_types,
#         )
#         return to_plaid_sample(sample_dict, self.cgns_types)


def unflatten_zarr_key(key: str) -> str:
    return key.replace("__", "/")


def to_var_sample_dict(zarr_dataset: zarr.core.group.Group, idx: int) -> dict:
    return zarr_dataset[idx]


def sample_to_var_sample_dict(
    zarr_sample: dict
) -> dict:
    return zarr_sample
