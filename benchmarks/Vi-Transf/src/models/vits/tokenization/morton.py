import torch


def splitBy2(n):
    n = n & 0x00000000FFFFFFFF
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2)) & 0x3333333333333333
    n = (n | (n << 1)) & 0x5555555555555555
    return n


def compute_2dmorton_order(point_cloud):
    assert point_cloud.shape[1] == 2, "Point cloud must be 2D"

    min_xy = point_cloud.min(axis=0).values.to(torch.float64)
    max_xy = point_cloud.max(axis=0).values.to(torch.float64)

    bounding_box_size = (max_xy - min_xy).max()
    leaf_size = bounding_box_size / (2**32 - 1)

    origin = min_xy - 0.5 * leaf_size

    with torch.no_grad():
        quantized_point_cloud = torch.floor((point_cloud - origin) / leaf_size)

    ij_split_by_2 = splitBy2(quantized_point_cloud.to(torch.int64))
    morton_code = ij_split_by_2[:, 0] | ij_split_by_2[:, 1] << 1

    morton_order = torch.argsort(morton_code.to(torch.uint64))

    return morton_order


def splitBy3(n):
    n = n & 0b0000000000000000000000000000000000000000000111111111111111111111
    n = (
        n | n << 32
    ) & 0b0000000000011111000000000000000000000000000000001111111111111111
    n = (
        n | n << 16
    ) & 0b0000000000011111000000000000000011111111000000000000000011111111
    n = (
        n | n << 8
    ) & 0b0001000000001111000000001111000000001111000000001111000000001111
    n = (
        n | n << 4
    ) & 0b0001000011000011000011000011000011000011000011000011000011000011
    n = (
        n | n << 2
    ) & 0b0001001001001001001001001001001001001001001001001001001001001001
    return n


def compute_3dmorton_order(point_cloud):
    assert point_cloud.shape[1] == 3, "Point cloud must be 3D"

    min_xyz = point_cloud.min(axis=0).values.to(torch.float64)
    max_xyz = point_cloud.max(axis=0).values.to(torch.float64)

    bounding_box_size = (max_xyz - min_xyz).max()
    leaf_size = bounding_box_size / (2**21 - 1)

    origin = min_xyz - 0.5 * leaf_size

    with torch.no_grad():
        quantized_point_cloud = torch.floor((point_cloud - origin) / leaf_size)

    ijk_split_by_3 = splitBy3(quantized_point_cloud.to(torch.int64))
    morton_code = (
        ijk_split_by_3[:, 0] | ijk_split_by_3[:, 1] << 1 | ijk_split_by_3[:, 2] << 2
    )

    morton_order = torch.argsort(morton_code)

    return morton_order


def compute_morton_order(point_cloud):
    if point_cloud.shape[1] == 2:
        return compute_2dmorton_order(point_cloud)
    elif point_cloud.shape[1] == 3:
        return compute_3dmorton_order(point_cloud)
    else:
        raise ValueError(
            f"Only works for 2d or 3d pointclouds, this won't work in dimension {point_cloud.shape[1]}"
        )
