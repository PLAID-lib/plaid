"""Direct CGNS/VTK conversion functions.

This fuction do not require any class of plaid.
Please keep this module free of plaid dependencies to make it usable in
the ParaView plugin without forcing users to install plaid.
"""

from typing import Any, List, Optional

import numpy as np

# Direct CGNS -> VTK conversion tables.  These maps deliberately use only CGNS
# element numbers and VTK cell numbers so the converter below does not depend on
CGNSNumberToVtkNumber = {
    2: 1,  # NODE     -> VTK_VERTEX
    3: 3,  # BAR_2    -> VTK_LINE
    4: 21,  # BAR_3    -> VTK_QUADRATIC_EDGE
    5: 5,  # TRI_3    -> VTK_TRIANGLE
    6: 22,  # TRI_6    -> VTK_QUADRATIC_TRIANGLE
    7: 9,  # QUAD_4   -> VTK_QUAD
    8: 23,  # QUAD_8   -> VTK_QUADRATIC_QUAD
    9: 28,  # QUAD_9   -> VTK_BIQUADRATIC_QUAD
    10: 10,  # TETRA_4  -> VTK_TETRA
    11: 24,  # TETRA_10 -> VTK_QUADRATIC_TETRA
    12: 14,  # PYRA_5   -> VTK_PYRAMID
    14: 13,  # PENTA_6  -> VTK_WEDGE
    15: 26,  # PENTA_15 -> VTK_QUADRATIC_WEDGE
    16: 32,  # PENTA_18 -> VTK_BIQUADRATIC_QUADRATIC_WEDGE
    17: 12,  # HEXA_8   -> VTK_HEXAHEDRON
    18: 25,  # HEXA_20  -> VTK_QUADRATIC_HEXAHEDRON
    19: 29,  # HEXA_27  -> VTK_TRIQUADRATIC_HEXAHEDRON
    21: 27,  # PYRA_13  -> VTK_QUADRATIC_PYRAMID
}

CGNSNumberOfNodes = {
    2: 1,
    3: 2,
    4: 3,
    5: 3,
    6: 6,
    7: 4,
    8: 8,
    9: 9,
    10: 4,
    11: 10,
    12: 5,
    13: 14,
    14: 6,
    15: 15,
    16: 18,
    17: 8,
    18: 20,
    19: 27,
    21: 13,
}

# CGNS and VTK share the same ordering for the linear and most quadratic cells
# used here.  The entries below cover the higher-order cells for which Muscat's
# CGNS bridge already documents an ordering difference and VTK supports the cell.
CGNSNumberToVtkPermutation = {
    15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11],
    16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11, 15, 16, 17],
    18: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15],
    19: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        12,
        13,
        14,
        15,
        24,
        22,
        21,
        23,
        20,
        25,
        26,
    ],
}

VtkNumberToCGNSNumber = {
    vtkNumber: cgnsNumber for cgnsNumber, vtkNumber in CGNSNumberToVtkNumber.items()
}

CGNSNumberDimension = {
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 3,
    11: 3,
    12: 3,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 3,
    21: 3,
}


def _vtk_cell_to_cgns(cell_type: int, points: np.ndarray) -> tuple[int, np.ndarray]:
    """Convert one VTK cell type and connectivity to CGNS numbering."""
    if cell_type not in VtkNumberToCGNSNumber:
        raise NotImplementedError(
            f"VTK cell type {cell_type} is not supported by direct CGNS conversion"
        )

    cgns_type = VtkNumberToCGNSNumber[cell_type]
    permutation = CGNSNumberToVtkPermutation.get(cgns_type)
    if permutation is not None:
        inverse = np.argsort(np.asarray(permutation, dtype=np.int64))
        points = points[inverse]
    return cgns_type, points


def _vtk_numpy_array(array, numpy_support) -> np.ndarray:
    """Return a VTK data array as a NumPy array, including string arrays."""
    if array.IsA("vtkStringArray"):
        return np.asarray(
            [array.GetValue(index) for index in range(array.GetNumberOfValues())]
        )
    return np.asarray(numpy_support.vtk_to_numpy(array))


def _vtk_array_is_binary_tag(array, numberOfTuples: int, numpy_support) -> bool:
    """Return whether a VTK array follows the binary character tag contract."""
    if array is None or array.GetNumberOfComponents() != 1:
        return False
    if array.GetDataTypeAsString() not in ["char", "signed char", "unsigned char"]:
        return False
    values = _vtk_numpy_array(array, numpy_support).ravel(order="C")
    if values.size != numberOfTuples:
        return False
    return bool(np.all((values == 0) | (values == 1)))


def _vtk_attributes_to_nodes(
    attributes,
    numpy_support,
    numberOfTuples: Optional[int] = None,
) -> list[list]:
    """Convert non-tag VTK data attributes to CGNS DataArray_t nodes."""
    nodes = []
    for index in range(attributes.GetNumberOfArrays()):
        array = attributes.GetArray(index)
        if array is None:
            continue
        if numberOfTuples is not None and _vtk_array_is_binary_tag(
            array, numberOfTuples, numpy_support
        ):
            continue
        name = array.GetName() or f"Array{index}"
        nodes.append([name, _vtk_numpy_array(array, numpy_support), [], "DataArray_t"])
    return nodes


def _vtk_tag_masks(
    attributes, numberOfTuples: int, numpy_support
) -> list[tuple[str, np.ndarray]]:
    """Return names and masks for binary character arrays in VTK attributes."""
    tags = []
    for index in range(attributes.GetNumberOfArrays()):
        array = attributes.GetArray(index)
        if not _vtk_array_is_binary_tag(array, numberOfTuples, numpy_support):
            continue
        name = array.GetName() or f"Array{index}"
        mask = _vtk_numpy_array(array, numpy_support).ravel(order="C").astype(bool)
        tags.append((name, mask))
    return tags


def _vtk_flow_solution(data_object, attributes_name: str, location: str, numpy_support):
    """Build a CGNS flow solution node from point or cell data."""
    attributes = getattr(data_object, attributes_name)()
    numberOfTuples = (
        data_object.GetNumberOfPoints()
        if location == "Vertex"
        else data_object.GetNumberOfCells()
    )
    arrays = _vtk_attributes_to_nodes(attributes, numpy_support, numberOfTuples)
    if not arrays:
        return None
    return [
        f"{location}Data",
        None,
        [["GridLocation", location, [], "GridLocation_t"], *arrays],
        "FlowSolution_t",
    ]


def _vtk_coordinates(data_object, numpy_support, points_in_2D=False) -> list[list]:
    """Convert VTK point coordinates into CGNS coordinate nodes."""
    points = data_object.GetPoints()
    if points is None:
        raise ValueError("VTK data object has no points")
    coordinates = np.asarray(numpy_support.vtk_to_numpy(points.GetData()))
    if coordinates.ndim != 2 or coordinates.shape[1] < 1:
        raise ValueError("VTK points must be a two-dimensional coordinate array")

    coordinate_nodes = []
    names = ("CoordinateX", "CoordinateY")
    if points_in_2D:
        names = names + ("CoordinateZ",)
    for index, name in enumerate(names):
        if index < coordinates.shape[1]:
            values = coordinates[:, index]
        else:
            values = np.zeros(coordinates.shape[0], dtype=coordinates.dtype)
        coordinate_nodes.append([name, values, [], "DataArray_t"])
    return coordinate_nodes


def _vtk_unstructured_elements(data_object, returnCellMapping: bool = False):
    """Convert VTK unstructured cells into CGNS Elements_t nodes."""
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    cellDimensions = np.empty(data_object.GetNumberOfCells(), dtype=np.int8)
    for cell_id in range(data_object.GetNumberOfCells()):
        cell_type = int(data_object.GetCellType(cell_id))
        cell_points = np.asarray(
            [
                data_object.GetCell(cell_id).GetPointId(point_id)
                for point_id in range(data_object.GetCell(cell_id).GetNumberOfPoints())
            ],
            dtype=np.int64,
        )
        cgns_type, cell_points = _vtk_cell_to_cgns(cell_type, cell_points)
        grouped.setdefault(cgns_type, []).append((cell_id, cell_points + 1))
        cellDimensions[cell_id] = CGNSNumberDimension[cgns_type]

    elements = []
    vtkToCgnsCell = np.empty(data_object.GetNumberOfCells(), dtype=np.int64)
    start = 1
    for cgns_type, indexedCells in grouped.items():
        connectivity = np.concatenate([cell for _, cell in indexedCells]).astype(
            np.int64, copy=False
        )
        end = start + len(indexedCells) - 1
        for localIndex, (cellId, _) in enumerate(indexedCells):
            vtkToCgnsCell[cellId] = start + localIndex
        elements.append(
            [
                f"Elements_{cgns_type}",
                np.asarray([cgns_type], dtype=np.int32),
                [
                    [
                        "ElementRange",
                        np.asarray([start, end], dtype=np.int32),
                        [],
                        "IndexRange_t",
                    ],
                    [
                        "ElementConnectivity",
                        connectivity,
                        [],
                        "DataArray_t",
                    ],
                ],
                "Elements_t",
            ]
        )
        start = end + 1
    if returnCellMapping:
        return elements, vtkToCgnsCell, cellDimensions
    return elements


def _cgns_character_value(value: str) -> np.ndarray:
    """Encode a CGNS character value as a one-byte NumPy array."""
    return np.asarray(list(value), dtype="|S1")


def _vtk_point_list(values: np.ndarray) -> list:
    """Build a CGNS PointList node from one-based indices."""
    pointList = np.asarray(values, dtype=np.int32).reshape((1, -1))
    return ["PointList", pointList, [], "IndexArray_t"]


def _vtk_bc_node(name: str, cgnsIds: np.ndarray, location: str) -> list:
    """Build a CGNS BC_t node for a point or boundary-cell tag."""
    return [
        name,
        _cgns_character_value("Null"),
        [
            _vtk_point_list(cgnsIds),
            [
                "GridLocation",
                _cgns_character_value(location),
                [],
                "GridLocation_t",
            ],
        ],
        "BC_t",
    ]


def _vtk_zone_subregion_node(
    name: str, cgnsIds: np.ndarray, location: str, topologicalDim: int
) -> list:
    """Build a CGNS ZoneSubRegion_t node for an element tag."""
    return [
        f"{name}_ZSR",
        np.asarray([[1, max(topologicalDim, 1)]], dtype=np.int32),
        [
            _vtk_point_list(cgnsIds),
            [
                "GridLocation",
                _cgns_character_value(location),
                [],
                "GridLocation_t",
            ],
            [
                "FamilyName",
                _cgns_character_value(name),
                [],
                "FamilyName_t",
            ],
        ],
        "ZoneSubRegion_t",
    ]


def _vtk_dataset_tag_nodes(
    data_object,
    numpy_support,
    vtkToCgnsCell: np.ndarray,
    cellDimensions: np.ndarray,
) -> list[list]:
    """Convert VTK binary character arrays into CGNS tag nodes."""
    children = []
    boundaryConditions = []
    for name, mask in _vtk_tag_masks(
        data_object.GetPointData(), data_object.GetNumberOfPoints(), numpy_support
    ):
        boundaryConditions.append(
            _vtk_bc_node(name, np.flatnonzero(mask) + 1, "Vertex")
        )

    topologicalDim = int(cellDimensions.max()) if cellDimensions.size else 0
    for name, mask in _vtk_tag_masks(
        data_object.GetCellData(), data_object.GetNumberOfCells(), numpy_support
    ):
        selected = np.flatnonzero(mask)
        selectedDimensions = cellDimensions[selected]
        cgnsIds = vtkToCgnsCell[selected]
        isBoundary = selected.size > 0 and np.all(
            selectedDimensions == topologicalDim - 1
        )
        if isBoundary and topologicalDim == 3:
            boundaryConditions.append(_vtk_bc_node(name, cgnsIds, "FaceCenter"))
        elif isBoundary and topologicalDim == 2:
            boundaryConditions.append(_vtk_bc_node(name, cgnsIds, "EdgeCenter"))
        else:
            children.append(
                _vtk_zone_subregion_node(name, cgnsIds, "CellCenter", topologicalDim)
            )

    if boundaryConditions:
        children.append(["ZoneBC", None, boundaryConditions, "ZoneBC_t"])
    return children


def _vtk_dataset_to_zone(
    data_object, name: str, numpy_support, points_in_2D=False
) -> list:
    """Convert one VTK dataset into a CGNS Zone_t node."""
    coordinates = _vtk_coordinates(
        data_object, numpy_support, points_in_2D=points_in_2D
    )
    children = [
        [
            "GridCoordinates",
            None,
            coordinates,
            "GridCoordinates_t",
        ]
    ]

    if data_object.IsA("vtkStructuredGrid"):
        dimensions = [0, 0, 0]
        data_object.GetDimensions(dimensions)
        dimensions = np.asarray(dimensions, dtype=np.int32)
        zone_size = np.asarray(
            [[dimensions[0], dimensions[1], dimensions[2]]], dtype=np.int32
        )
        zone_type = "Structured"
        vtkToCgnsCell = np.arange(1, data_object.GetNumberOfCells() + 1, dtype=np.int64)
        topologicalDim = max(int(np.count_nonzero(dimensions > 1)), 1)
        cellDimensions = np.full(
            data_object.GetNumberOfCells(), topologicalDim, dtype=np.int8
        )
    else:
        zone_size = np.asarray(
            [[data_object.GetNumberOfPoints(), data_object.GetNumberOfCells(), 0]],
            dtype=np.int32,
        )
        zone_type = "Unstructured"
        elements, vtkToCgnsCell, cellDimensions = _vtk_unstructured_elements(
            data_object, returnCellMapping=True
        )
        children.extend(elements)

    children.append(["ZoneType", zone_type, [], "ZoneType_t"])
    point_solution = _vtk_flow_solution(
        data_object, "GetPointData", "Vertex", numpy_support
    )
    cell_solution = _vtk_flow_solution(
        data_object, "GetCellData", "CellCenter", numpy_support
    )
    if point_solution is not None:
        children.append(point_solution)
    if cell_solution is not None:
        children.append(cell_solution)
    children.extend(
        _vtk_dataset_tag_nodes(
            data_object, numpy_support, vtkToCgnsCell, cellDimensions
        )
    )
    return [name, zone_size, children, "Zone_t"]


def _vtk_dataset_blocks(data_object) -> list[tuple[str, object]]:
    """Return named leaf datasets from a VTK data object."""
    if not data_object.IsA("vtkMultiBlockDataSet"):
        return [("Zone", data_object)]

    blocks = []
    for index in range(data_object.GetNumberOfBlocks()):
        block = data_object.GetBlock(index)
        if block is None:
            continue
        name = f"Block_{index}"
        metadata = data_object.GetMetaData(index)
        if metadata is not None and metadata.Has(data_object.NAME()):
            name = metadata.Get(data_object.NAME())
        blocks.extend(
            (f"{name}_{suffix}", leaf) for suffix, leaf in _vtk_dataset_blocks(block)
        )
    return blocks


def VtkToCGNSTree(data_object, points_in_2D=False) -> list:
    """Convert VTK datasets to a CGNS tree without an intermediate file.

    Args:
        data_object: VTK dataset or multiblock dataset.
        points_in_2D: If true, retain a zero-valued third coordinate array.

    Returns:
        A pyCGNS-style tree containing geometry, topology, and data arrays.

    Raises:
        TypeError: If ``data_object`` is not a supported VTK data object.
        ValueError: If a dataset has no points or a multiblock has no datasets.
        NotImplementedError: If a cell type cannot be represented in CGNS.
    """
    try:
        from paraview.vtk.util import numpy_support
    except Exception:
        from vtkmodules.util import numpy_support

    if not data_object.IsA("vtkDataSet") and not data_object.IsA(
        "vtkMultiBlockDataSet"
    ):
        raise TypeError("VtkToCGNSTree expects a VTK dataset or multiblock dataset")

    zones = [
        _vtk_dataset_to_zone(dataset, name, numpy_support, points_in_2D=points_in_2D)
        for name, dataset in _vtk_dataset_blocks(data_object)
    ]
    if not zones:
        raise ValueError("VTK multiblock dataset contains no data sets")
    field_data = data_object.GetFieldData() if data_object.IsA("vtkDataSet") else None
    global_children = []
    if field_data is not None:
        global_children = _vtk_attributes_to_nodes(field_data, numpy_support)

    bases = []
    if global_children:
        bases.append(["Global", None, global_children, "CGNSBase_t"])
    bases.append(["Base_2_2", np.asarray([2, 2], dtype=np.int32), zones, "CGNSBase_t"])
    return [
        "CGNSTree",
        None,
        bases,
        "CGNSTree_t",
    ]


def _cgns_children_by_label(node: list, label: str) -> List[list]:
    """Return direct children of a CGNS/Python node matching a label."""
    return [child for child in node[2] if len(child) > 3 and child[3] == label]


def _cgns_child_by_name(node: list, name: str) -> Optional[list]:
    """Return a direct child of a CGNS/Python node by name."""
    for child in node[2]:
        if child[0] == name:
            return child
    return None


def _cgns_value_as_string(node: Optional[list]) -> Optional[str]:
    """Decode a CGNS character-array node without using CGNS or Muscat helpers."""
    if node is None or node[1] is None:
        return None
    value = node[1]
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.kind in ["S", "U"]:
        return (
            b"".join(np.asarray(array, dtype="|S1").ravel(order="F").tolist())
            .decode("ascii", errors="ignore")
            .strip("\x00 ")
        )
    return str(value)


def _import_vtk_for_direct_cgns():
    """Import VTK classes needed by the direct CGNS -> VTK converter."""
    try:  # pragma: no cover
        from paraview.vtk import (
            vtkCellArray,
            vtkMultiBlockDataSet,
            vtkPoints,
            vtkStructuredGrid,
            vtkUnstructuredGrid,
        )
        from paraview.vtk.util import numpy_support
    except Exception:
        from vtkmodules.util import numpy_support
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import (
            vtkCellArray,
            vtkMultiBlockDataSet,
            vtkStructuredGrid,
            vtkUnstructuredGrid,
        )
    return (
        vtkStructuredGrid,
        vtkUnstructuredGrid,
        vtkPoints,
        vtkCellArray,
        vtkMultiBlockDataSet,
        numpy_support,
    )


def _cgns_zone_points_to_vtk_points(
    zoneNode: list, physicalDim: int, numpy_support, vtkPoints
):
    """Read GridCoordinates_t from one CGNS zone and return vtkPoints plus coordinate shape."""
    gridCoordinatesNodes = _cgns_children_by_label(zoneNode, "GridCoordinates_t")
    if not gridCoordinatesNodes:
        raise ValueError(f"CGNS zone '{zoneNode[0]}' has no GridCoordinates_t child")

    gridCoordinates = gridCoordinatesNodes[0]
    xNode = _cgns_child_by_name(gridCoordinates, "CoordinateX")
    yNode = _cgns_child_by_name(gridCoordinates, "CoordinateY")
    zNode = _cgns_child_by_name(gridCoordinates, "CoordinateZ")
    if xNode is None or xNode[1] is None:
        raise ValueError(f"CGNS zone '{zoneNode[0]}' has no CoordinateX array")

    x = np.asarray(xNode[1])
    y = np.zeros_like(x) if yNode is None or yNode[1] is None else np.asarray(yNode[1])
    z = np.zeros_like(x) if zNode is None or zNode[1] is None else np.asarray(zNode[1])

    pointsArray = np.empty((x.size, 3), dtype=np.float64)
    pointsArray[:, 0] = x.ravel(order="C")
    pointsArray[:, 1] = y.ravel(order="C")
    if physicalDim > 2 or zNode is not None:
        pointsArray[:, 2] = z.ravel(order="C")
    else:
        pointsArray[:, 2] = 0.0

    points = vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(pointsArray, deep=True))
    return points, x.shape


def _cgns_add_numpy_array_to_vtk_attributes(
    attributes, name: str, data: np.ndarray, numberOfTuples: int, numpy_support
) -> bool:
    """Add one numeric CGNS DataArray_t value to VTK attributes if its size is compatible."""
    array = np.asarray(data)
    if array.dtype.kind in ["S", "U", "O"] or numberOfTuples <= 0:
        return False

    flat = np.asarray(array.ravel(order="C"))
    if flat.size % numberOfTuples != 0:
        return False

    numberOfComponents = flat.size // numberOfTuples
    if numberOfComponents == 1:
        vtkArray = numpy_support.numpy_to_vtk(flat, deep=True)
    else:
        vtkArray = numpy_support.numpy_to_vtk(
            flat.reshape((numberOfTuples, numberOfComponents)), deep=True
        )
        vtkArray.SetNumberOfComponents(numberOfComponents)
    vtkArray.SetName(name)
    attributes.AddArray(vtkArray)
    return True


def _cgns_add_flow_solutions_to_vtk(zoneNode: list, vtkObject, numpy_support) -> None:
    """Transfer immediate FlowSolution_t/DataArray_t nodes from a CGNS zone to VTK data arrays."""
    numberOfPoints = vtkObject.GetNumberOfPoints()
    numberOfCells = vtkObject.GetNumberOfCells()
    for flow in _cgns_children_by_label(zoneNode, "FlowSolution_t"):
        gridLocationNode = None
        for child in flow[2]:
            if child[3] == "GridLocation_t":
                gridLocationNode = child
                break
        gridLocation = _cgns_value_as_string(gridLocationNode) or "Vertex"

        if gridLocation == "Vertex":
            attributes = vtkObject.GetPointData()
            numberOfTuples = numberOfPoints
        elif gridLocation in ["CellCenter", "FaceCenter", "EdgeCenter"]:
            attributes = vtkObject.GetCellData()
            numberOfTuples = numberOfCells
        else:
            continue

        for dataNode in _cgns_children_by_label(flow, "DataArray_t"):
            if dataNode[1] is None:
                continue
            _cgns_add_numpy_array_to_vtk_attributes(
                attributes, dataNode[0], dataNode[1], numberOfTuples, numpy_support
            )


def _cgns_index_values(node: list, dimensions: tuple[int, ...]) -> np.ndarray:
    """Read one-based CGNS index arrays and ranges from a node."""
    values = []
    for child in node[2]:
        if child[1] is None:
            continue
        if child[3] == "IndexArray_t":
            values.extend(np.asarray(child[1], dtype=np.int64).ravel(order="F"))
        elif child[3] == "IndexRange_t":
            indexRange = np.asarray(child[1], dtype=np.int64)
            if indexRange.ndim == 1 and indexRange.size == 2:
                start, end = indexRange
                step = 1 if end >= start else -1
                values.extend(np.arange(start, end + step, step, dtype=np.int64))
            elif indexRange.ndim == 2 and indexRange.shape[1] == 2:
                axes = []
                for start, end in indexRange:
                    step = 1 if end >= start else -1
                    axes.append(np.arange(start, end + step, step, dtype=np.int64))
                grids = np.meshgrid(*axes, indexing="ij")
                zeroBased = tuple(grid.ravel(order="C") - 1 for grid in grids)
                activeDimensions = tuple(dimensions[: len(zeroBased)])
                linear = np.ravel_multi_index(zeroBased, activeDimensions, order="C")
                values.extend(linear + 1)
    return np.asarray(values, dtype=np.int64)


def _cgns_tag_location(node: list) -> str:
    """Return the CGNS grid location of a tag node."""
    for child in node[2]:
        if child[3] == "GridLocation_t":
            return _cgns_value_as_string(child) or "Vertex"
    return "Vertex"


def _cgns_element_range(elementsNode: list, numberOfCells: int) -> np.ndarray:
    """Return CGNS element numbers for cells contained in an Elements_t node."""
    for child in elementsNode[2]:
        if child[3] == "IndexRange_t" and child[1] is not None:
            values = np.asarray(child[1], dtype=np.int64).ravel(order="C")
            if values.size >= 2:
                return np.arange(values[0], values[1] + 1, dtype=np.int64)
    return np.arange(1, numberOfCells + 1, dtype=np.int64)


def _cgns_add_tag_array(attributes, name: str, mask: np.ndarray, numpy_support) -> None:
    """Add or merge a binary signed-character tag array in VTK attributes."""
    existing = attributes.GetArray(name) if hasattr(attributes, "GetArray") else None
    if existing is not None:
        existingMask = np.asarray(numpy_support.vtk_to_numpy(existing)).ravel() != 0
        mask = existingMask | mask
        attributes.RemoveArray(name)
    vtkArray = numpy_support.numpy_to_vtk(np.asarray(mask, dtype=np.int8), deep=True)
    vtkArray.SetName(name)
    attributes.AddArray(vtkArray)


def _cgns_tag_name(node: list) -> str:
    """Return a CGNS tag name, preferring a FamilyName_t child."""
    for child in node[2]:
        if child[3] == "FamilyName_t":
            name = _cgns_value_as_string(child)
            if name:
                return name
    if node[3] == "ZoneSubRegion_t" and node[0].endswith("_ZSR"):
        return node[0][:-4]
    return node[0]


def _cgns_add_tags_to_vtk(
    zoneNode: list,
    vtkObject,
    numpy_support,
    cgnsElementToVtkCell: Optional[dict[int, int]] = None,
    vtkCellDimensions: Optional[np.ndarray] = None,
) -> None:
    """Transfer CGNS boundary, subregion, and family tags to VTK arrays."""
    numberOfPoints = vtkObject.GetNumberOfPoints()
    numberOfCells = vtkObject.GetNumberOfCells()
    coordinateShape = ()
    coordinateNodes = _cgns_children_by_label(zoneNode, "GridCoordinates_t")
    if coordinateNodes:
        xNode = _cgns_child_by_name(coordinateNodes[0], "CoordinateX")
        if xNode is not None and xNode[1] is not None:
            coordinateShape = tuple(np.asarray(xNode[1]).shape)

    if cgnsElementToVtkCell is None:
        cgnsElementToVtkCell = {index + 1: index for index in range(numberOfCells)}
    if vtkCellDimensions is None:
        vtkCellDimensions = np.zeros(numberOfCells, dtype=np.int8)

    def addTag(node: list) -> None:
        location = _cgns_tag_location(node)
        name = _cgns_tag_name(node)
        dimensions = coordinateShape if location == "Vertex" else (numberOfCells,)
        cgnsIds = _cgns_index_values(node, dimensions)
        if location == "Vertex":
            mask = np.zeros(numberOfPoints, dtype=bool)
            vtkIds = cgnsIds - 1
            if np.any((vtkIds < 0) | (vtkIds >= numberOfPoints)):
                raise ValueError(f"CGNS point tag '{name}' contains invalid indices")
            mask[vtkIds] = True
            _cgns_add_tag_array(vtkObject.GetPointData(), name, mask, numpy_support)
        elif location in ["CellCenter", "FaceCenter", "EdgeCenter"]:
            mask = np.zeros(numberOfCells, dtype=bool)
            try:
                vtkIds = np.asarray(
                    [cgnsElementToVtkCell[int(value)] for value in cgnsIds],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    f"CGNS cell tag '{name}' contains invalid element number {error.args[0]}"
                ) from error
            mask[vtkIds] = True
            _cgns_add_tag_array(vtkObject.GetCellData(), name, mask, numpy_support)

    for zoneBC in _cgns_children_by_label(zoneNode, "ZoneBC_t"):
        for boundaryCondition in _cgns_children_by_label(zoneBC, "BC_t"):
            addTag(boundaryCondition)
    for subregion in _cgns_children_by_label(zoneNode, "ZoneSubRegion_t"):
        addTag(subregion)

    topologicalDim = int(vtkCellDimensions.max()) if vtkCellDimensions.size else 0
    topologicalMask = vtkCellDimensions == topologicalDim
    for child in zoneNode[2]:
        if child[3] not in ["FamilyName_t", "AdditionalFamilyName_t"]:
            continue
        name = _cgns_value_as_string(child)
        if name:
            _cgns_add_tag_array(
                vtkObject.GetCellData(), name, topologicalMask, numpy_support
            )


def _cgns_element_connectivity_node(elementsNode: list) -> Optional[list]:
    """Return the ElementConnectivity child from a CGNS Elements_t node."""
    child = _cgns_child_by_name(elementsNode, "ElementConnectivity")
    if child is not None:
        return child
    for child in elementsNode[2]:
        if child[3] == "DataArray_t" and child[0].endswith("ElementConnectivity"):
            return child
    return None


def _cgns_insert_cells_from_elements_node(
    elementsNode: list,
    cellTypes: list,
    offsets: list,
    connectivity: list,
    cgnsElementToVtkCell: Optional[dict[int, int]] = None,
    vtkCellDimensions: Optional[list[int]] = None,
) -> None:
    """Append VTK cell data and optional CGNS element-number mappings."""
    cgnsElementType = int(np.asarray(elementsNode[1]).ravel()[0])
    connectivityNode = _cgns_element_connectivity_node(elementsNode)
    if connectivityNode is None or connectivityNode[1] is None:
        return
    cgnsConnectivity = np.asarray(connectivityNode[1], dtype=np.int64).ravel(order="C")

    def registerCell(localCgnsType: int) -> None:
        vtkCellIndex = len(cellTypes) - 1
        if vtkCellDimensions is not None:
            vtkCellDimensions.append(CGNSNumberDimension[localCgnsType])
        if cgnsElementToVtkCell is not None:
            localIndex = vtkCellIndex - initialCellCount
            if localIndex < elementNumbers.size:
                cgnsElementToVtkCell[int(elementNumbers[localIndex])] = vtkCellIndex

    initialCellCount = len(cellTypes)
    estimatedCells = 0
    if cgnsElementType == 20:
        cursor = 0
        while cursor < cgnsConnectivity.size:
            localType = int(cgnsConnectivity[cursor])
            if localType not in CGNSNumberOfNodes:
                break
            estimatedCells += 1
            cursor += 1 + CGNSNumberOfNodes[localType]
    elif cgnsElementType in CGNSNumberOfNodes:
        estimatedCells = cgnsConnectivity.size // CGNSNumberOfNodes[cgnsElementType]
    elementNumbers = _cgns_element_range(elementsNode, estimatedCells)

    if cgnsElementType == 20:
        cursor = 0
        while cursor < cgnsConnectivity.size:
            localCgnsType = int(cgnsConnectivity[cursor])
            cursor += 1
            if (
                localCgnsType not in CGNSNumberToVtkNumber
                or localCgnsType not in CGNSNumberOfNodes
            ):
                raise NotImplementedError(
                    f"CGNS element type {localCgnsType} is not supported by direct VTK conversion"
                )
            numberOfNodes = CGNSNumberOfNodes[localCgnsType]
            localConnectivity = cgnsConnectivity[cursor : cursor + numberOfNodes] - 1
            cursor += numberOfNodes
            permutation = CGNSNumberToVtkPermutation.get(localCgnsType, None)
            if permutation is not None:
                localConnectivity = localConnectivity[permutation]
            cellTypes.append(CGNSNumberToVtkNumber[localCgnsType])
            offsets.append(offsets[-1] + numberOfNodes)
            connectivity.extend(localConnectivity.tolist())
            registerCell(localCgnsType)
        return

    if (
        cgnsElementType not in CGNSNumberToVtkNumber
        or cgnsElementType not in CGNSNumberOfNodes
    ):
        raise NotImplementedError(
            f"CGNS element type {cgnsElementType} is not supported by direct VTK conversion"
        )

    numberOfNodes = CGNSNumberOfNodes[cgnsElementType]
    localConnectivity = cgnsConnectivity.reshape((-1, numberOfNodes)) - 1
    permutation = CGNSNumberToVtkPermutation.get(cgnsElementType, None)
    if permutation is not None:
        localConnectivity = localConnectivity[:, permutation]

    vtkCellType = CGNSNumberToVtkNumber[cgnsElementType]
    for cellConnectivity in localConnectivity:
        cellTypes.append(vtkCellType)
        offsets.append(offsets[-1] + numberOfNodes)
        connectivity.extend(cellConnectivity.tolist())
        registerCell(cgnsElementType)


def _cgns_structured_zone_to_vtk(zoneNode: list, physicalDim: int):
    """Convert one CGNS structured Zone_t node directly to vtkStructuredGrid."""
    vtkStructuredGrid, _, vtkPoints, _, _, numpy_support = _import_vtk_for_direct_cgns()
    output = vtkStructuredGrid()
    points, _ = _cgns_zone_points_to_vtk_points(
        zoneNode, physicalDim, numpy_support, vtkPoints
    )
    output.SetPoints(points)

    zsize = np.asarray(zoneNode[1])
    dimensions = [1, 1, 1]
    for i, value in enumerate(np.asarray(zsize[:, 0], dtype=int).ravel()[:3]):
        dimensions[i] = int(value)
    output.SetDimensions(dimensions)
    _cgns_add_flow_solutions_to_vtk(zoneNode, output, numpy_support)
    topologicalDim = max(sum(value > 1 for value in dimensions), 1)
    vtkCellDimensions = np.full(
        output.GetNumberOfCells(), topologicalDim, dtype=np.int8
    )
    _cgns_add_tags_to_vtk(
        zoneNode,
        output,
        numpy_support,
        vtkCellDimensions=vtkCellDimensions,
    )
    return output


def _cgns_unstructured_zone_to_vtk(zoneNode: list, physicalDim: int):
    """Convert one CGNS unstructured Zone_t node directly to vtkUnstructuredGrid."""
    _, vtkUnstructuredGrid, vtkPoints, vtkCellArray, _, numpy_support = (
        _import_vtk_for_direct_cgns()
    )
    output = vtkUnstructuredGrid()
    points, _ = _cgns_zone_points_to_vtk_points(
        zoneNode, physicalDim, numpy_support, vtkPoints
    )
    output.SetPoints(points)

    cellTypes = []
    offsets = [0]
    connectivity = []
    cgnsElementToVtkCell = {}
    vtkCellDimensions = []
    for elementsNode in _cgns_children_by_label(zoneNode, "Elements_t"):
        _cgns_insert_cells_from_elements_node(
            elementsNode,
            cellTypes,
            offsets,
            connectivity,
            cgnsElementToVtkCell,
            vtkCellDimensions,
        )

    if cellTypes:
        vtkOffsets = numpy_support.numpy_to_vtkIdTypeArray(
            np.asarray(offsets, dtype=np.int64), deep=True
        )
        vtkConnectivity = numpy_support.numpy_to_vtkIdTypeArray(
            np.asarray(connectivity, dtype=np.int64), deep=True
        )
        cellArray = vtkCellArray()
        cellArray.SetData(vtkOffsets, vtkConnectivity)
        output.SetCells(cellTypes, cellArray)

    _cgns_add_flow_solutions_to_vtk(zoneNode, output, numpy_support)
    _cgns_add_tags_to_vtk(
        zoneNode,
        output,
        numpy_support,
        cgnsElementToVtkCell,
        np.asarray(vtkCellDimensions, dtype=np.int8),
    )
    return output


def CGNSBaseExtractGlobals(baseNode: list) -> dict:
    """Extract global fields from a CGNSBase_t node as a dictionary of name -> numpy array.

    Arguments:
        baseNode (list): CGNS ``CGNSBase_t`` node.

    Returns:
        dict: A dictionary mapping field names to numpy arrays.

    """
    globals = {}
    for xNode in baseNode[2]:
        if xNode[1] is not None:
            globals[xNode[0]] = np.asarray(xNode[1])
    return globals


def CGNSTreeToVtk(treeNode: list) -> Any:
    """Convert a full CGNS tree to VTK objects, one per base, and return either the single base or a multi-block of bases.

    Arguments:
        treeNode (list): CGNS tree as read by cgns_tree_from_json_payload.

    Returns:
        vtkStructuredGrid, vtkUnstructuredGrid, or vtkMultiBlockDataSet: the VTK
        representation of the tree. A single-zone base returns the zone object;
        a multi-zone base returns one block per zone; a multi-base tree returns
        one block per base.
    """
    _, _, _, _, vtkMultiBlockDataSet, numpy_support = _import_vtk_for_direct_cgns()

    bases = _cgns_children_by_label(treeNode, "CGNSBase_t")
    globals: dict[str, Any] = {}

    for baseNode in bases:
        if baseNode[0] == "Global":
            globals.update(CGNSBaseExtractGlobals(baseNode))

    baseVtkObjects = []
    basenames = []
    for baseNode in bases:
        if baseNode[0] == "Global":
            continue
        baseVtkObjects.append(CGNSBaseToVtk(baseNode))
        basenames.append(baseNode[0])

    new_output = baseVtkObjects[0]
    # Add field Data
    field_data = new_output.GetFieldData()

    for key, value in globals.items():
        if value.dtype == "|S1":
            from vtkmodules.vtkCommonCore import vtkStringArray

            labels = vtkStringArray()
            labels.SetName(key)
            labels.SetNumberOfValues(len(value))
            for i, v in enumerate(value):
                labels.SetValue(i, v)
            field_data.AddArray(labels)
            continue

        array = numpy_support.numpy_to_vtk(value)
        array.SetName(key)
        field_data.AddArray(array)

    if len(baseVtkObjects) == 1:
        return baseVtkObjects[0]

    multiBlock = vtkMultiBlockDataSet()
    multiBlock.SetNumberOfBlocks(len(baseVtkObjects))
    for i, (name, zoneVtkObject) in enumerate(zip(basenames, baseVtkObjects)):
        multiBlock.SetBlock(i, zoneVtkObject)
        multiBlock.GetMetaData(i).Set(multiBlock.NAME(), name)
    return multiBlock


def CGNSBaseToVtk(baseNode: list) -> Any:
    """Convert a CGNSBase_t node directly to a VTK object.

    This function intentionally bypasses Muscat mesh/conversion functions.  It
    reads the CGNS/Python tree node lists directly and creates native VTK data
    objects using only VTK and NumPy.

    Args:
        baseNode (list): CGNS ``CGNSBase_t`` node.

    Returns:
        vtkStructuredGrid, vtkUnstructuredGrid, or vtkMultiBlockDataSet: the VTK
        representation of the base. A single-zone base returns the zone object;
        a multi-zone base returns one block per zone.
    """
    if (
        not isinstance(baseNode, list)
        or len(baseNode) < 4
        or baseNode[3] != "CGNSBase_t"
    ):
        raise ValueError("CGNSBaseToVtk expects a CGNSBase_t node")
    if baseNode[1] is None:
        raise ValueError(f"CGNS base '{baseNode[0]}' has no base dimensionality value")

    baseDims = np.asarray(baseNode[1], dtype=int).ravel()
    physicalDim = int(baseDims[1]) if baseDims.size > 1 else 3
    zones = _cgns_children_by_label(baseNode, "Zone_t")
    if not zones:
        raise ValueError(f"CGNS base '{baseNode[0]}' has no Zone_t children")

    zoneVtkObjects = []
    for zoneNode in zones:
        zoneType = (
            _cgns_value_as_string(_cgns_child_by_name(zoneNode, "ZoneType"))
            or "Unstructured"
        )
        if zoneType == "Structured":
            zoneVtkObjects.append(_cgns_structured_zone_to_vtk(zoneNode, physicalDim))
        elif zoneType == "Unstructured":
            zoneVtkObjects.append(_cgns_unstructured_zone_to_vtk(zoneNode, physicalDim))
        else:
            raise NotImplementedError(
                f"CGNS ZoneType '{zoneType}' is not supported by direct VTK conversion"
            )

    if len(zoneVtkObjects) == 1:
        return zoneVtkObjects[0]

    _, _, _, _, vtkMultiBlockDataSet, _ = _import_vtk_for_direct_cgns()
    multiBlock = vtkMultiBlockDataSet()
    multiBlock.SetNumberOfBlocks(len(zoneVtkObjects))
    for i, (zoneNode, zoneVtkObject) in enumerate(zip(zones, zoneVtkObjects)):
        multiBlock.SetBlock(i, zoneVtkObject)
        multiBlock.GetMetaData(i).Set(multiBlock.NAME(), zoneNode[0])
    return multiBlock
