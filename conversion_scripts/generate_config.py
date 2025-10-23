"""Create config for PLAID dataset convert."""

from pathlib import Path

import yaml

# dataset = "Tensile2d"
dataset = "VKI-LS59"

# ------------------------------------------------------------------------------------------

match dataset:
    case "Tensile2d":
        repo_id = "PLAID-datasets/Tensile2d"
        split_names = ["train_500", "test", "OOD"]
        split_names_out = ["train", "test", "OOD"]
        pb_def_names = [
            "regression_8",
            "regression_16",
            "regression_32",
            "regression_64",
            "regression_125",
            "regression_250",
            "regression_500",
            "PLAID_benchmark",
        ]
        train_split_names = [
            "train_8",
            "train_16",
            "train_32",
            "train_64",
            "train_125",
            "train_250",
            "train_500",
            "train_500",
        ]
        test_split_names = ["test"]
        repo_id_out = "fabiencasenave/Tensile2d"

        constant_features = sorted(
            [
                "Base_2_2",
                "Base_2_2/Tensile2d",
                "Base_2_2/Zone/CellData",
                "Base_2_2/Zone/CellData/GridLocation",
                "Base_2_2/Zone/Elements_TRI_3",
                "Base_2_2/Zone/FamilyName",
                "Base_2_2/Zone/GridCoordinates",
                "Base_2_2/Zone/PointData",
                "Base_2_2/Zone/PointData/GridLocation",
                "Base_2_2/Zone/SurfaceData",
                "Base_2_2/Zone/SurfaceData/GridLocation",
                "Base_2_2/Zone/ZoneBC",
                "Base_2_2/Zone/ZoneBC/Bottom",
                "Base_2_2/Zone/ZoneBC/Bottom/GridLocation",
                "Base_2_2/Zone/ZoneBC/BottomLeft",
                "Base_2_2/Zone/ZoneBC/BottomLeft/GridLocation",
                "Base_2_2/Zone/ZoneBC/Top",
                "Base_2_2/Zone/ZoneBC/Top/GridLocation",
                "Base_2_2/Zone/ZoneType",
                "Global",
            ]
        )

        input_features = sorted(
            [
                "Base_2_2/Zone",
                "Base_2_2/Zone/Elements_TRI_3/ElementConnectivity",
                "Base_2_2/Zone/Elements_TRI_3/ElementRange",
                "Base_2_2/Zone/GridCoordinates/CoordinateX",
                "Base_2_2/Zone/GridCoordinates/CoordinateY",
                "Base_2_2/Zone/ZoneBC/Bottom/PointList",
                "Base_2_2/Zone/ZoneBC/BottomLeft/PointList",
                "Base_2_2/Zone/ZoneBC/Top/PointList",
                "Global/P",
                "Global/p1",
                "Global/p2",
                "Global/p3",
                "Global/p4",
                "Global/p5",
            ]
        )

        output_features = sorted(
            [
                "Base_2_2/Zone/PointData/U1",
                "Base_2_2/Zone/PointData/U2",
                "Base_2_2/Zone/PointData/q",
                "Base_2_2/Zone/PointData/sig11",
                "Base_2_2/Zone/PointData/sig12",
                "Base_2_2/Zone/PointData/sig22",
                "Global/max_U2_top",
                "Global/max_q",
                "Global/max_sig22_top",
                "Global/max_von_mises",
            ]
        )

        constant_features_benchmark = sorted(
            [
                "Base_2_2",
                "Base_2_2/Tensile2d",
                "Base_2_2/Zone/CellData",
                "Base_2_2/Zone/CellData/GridLocation",
                "Base_2_2/Zone/Elements_TRI_3",
                "Base_2_2/Zone/FamilyName",
                "Base_2_2/Zone/GridCoordinates",
                "Base_2_2/Zone/PointData",
                "Base_2_2/Zone/PointData/GridLocation",
                "Base_2_2/Zone/SurfaceData",
                "Base_2_2/Zone/SurfaceData/GridLocation",
                "Base_2_2/Zone/ZoneBC",
                "Base_2_2/Zone/ZoneBC/Bottom",
                "Base_2_2/Zone/ZoneBC/Bottom/GridLocation",
                "Base_2_2/Zone/ZoneBC/BottomLeft",
                "Base_2_2/Zone/ZoneBC/BottomLeft/GridLocation",
                "Base_2_2/Zone/ZoneBC/Top",
                "Base_2_2/Zone/ZoneBC/Top/GridLocation",
                "Base_2_2/Zone/ZoneType",
                "Global",
            ]
        )

        input_features_benchmark = sorted(
            [
                "Base_2_2/Zone",
                "Base_2_2/Zone/Elements_TRI_3/ElementConnectivity",
                "Base_2_2/Zone/Elements_TRI_3/ElementRange",
                "Base_2_2/Zone/GridCoordinates/CoordinateX",
                "Base_2_2/Zone/GridCoordinates/CoordinateY",
                "Base_2_2/Zone/ZoneBC/Bottom/PointList",
                "Base_2_2/Zone/ZoneBC/BottomLeft/PointList",
                "Base_2_2/Zone/ZoneBC/Top/PointList",
                "Global/P",
                "Global/p1",
                "Global/p2",
                "Global/p3",
                "Global/p4",
                "Global/p5",
            ]
        )

        output_features_benchmark = sorted(
            [
                "Base_2_2/Zone/PointData/U1",
                "Base_2_2/Zone/PointData/U2",
                "Base_2_2/Zone/PointData/sig11",
                "Base_2_2/Zone/PointData/sig12",
                "Base_2_2/Zone/PointData/sig22",
                "Global/max_U2_top",
                "Global/max_sig22_top",
                "Global/max_von_mises",
            ]
        )

        data_config = {}
        data_config["split_names"] = split_names
        data_config["split_names_out"] = split_names_out
        data_config["pb_def_names"] = pb_def_names
        data_config["train_split_names"] = train_split_names
        data_config["test_split_names"] = test_split_names
        data_config["repo_id"] = repo_id
        data_config["repo_id_out"] = repo_id_out
        data_config["constant_features"] = constant_features
        data_config["input_features"] = input_features
        data_config["output_features"] = output_features
        data_config["constant_features_benchmark"] = constant_features_benchmark
        data_config["input_features_benchmark"] = input_features_benchmark
        data_config["output_features_benchmark"] = output_features_benchmark

    case "VKI-LS59":
        repo_id = "PLAID-datasets/VKI-LS59"
        split_names = ["train", "test"]
        split_names_out = ["train", "test"]
        pb_def_names = [
            "regression_8",
            "regression_16",
            "regression_32",
            "regression_64",
            "regression_125",
            "regression_250",
            "regression_500",
            "regression",
            "PLAID_benchmark",
        ]
        train_split_names = [
            "train_8",
            "train_16",
            "train_32",
            "train_64",
            "train_125",
            "train_250",
            "train_500",
            "train",
            "train",
        ]
        test_split_names = ["test"]
        repo_id_out = "fabiencasenave/VKI-LS59"

        constant_features = sorted(
            [
                "Base_1_2",
                "Base_1_2/Blade",
                "Base_1_2/Zone",
                "Base_1_2/Zone/CellData",
                "Base_1_2/Zone/CellData/GridLocation",
                "Base_1_2/Zone/Elements_BAR_2",
                "Base_1_2/Zone/Elements_BAR_2/ElementConnectivity",
                "Base_1_2/Zone/Elements_BAR_2/ElementRange",
                "Base_1_2/Zone/FamilyName",
                "Base_1_2/Zone/GridCoordinates",
                "Base_1_2/Zone/PointData",
                "Base_1_2/Zone/PointData/GridLocation",
                "Base_1_2/Zone/SurfaceData",
                "Base_1_2/Zone/SurfaceData/GridLocation",
                "Base_1_2/Zone/ZoneBC",
                "Base_1_2/Zone/ZoneBC/1D",
                "Base_1_2/Zone/ZoneBC/1D/GridLocation",
                "Base_1_2/Zone/ZoneBC/1D/PointList",
                "Base_1_2/Zone/ZoneType",
                "Base_2_2",
                "Base_2_2/Blade",
                "Base_2_2/Zone",
                "Base_2_2/Zone/CellData",
                "Base_2_2/Zone/CellData/GridLocation",
                "Base_2_2/Zone/Elements_QUAD_4",
                "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
                "Base_2_2/Zone/Elements_QUAD_4/ElementRange",
                "Base_2_2/Zone/FamilyName",
                "Base_2_2/Zone/GridCoordinates",
                "Base_2_2/Zone/PointData",
                "Base_2_2/Zone/PointData/GridLocation",
                "Base_2_2/Zone/SurfaceData",
                "Base_2_2/Zone/SurfaceData/GridLocation",
                "Base_2_2/Zone/ZoneBC",
                "Base_2_2/Zone/ZoneBC/Extrado",
                "Base_2_2/Zone/ZoneBC/Extrado/GridLocation",
                "Base_2_2/Zone/ZoneBC/Extrado/PointList",
                "Base_2_2/Zone/ZoneBC/Inflow",
                "Base_2_2/Zone/ZoneBC/Inflow/GridLocation",
                "Base_2_2/Zone/ZoneBC/Inflow/PointList",
                "Base_2_2/Zone/ZoneBC/Intrado",
                "Base_2_2/Zone/ZoneBC/Intrado/GridLocation",
                "Base_2_2/Zone/ZoneBC/Intrado/PointList",
                "Base_2_2/Zone/ZoneBC/Outflow",
                "Base_2_2/Zone/ZoneBC/Outflow/GridLocation",
                "Base_2_2/Zone/ZoneBC/Outflow/PointList",
                "Base_2_2/Zone/ZoneBC/Periodic_1",
                "Base_2_2/Zone/ZoneBC/Periodic_1/GridLocation",
                "Base_2_2/Zone/ZoneBC/Periodic_1/PointList",
                "Base_2_2/Zone/ZoneBC/Periodic_2",
                "Base_2_2/Zone/ZoneBC/Periodic_2/GridLocation",
                "Base_2_2/Zone/ZoneBC/Periodic_2/PointList",
                "Base_2_2/Zone/ZoneType",
                "Global",
            ]
        )

        input_features = sorted(
            [
                "Base_1_2/Zone/GridCoordinates/CoordinateX",
                "Base_1_2/Zone/GridCoordinates/CoordinateY",
                "Base_2_2/Zone/GridCoordinates/CoordinateX",
                "Base_2_2/Zone/GridCoordinates/CoordinateY",
                "Base_2_2/Zone/PointData/sdf",
                "Global/angle_in",
                "Global/mach_out",
            ]
        )

        output_features = sorted(
            [
                "Base_1_2/Zone/PointData/M_iso",
                "Base_2_2/Zone/PointData/mach",
                "Base_2_2/Zone/PointData/nut",
                "Base_2_2/Zone/PointData/ro",
                "Base_2_2/Zone/PointData/roe",
                "Base_2_2/Zone/PointData/rou",
                "Base_2_2/Zone/PointData/rov",
                "Global/Pr",
                "Global/Q",
                "Global/Tr",
                "Global/angle_out",
                "Global/eth_is",
                "Global/power",
            ]
        )

        constant_features_benchmark = sorted(
            [
                "Base_2_2",
                "Base_2_2/Blade",
                "Base_2_2/Zone",
                "Base_2_2/Zone/CellData",
                "Base_2_2/Zone/CellData/GridLocation",
                "Base_2_2/Zone/Elements_QUAD_4",
                "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
                "Base_2_2/Zone/Elements_QUAD_4/ElementRange",
                "Base_2_2/Zone/FamilyName",
                "Base_2_2/Zone/GridCoordinates",
                "Base_2_2/Zone/PointData",
                "Base_2_2/Zone/PointData/GridLocation",
                "Base_2_2/Zone/SurfaceData",
                "Base_2_2/Zone/SurfaceData/GridLocation",
                "Base_2_2/Zone/ZoneBC",
                "Base_2_2/Zone/ZoneBC/Extrado",
                "Base_2_2/Zone/ZoneBC/Extrado/GridLocation",
                "Base_2_2/Zone/ZoneBC/Extrado/PointList",
                "Base_2_2/Zone/ZoneBC/Inflow",
                "Base_2_2/Zone/ZoneBC/Inflow/GridLocation",
                "Base_2_2/Zone/ZoneBC/Inflow/PointList",
                "Base_2_2/Zone/ZoneBC/Intrado",
                "Base_2_2/Zone/ZoneBC/Intrado/GridLocation",
                "Base_2_2/Zone/ZoneBC/Intrado/PointList",
                "Base_2_2/Zone/ZoneBC/Outflow",
                "Base_2_2/Zone/ZoneBC/Outflow/GridLocation",
                "Base_2_2/Zone/ZoneBC/Outflow/PointList",
                "Base_2_2/Zone/ZoneBC/Periodic_1",
                "Base_2_2/Zone/ZoneBC/Periodic_1/GridLocation",
                "Base_2_2/Zone/ZoneBC/Periodic_1/PointList",
                "Base_2_2/Zone/ZoneBC/Periodic_2",
                "Base_2_2/Zone/ZoneBC/Periodic_2/GridLocation",
                "Base_2_2/Zone/ZoneBC/Periodic_2/PointList",
                "Base_2_2/Zone/ZoneType",
                "Global",
            ]
        )

        input_features_benchmark = sorted(
            [
                "Base_2_2/Zone/GridCoordinates/CoordinateX",
                "Base_2_2/Zone/GridCoordinates/CoordinateY",
                "Base_2_2/Zone/PointData/sdf",
                "Global/angle_in",
                "Global/mach_out",
            ]
        )

        output_features_benchmark = sorted(
            [
                "Base_2_2/Zone/PointData/mach",
                "Base_2_2/Zone/PointData/nut",
                "Global/Pr",
                "Global/Q",
                "Global/Tr",
                "Global/angle_out",
                "Global/eth_is",
                "Global/power",
            ]
        )

        data_config = {}
        data_config["split_names"] = split_names
        data_config["split_names_out"] = split_names_out
        data_config["pb_def_names"] = pb_def_names
        data_config["train_split_names"] = train_split_names
        data_config["test_split_names"] = test_split_names
        data_config["repo_id"] = repo_id
        data_config["repo_id_out"] = repo_id_out
        data_config["constant_features"] = constant_features
        data_config["input_features"] = input_features
        data_config["output_features"] = output_features
        data_config["constant_features_benchmark"] = constant_features_benchmark
        data_config["input_features_benchmark"] = input_features_benchmark
        data_config["output_features_benchmark"] = output_features_benchmark

        path = Path("./config_Tensile2d.yaml")

# -----------------------------------------------------------

path = Path(f"./config_{dataset}.yaml")
with path.open("w") as file:
    yaml.dump(data_config, file, default_flow_style=False, sort_keys=True)
