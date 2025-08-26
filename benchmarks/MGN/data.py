import csv

import dgl
import Muscat.Containers.ElementsDescription as ED
import numpy as np
import torch
from dgl.data import DGLDataset
from Muscat.Bridges.CGNSBridge import CGNSToMesh
from Muscat.Containers import MeshModificationTools as MMT
from Muscat.Containers.Filters import FilterObjects as FO
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.FETools import PrepareFEComputation
from Muscat.FE.Fields.FEField import FEField
from rich.progress import track
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import one_hot

from plaid.containers.sample import Sample


def tri_cells_to_edges(cells):
    edges = torch.cat([cells[:, :2], cells[:, 1:], cells[:, ::2]], dim=0)
    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=0)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0)
    return unique_edges


def quad_cells_to_edges(cells):
    edges = torch.cat(
        [
            cells[:, 0:2],
            cells[:, 1:3],
            cells[:, 2:4],
            torch.stack((cells[:, 3], cells[:, 0]), dim=-1),
        ],
        dim=0,
    )
    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=0)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0)
    return unique_edges


def distance_field(mesh, nTag=None):
    MMT.ComputeSkin(mesh, md=None, inPlace=True, skinTagName="Skin")
    dim = int(mesh.GetElementsDimensionality())
    Tspace, Tnumberings, _, _ = PrepareFEComputation(mesh, numberOfComponents=1)
    field_mesh = FEField("", mesh=mesh, space=Tspace, numbering=Tnumberings[0])
    opSkin, _, _ = GetFieldTransferOp(
        inputField=field_mesh,
        targetPoints=mesh.nodes,
        method="Interp/Clamp",
        elementFilter=FO.ElementFilter(dimensionality=dim - 1, nTag=nTag),
        verbose=False,
    )
    skinpos = opSkin.dot(mesh.nodes)
    distance = np.sqrt(np.sum((skinpos - mesh.nodes) ** 2, axis=1))
    return distance


def get_data(mesh, dataset_name=None, load_fields=True):
    # Automatically retrieve node fields
    if load_fields:
        node_fields = {k: torch.tensor(v) for k, v in mesh.nodeFields.items()}
    else:
        node_fields = {}

    # Automatically retrieve nodetags
    tag_label_map = {
        "Airfoil": 2,
        "Holes": 2,
        "Top": 2,
        "Inlet": 4,
        "Bottom": 4,
        "Inflow": 4,
        "Ext_bound": 6,
        "Outflow": 6,
        "Intrado": 1,
        "Extrado": 3,
        "Periodic_1": 5,
        "Periodic_2": 7,
    }

    labels = np.zeros(mesh.GetNumberOfNodes(), dtype=int)
    for tag_name in mesh.nodesTags:
        tag_name_str = str(tag_name).split("(")[0].strip()
        tag_ids = mesh.GetNodalTag(tag_name_str).GetIds()
        label_value = tag_label_map.get(tag_name_str, 0)
        labels[tag_ids] = label_value
    node_type = torch.tensor(labels)

    # Calculate distance field
    nTag = {
        "2D_Profile": "Airfoil",
        "2D_Multiscale": "Holes",
        # "Tensile2D": "Top",
        "AirfRANS": "Airfoil",
    }.get(dataset_name, None)

    if dataset_name == "VKI_LS59":
        distance = node_fields["sdf"]
    else:
        dst = distance_field(mesh, nTag=nTag)
        distance = torch.tensor(dst)

    # Select element type and compute edges
    element_type = ED.Quadrangle_4 if dataset_name == "VKI_LS59" else ED.Triangle_3
    cells = torch.tensor(mesh.elements[element_type].connectivity)
    edge_function = (
        quad_cells_to_edges if dataset_name == "VKI_LS59" else tri_cells_to_edges
    )
    edges = edge_function(cells)

    mesh_pos = torch.tensor(mesh.nodes)

    # Structure return values into distinct parts
    node_fields_dict = node_fields
    node_features_dict = {
        "distance": distance,
        "node_type": node_type,
        "mesh_pos": mesh_pos,
    }

    return node_fields_dict, node_features_dict, edges


def read_indices_from_csv(dataset_path, split_train_name=None, split_test_name=None):
    train_indices = []
    test_indices = []
    with open(f"{dataset_path}/problem_definition/split.csv", mode="r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] == split_train_name:
                train_indices = list(map(int, row[1:]))
            elif row[0] == split_test_name:
                test_indices = list(map(int, row[1:]))
    return train_indices, test_indices


def load_datasets(
    dataset_name,
    dataset_path,
    split_train_name=None,
    split_test_name=None,
    target_field="all_fields",
):
    # Read indices from CSV
    train_indices, test_indices = read_indices_from_csv(
        dataset_path, split_train_name, split_test_name
    )

    # Scalar input and output definitions
    scalar_input_dict = {
        "VKI_LS59": ["angle_in", "mach_out"],
        "Tensile2D": ["P", "p1", "p2", "p3", "p4", "p5"],
        "2D_Multiscale": ["C11", "C12", "C22"],
        "Rotor37": ["Omega", "P"],
    }

    scalar_output_dict = {
        "VKI_LS59": ["Q", "power", "Pr", "Tr", "eth_is", "angle_out"],
        "Tensile2D": ["max_von_mises", "max_U2_top", "max_sig22_top"],
        "2D_Multiscale": ["effective_energy"],
        "Rotor37": ["Massflow", "Compression_ratio", "Efficiency"],
    }

    # Fields expected to be retrieved
    field_names_dict = {
        "VKI_LS59": ["mach", "nut"],
        "Tensile2D": ["U1", "U2", "sig11", "sig22", "sig12"],
        "2D_Multiscale": ["u1", "u2", "P11", "P12", "P22", "P21", "psi"],
        "Rotor37": ["Density", "Pressure", "Temperature"],
        "2D_Profile": ["Mach", "Pressure", "Velocity-x", "Velocity-y"],
    }

    def process_samples(dataset_name, dataset_path, indices, field_names, process_type):
        X_nodes, X_edges, X_node_tags, X_scalars, X_distances = [], [], [], [], []
        Y_fields, Y_scalars = [], []

        description = f"✅ Processing {process_type} samples"

        for i in track(
            range(len(indices)), total=len(indices), description=description
        ):
            id_sample = indices[i]
            sample_path = f"{dataset_path}/dataset/samples/sample_{id_sample:09}/"
            mesh_data = Sample.load_from_dir(sample_path)
            tree = mesh_data.get_mesh()

            if dataset_name == "VKI_LS59":
                mesh = CGNSToMesh(tree, baseNames=["Base_2_2"])
            else:
                mesh = CGNSToMesh(tree)

            # Get data from mesh
            load_fields = process_type == "train"
            node_fields, node_features, edges = get_data(
                mesh, dataset_name=dataset_name, load_fields=load_fields
            )

            nodes = node_features["mesh_pos"]
            node_tags = node_features["node_type"]
            distances = node_features["distance"]

            # Determine which fields to retrieve
            if target_field == "all_fields":
                effective_field_names = field_names
            elif target_field and target_field in field_names:
                effective_field_names = [target_field]
            else:
                effective_field_names = []

            # Handle case where no valid target field is specified
            if not effective_field_names:
                raise ValueError(
                    "No valid field selected for processing. Please specify a valid target field or use 'all_fields'."
                )

            if process_type == "train":
                fields = [node_fields[fn] for fn in effective_field_names]
                fields = torch.column_stack(fields)
                Y_fields += [fields]

            # Retrieve input scalars
            in_scalars_names = scalar_input_dict.get(dataset_name, [])
            X_scalars.append(
                [mesh_data.scalars.get(fn) for fn in in_scalars_names]
                if in_scalars_names
                else []
            )

            # Retrieve output scalars
            out_scalars_names = scalar_output_dict.get(dataset_name, [])
            Y_scalars.append(
                [mesh_data.scalars.get(fn) for fn in out_scalars_names]
                if out_scalars_names
                else []
            )

            X_nodes += [nodes]
            X_edges += [edges]
            X_node_tags += [node_tags]
            X_distances += [distances]

        X_scalars = np.array(X_scalars)
        Y_scalars = np.array(Y_scalars)

        # Processed data
        data = {
            "X_nodes": X_nodes,
            "X_edges": X_edges,
            "X_node_tags": X_node_tags,
            "X_distances": X_distances,
            "X_scalars": X_scalars,
            "Y_fields": Y_fields,
            "Y_scalars": Y_scalars,
        }

        return data

    # Get the field names specific to the dataset
    field_names = field_names_dict.get(dataset_name, [])

    # Process train and test samples
    train_data = process_samples(
        dataset_name, dataset_path, train_indices, field_names, "train"
    )
    test_data = process_samples(
        dataset_name, dataset_path, test_indices, field_names, "test"
    )

    return train_data, test_data, train_indices, test_indices


class GraphDataset(DGLDataset):
    def __init__(
        self,
        args,
        data,
        data_type,
        in_scaler=None,
        out_scaler=None,
        fields_min=None,
        fields_max=None,
    ):
        super().__init__(name="graph_dataset")

        self.data = data
        self.in_scaler = in_scaler
        self.out_scaler = out_scaler
        self.fields_min = fields_min
        self.fields_max = fields_max

        self.num_samples = len(data["X_nodes"])
        self.graphs = []

        if data_type == "train":
            self.in_scaler = StandardScaler()
            self.out_scaler = StandardScaler()

            self.input_globals = (
                torch.tensor(
                    self.in_scaler.fit_transform(data["X_scalars"]), dtype=torch.float32
                )
                if data["X_scalars"].size > 0
                else torch.tensor([])
            )
            self.output_globals = (
                torch.tensor(
                    self.out_scaler.fit_transform(data["Y_scalars"]),
                    dtype=torch.float32,
                )
                if data["Y_scalars"].size > 0
                else torch.tensor([])
            )

            self.fields_min = torch.min(
                torch.cat(data["Y_fields"], dim=0), dim=0
            ).values
            self.fields_max = torch.max(
                torch.cat(data["Y_fields"], dim=0), dim=0
            ).values

        elif data_type == "test":
            assert all(
                v is not None for v in [in_scaler, out_scaler, fields_min, fields_max]
            )

            self.input_globals = (
                torch.tensor(
                    self.in_scaler.transform(data["X_scalars"]), dtype=torch.float32
                )
                if data["X_scalars"].size > 0
                else torch.tensor([])
            )
            self.output_globals = (
                torch.tensor(
                    self.out_scaler.transform(data["Y_scalars"]), dtype=torch.float32
                )
                if data["Y_scalars"].size > 0
                else torch.tensor([])
            )

        description = f"🚀 Processing {data_type} graphs"
        for i in track(
            range(self.num_samples), total=self.num_samples, description=description
        ):
            pos = data["X_nodes"][i].to(torch.float32)
            edge_index = data["X_edges"][i].t().long()

            tags = one_hot(data["X_node_tags"][i], num_classes=9).to(torch.float32)
            dis = data["X_distances"][i].unsqueeze(1).to(torch.float32)

            num_sca = (
                self.input_globals[i].shape[-1] if self.input_globals.size(0) > 0 else 0
            )
            sca = (
                torch.ones((len(pos), num_sca), dtype=torch.float32)
                * self.input_globals[i].unsqueeze(0)
                if num_sca
                else torch.tensor([])
            )

            src, dst = edge_index[0], edge_index[1]
            graph = dgl.graph((src, dst))

            node_features = [pos, tags, dis]
            if sca.size(0) > 0:
                node_features.append(sca)

            graph.ndata["x"] = torch.cat(node_features, dim=1)

            if data_type == "train":
                Y_fields = (
                    (
                        (data["Y_fields"][i] - self.fields_min)
                        / (self.fields_max - self.fields_min)
                    )
                    .clone()
                    .detach()
                    .to(torch.float32)
                )
                graph.ndata["y"] = Y_fields

            graph.ndata["pos"] = pos

            # Calculate squared distances
            bandwidth = args.bandwidth * 10
            differences = (
                (data["X_nodes"][i][src] - data["X_nodes"][i][dst]).clone().detach()
            )
            sqdists = torch.sum(differences**2, dim=1).unsqueeze(1)
            sqdists = torch.exp(-0.5 * sqdists / bandwidth).to(torch.float32)
            graph.edata["f"] = sqdists

            self.graphs.append(graph)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        input_globals = (
            self.input_globals[idx]
            if self.input_globals.numel() > 0
            else torch.tensor([])
        )
        output_globals = (
            self.output_globals[idx]
            if self.output_globals.numel() > 0
            else torch.tensor([])
        )

        return graph, input_globals, output_globals

    def __len__(self):
        return len(self.graphs)
