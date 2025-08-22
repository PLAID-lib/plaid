import argparse
import os
import pickle
import time

import dgl
import numpy as np
import torch
import torch.distributed as dist
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from plaid.containers.dataset import Dataset as PlaidDataset
from plaid.problem_definition import ProblemDefinition

from data import *
from utils import *
from model import MeshGraphNet

class ElastoDataset(Dataset):
    def __init__(
        self, plaid_dataset, plaid_problem, split="train", fields=("U_x", "U_y")
    ):
        self.plaid_ds = plaid_dataset
        self.ids = plaid_problem.get_split(split)
        sample0 = self.plaid_ds[self.ids[0]]
        self.time_steps = sample0.get_all_mesh_times()
        self.n_steps = len(self.time_steps) - 1
        self.fields = fields

        # Precompute mesh-level features
        self.mesh_list = []
        for sid in self.ids:
            sample = self.plaid_ds[sid]
            pos = torch.tensor(sample.get_nodes(), dtype=torch.float32)
            cells = sample.get_elements()["TRI_3"]
            edge_index = (
                tri_cells_to_edges(torch.tensor(cells, dtype=torch.long))
                .t()
                .contiguous()
            )

            _, sdf = get_distances_to_borders(sample.get_nodes(), cells)
            sdf = torch.tensor(sdf, dtype=torch.float32)
            sdf_sine = sinusoidal_embedding(
                sdf, num_basis=4, max_coord=4, spacing=0.001
            )
            angles = angles_to_planes(pos)
            sph = torch.cat(
                [spherical_harmonics(angles[:, i], l_max=4)[:, 1:] for i in range(4)],
                dim=1,
            )

            src, dst = edge_index
            disp = pos[src] - pos[dst]
            sqd = torch.exp(
                -0.5 * (disp**2).sum(1, keepdim=True) / (5.757861066563731 * 10)
            )
            edge_attr = torch.cat([sqd, disp], dim=-1)

            self.mesh_list.append(
                {
                    "pos": pos,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "sdf": sdf,
                    "sdf_sine": sdf_sine,
                    "sph": sph,
                }
            )

    def __len__(self):
        return len(self.ids) * self.n_steps

    def __getitem__(self, idx):
        sim_idx = idx // self.n_steps
        t = idx % self.n_steps
        sid = self.ids[sim_idx]
        sample = self.plaid_ds[sid]
        mesh = self.mesh_list[sim_idx]

        # Velocities at t and t+1
        ux_t = torch.tensor(
            sample.get_field(self.fields[0], time=self.time_steps[t]),
            dtype=torch.float32,
        )
        uy_t = torch.tensor(
            sample.get_field(self.fields[1], time=self.time_steps[t]),
            dtype=torch.float32,
        )
        ux_tp = torch.tensor(
            sample.get_field(self.fields[0], time=self.time_steps[t + 1]),
            dtype=torch.float32,
        )
        uy_tp = torch.tensor(
            sample.get_field(self.fields[1], time=self.time_steps[t + 1]),
            dtype=torch.float32,
        )

        u_t = torch.stack([ux_t, uy_t], dim=-1)
        u_tp = torch.stack([ux_tp, uy_tp], dim=-1)

        # Input features: [u_t, pos, sdf, sph]
        x = torch.cat([u_t, mesh["pos"], mesh["sdf"], mesh["sph"]], dim=-1)
        # Target: Δu = u_{t+1}-u_t
        y = u_tp - u_t

        # Build DGL graph
        g = dgl.graph((mesh["edge_index"][0], mesh["edge_index"][1]))
        g.ndata["x"] = x
        g.ndata["y"] = y
        g.edata["f"] = mesh["edge_attr"]
        return g


def compute_minmax_scaler(train_ds, args, device):
    loader = GraphDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    xs, ys = [], []
    for batch in loader:
        xs.append(batch.ndata["x"][:, :2])
        ys.append(batch.ndata["y"])
    all_x = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    return {
        "type": "minmax",
        "min_x": all_x.min(0)[0].to(device),
        "max_x": all_x.max(0)[0].to(device),
        "min_y": all_y.min(0)[0].to(device),
        "max_y": all_y.max(0)[0].to(device),
    }


def compute_standard_scaler(train_ds, args, device):
    loader = GraphDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    xs, ys = [], []
    for batch in loader:
        xs.append(batch.ndata["x"][:, :2])
        ys.append(batch.ndata["y"])
    all_x = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    return {
        "type": "standard",
        "mean_x": all_x.mean(0).to(device),
        "std_x": all_x.std(0).to(device),
        "mean_y": all_y.mean(0).to(device),
        "std_y": all_y.std(0).to(device),
    }


def save_checkpoint(path, model, optimizer, epoch, rank):
    if rank == 0:
        state = {
            "epoch": epoch,
            "model_state": model.module.state_dict()
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model.state_dict(),
            "optim_state": optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"→ [Rank {rank}] Saved checkpoint to {path}")


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    if not os.path.isfile(path):
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"→ Loaded checkpoint '{path}', resume at epoch {start_epoch}")
    return start_epoch


def inference(model, device, test_ds, scaler, args, rank):
    if rank != 0:
        return None
    print("🔍 Starting inference on test set")
    t0 = time.perf_counter()
    model.eval()
    preds = []

    with torch.no_grad():
        for sim_idx, sid in enumerate(test_ds.ids):
            mesh = test_ds.mesh_list[sim_idx]
            sample = test_ds.plaid_ds[sid]

            # Init u_pred at t0
            ux0 = torch.tensor(sample.get_field("U_x", time=test_ds.time_steps[0])).to(
                device
            )
            uy0 = torch.tensor(sample.get_field("U_y", time=test_ds.time_steps[0])).to(
                device
            )
            u_pred = torch.stack([ux0, uy0], dim=-1)

            pred_dict = {"U_x": [], "U_y": []}

            for i, fn in enumerate(["U_x", "U_y"]):
                pred_dict[fn].append(u_pred[:, i].cpu().numpy())

            for t in range(test_ds.n_steps):
                # Prepare x0
                if scaler["type"] == "minmax":
                    x0 = (u_pred - scaler["min_x"]) / (
                        scaler["max_x"] - scaler["min_x"]
                    )
                elif scaler["type"] == "standard":
                    x0 = (u_pred - scaler["mean_x"]) / scaler["std_x"]
                else:
                    x0 = u_pred

                x = torch.cat(
                    [
                        x0,
                        mesh["pos"].to(device),
                        mesh["sdf"].to(device),
                        mesh["sph"].to(device),
                    ],
                    dim=-1,
                )

                g = dgl.graph((mesh["edge_index"][0], mesh["edge_index"][1])).to(device)
                g.ndata["x"], g.edata["f"] = x, mesh["edge_attr"].to(device)

                out = model(g.ndata["x"], g.edata["f"], g)

                if scaler["type"] == "minmax":
                    dv = out * (scaler["max_y"] - scaler["min_y"]) + scaler["min_y"]
                elif scaler["type"] == "standard":
                    dv = out * scaler["std_y"] + scaler["mean_y"]
                else:
                    dv = out

                u_pred = u_pred + dv

                # Ground truth at t+1
                ux_tp = torch.tensor(
                    sample.get_field("U_x", time=test_ds.time_steps[t + 1])
                ).to(device)
                uy_tp = torch.tensor(
                    sample.get_field("U_y", time=test_ds.time_steps[t + 1])
                ).to(device)
                torch.stack([ux_tp, uy_tp], dim=-1)

                for i, fn in enumerate(["U_x", "U_y"]):
                    pred_dict[fn].append(u_pred[:, i].cpu().numpy())

            # Stack along time
            for fn in pred_dict:
                pred_dict[fn] = np.stack(pred_dict[fn], axis=1).T
            preds.append(pred_dict)

    with open(args.submission_path, "wb") as f:
        pickle.dump(preds, f)

    t1 = time.perf_counter()
    print(f"🎯 Inference done in {t1 - t0:.2f}s — saving to {args.submission_path}")


def main_worker(args):
    t_start = time.perf_counter()

    # Init DDP
    # os.environ.setdefault("MASTER_ADDR", args.master_addr)
    # os.environ.setdefault("MASTER_PORT", str(args.master_port))
    rank = int(os.environ.get("SLURM_PROCID"))
    world_size = int(os.environ.get("SLURM_NTASKS"))
    ngpus_per_node = torch.cuda.device_count()
    local_rank = rank % ngpus_per_node

    t0 = time.perf_counter()
    dist.init_process_group("nccl", world_size=world_size, rank=rank)
    t1 = time.perf_counter()
    if rank == 0:
        print(f"🚀 [DDP init] {t1 - t0:.2f}s — world_size={world_size}")

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load data & problem
    if rank == 0:
        print("📂 Loading Plaid data")
    t0 = time.perf_counter()
    plaid_ds = PlaidDataset()
    plaid_ds._load_from_dir_(args.data_dir, verbose=(rank == 0))
    problem = ProblemDefinition()
    problem._load_from_dir_(args.problem_dir)
    t1 = time.perf_counter()
    if rank == 0:
        print(
            f"✔️ [Data load] {t1 - t0:.2f}s — train sims={len(problem.get_split('train'))}, test sims={len(problem.get_split('test'))}"
        )

    # Build datasets
    if rank == 0:
        print("🚀 Building datasets")
    t0 = time.perf_counter()
    train_ds = ElastoDataset(plaid_ds, problem, split="train")
    test_ds = ElastoDataset(plaid_ds, problem, split="test")
    t1 = time.perf_counter()
    if rank == 0:
        print(
            f"✔️ [Dataset] {t1 - t0:.2f}s — train size={len(train_ds)}, test size={len(test_ds)}"
        )

    # Infer dims
    sample0 = train_ds[0]
    dim_x = sample0.ndata["x"].shape[1]
    dim_f = sample0.edata["f"].shape[1]
    dim_y = sample0.ndata["y"].shape[1]
    if rank == 0:
        print(f"🎯 Dimensions → node_in={dim_x}, edge_in={dim_f}, node_out={dim_y}")

    # Compute scaler
    scaler = {"type": "none"}
    if args.scaler == "minmax":
        if rank == 0:
            print("📏 Computing Min-Max scaler")
        t0 = time.perf_counter()
        tmp = compute_minmax_scaler(train_ds, args, device)
        dist.broadcast_object_list([tmp], src=0)
        scaler = tmp
        t1 = time.perf_counter()
        if rank == 0:
            print(f"[Scaler minmax] {t1 - t0:.2f}s → {scaler}")
    elif args.scaler == "standard":
        if rank == 0:
            print("📏 Computing Standard scaler")
        t0 = time.perf_counter()
        tmp = compute_standard_scaler(train_ds, args, device)
        dist.broadcast_object_list([tmp], src=0)
        scaler = tmp
        t1 = time.perf_counter()
        if rank == 0:
            print(f"✔️ [Scaler std] {t1 - t0:.2f}s → {scaler}")

    # DataLoader + sampler
    t0 = time.perf_counter()
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = GraphDataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    t1 = time.perf_counter()
    if rank == 0:
        print(f"[Loader build] {t1 - t0:.2f}s — batches={len(train_loader)}")

    # Model, optimizer, scheduler
    if rank == 0:
        print("🚀 Building model")
    t0 = time.perf_counter()
    model = MeshGraphNet(
        input_dim_nodes=dim_x,
        input_dim_edges=dim_f,
        output_dim=dim_y,
        processor_size=args.processor_size,
        num_layers_node_processor=args.num_layers_node_processor,
        num_layers_edge_processor=args.num_layers_edge_processor,
        hidden_dim_node_encoder=args.hidden_dim_node_encoder,
        num_layers_node_encoder=args.num_layers_node_encoder,
        hidden_dim_edge_encoder=args.hidden_dim_edge_encoder,
        num_layers_edge_encoder=args.num_layers_edge_encoder,
        hidden_dim_node_decoder=args.hidden_dim_node_decoder,
        num_layers_node_decoder=args.num_layers_node_decoder,
        aggregation=args.aggregation,
        activation=args.activation,
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: args.lr_decay_rate**e
    )
    use_amp = args.amp and device.type.startswith("cuda")
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None
    t1 = time.perf_counter()
    if rank == 0:
        print(f"✔️ [Model build] {t1 - t0:.2f}s")

    start_epoch = 0
    best_loss = float("inf")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        ep0 = time.perf_counter()
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for batch in train_loader:
            g = batch.to(device)

            if scaler["type"] == "minmax":
                g.ndata["x"][:, :2] = (g.ndata["x"][:, :2] - scaler["min_x"]) / (
                    scaler["max_x"] - scaler["min_x"]
                )
                g.ndata["y"] = (g.ndata["y"] - scaler["min_y"]) / (
                    scaler["max_y"] - scaler["min_y"]
                )
            elif scaler["type"] == "standard":
                g.ndata["x"][:, :2] = (g.ndata["x"][:, :2] - scaler["mean_x"]) / scaler[
                    "std_x"
                ]
                g.ndata["y"] = (g.ndata["y"] - scaler["mean_y"]) / scaler["std_y"]

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(g.ndata["x"], g.edata["f"], g)
                    loss = criterion(pred, g.ndata["y"])
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                pred = model(g.ndata["x"], g.edata["f"], g)
                loss = criterion(pred, g.ndata["y"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # Reduce & log
        loss_tensor = torch.tensor([running_loss, len(train_loader)], device=device)
        dist.reduce(loss_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            tot, cnt = loss_tensor.tolist()
            avg_loss = tot / (world_size * cnt)
            ep1 = time.perf_counter()
            print(
                f"✅ [Epoch {epoch + 1}/{args.epochs}] loss={avg_loss:.9f} — time={ep1 - ep0:.2f}s"
            )

            os.makedirs(args.ckpt_path, exist_ok=True)

            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    f"{args.ckpt_path}/epoch_{epoch + 1}.pth",
                    model,
                    optimizer,
                    epoch,
                    rank,
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    f"{args.ckpt_path}/best.pth", model, optimizer, epoch, rank
                )
                inference(model, device, test_ds, scaler, args, rank)

    t_end = time.perf_counter()
    if rank == 0:
        print(f"All done — total time {(t_end - t_start) / 60:.1f} min")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DDP MeshGraphNet on Plaid CFD")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--problem_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay_rate", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--scaler", choices=["none", "minmax", "standard"], default="standard"
    )
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt_path")
    parser.add_argument("--submission_path", type=str, default="./submission.pkl")
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--processor_size", type=int, default=15)
    parser.add_argument("--num_layers_node_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_processor", type=int, default=2)
    parser.add_argument("--hidden_dim_node_encoder", type=int, default=64)
    parser.add_argument("--num_layers_node_encoder", type=int, default=2)
    parser.add_argument("--hidden_dim_edge_encoder", type=int, default=64)
    parser.add_argument("--num_layers_edge_encoder", type=int, default=2)
    parser.add_argument("--hidden_dim_node_decoder", type=int, default=64)
    parser.add_argument("--num_layers_node_decoder", type=int, default=2)
    parser.add_argument(
        "--aggregation", type=str, choices=["sum", "mean", "max"], default="sum"
    )
    parser.add_argument("--activation", type=str, default="leaky")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=65325)
    args = parser.parse_args()

    main_worker(args)
