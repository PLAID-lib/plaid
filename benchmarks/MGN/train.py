import os
import time
import torch
import pandas as pd
from data import GraphDataset
from dgl.dataloading import GraphDataLoader
from utils import save_fields, relative_rmse_field


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(torch.cuda.device_count()):
   print(f"💻 Using device {i}: {torch.cuda.get_device_properties(i).name}")


def train(args, model, optimizer, loss_fn, train_data, test_data):
    model.to(device)

    # Create necessary directories
    checkpoint_dir = os.path.join(args.save_path, f"{args.dataset_name}/{args.run_name}/checkpoints")
    predictions_dir = os.path.join(args.save_path, f"{args.dataset_name}/{args.run_name}/predictions")
    metrics_dir = os.path.join(args.save_path, f"{args.dataset_name}/{args.run_name}/metrics")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Dataset
    train_dataset = GraphDataset(args, train_data, data_type="train")
    test_dataset = GraphDataset(args, test_data, data_type="test", in_scaler=train_dataset.in_scaler, out_scaler=train_dataset.out_scaler,
                                fields_min=train_dataset.fields_min, fields_max=train_dataset.fields_max)

    fields_min = train_dataset.fields_min.clone().detach().to(device)
    fields_max = train_dataset.fields_max.clone().detach().to(device)

    # Dataloader
    train_dataloader = GraphDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_dataloader = GraphDataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"⚙️ Total number of model parameters: {total_params}")

    # Record the start time
    start_time = time.time()

    # Data structure for storing metrics
    metrics = []

    # Training loop
    num_epochs = args.num_epochs

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss = 0.0
        model.train()
        y_trains, y_train_preds = [], []

        for idx, (graph, _, _) in enumerate(train_dataloader):
            optimizer.zero_grad()

            graph = graph.to(device)
            pred = model(graph.ndata["x"], graph.edata["f"], graph)
            loss = loss_fn(graph.ndata["y"], pred)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            y_trains.append(graph.ndata["y"])
            y_train_preds.append(pred)

        train_loss /= (idx+1)

        model.eval()
        y_test_preds = []

        with torch.no_grad():
            for idx, (graph, _, _) in enumerate(test_dataloader):
                graph = graph.to(device)
                pred = model(graph.ndata["x"], graph.edata["f"], graph)

                pred = pred * (fields_max - fields_min) + fields_min
                y_test_preds.append(pred)

        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"state_epoch_{epoch+1}.pt"))
            save_fields(os.path.join(predictions_dir, f"predicted_fields_{epoch+1}.h5"), y_test_preds)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Collect metrics for this epoch
        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "duration": epoch_duration
        })

        metrics_str = (f"🌟"
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.7f} | "
            f"Duration: {epoch_duration:.2f} (s) "
        )
        print(metrics_str)

    # Saving collected metrics to a CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(metrics_dir, "metrics.csv"), index=False)

    # Record the end time
    end_time = time.time()

    # Calculate the training duration
    training_duration = end_time - start_time
    print(f"⏰ Training took {training_duration:.7f} seconds")
