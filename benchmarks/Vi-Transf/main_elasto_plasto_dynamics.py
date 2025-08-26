import datetime
import logging
import os
import pickle
import time

import hydra
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.utils import split_temporal_pyg_train_test


def seed_everything(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NoWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove all handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(f"{log_dir}/train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    file_handler.addFilter(NoWarningFilter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    console_handler.addFilter(NoWarningFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="2d_elasto_plasto_dynamics.yaml",
)
def main(cfg):
    # Use Hydra's run directory for all outputs & logging
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    log_dir = os.path.join("logs", cfg.bridge.data_info.name, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=False)
    tb_logger = SummaryWriter(log_dir=log_dir)
    logger = setup_logger(log_dir)

    # write config to log_dir:
    with open(f"{log_dir}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # loading components
    loader = hydra.utils.instantiate(cfg.loader)
    scaler = hydra.utils.instantiate(cfg.scaler)
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer, params=model.parameters()
    )

    seed = cfg.get("seed", None)
    if seed is not None:
        logger.info(f"Using seed {seed}")
        seed_everything(seed)

    # loading datasets
    logger.info("\n------\nData pre-processing\n------")
    preprocessing_start = time.time()
    problem_definition, plaid_train_dataset, plaid_test_dataset = loader.load_plaid()
    test_ids = plaid_test_dataset.get_sample_ids()
    train_ids = plaid_train_dataset.get_sample_ids()
    del plaid_train_dataset, plaid_test_dataset

    problem_definition, train_dataset, test_dataset = loader.load()

    # splitting datasets into train/val splits
    train_ids, val_ids = train_test_split(train_ids, test_size=cfg.training.test_size)
    print(train_ids)
    print(val_ids)
    logger.info(f"Train_ids: {train_ids}")
    logger.info(f"Val_ids: {val_ids}")

    split_train_dataset, split_val_dataset = split_temporal_pyg_train_test(
        train_dataset, train_ids=train_ids, test_ids=val_ids
    )
    train_dataset, val_dataset = split_train_dataset, split_val_dataset

    # unpacking
    train_dataset = [sample for sample_list in train_dataset for sample in sample_list]
    test_dataset = [sample for sample_list in test_dataset for sample in sample_list]
    val_dataset = [sample for sample_list in val_dataset for sample in sample_list]

    # scaling the pyg dataset
    train_dataset = scaler.fit_transform(train_dataset)
    val_dataset = scaler.transform(val_dataset)
    test_dataset = scaler.transform(test_dataset)

    # repacking by sample id
    packed_train_dataset = []
    for id in train_ids:
        sample_list = [sample for sample in train_dataset if sample.sample_id == id]
        packed_train_dataset.append(sample_list)
    packed_test_dataset = []
    for id in test_ids:
        sample_list = [sample for sample in test_dataset if sample.sample_id == id]
        packed_test_dataset.append(sample_list)
    packed_val_dataset = []
    for id in val_ids:
        sample_list = [sample for sample in val_dataset if sample.sample_id == id]
        packed_val_dataset.append(sample_list)

    train_dataset, test_dataset, val_dataset = (
        packed_train_dataset,
        packed_test_dataset,
        packed_val_dataset,
    )

    # preprocessing the datasets
    train_dataset = model.preprocess(
        pyg_dataset=train_dataset,
        plaid_dataset=None,
        seed=cfg.model.preprocessing.get("train_seed", None),
        type="train",
    )

    val_dataset = model.preprocess(
        pyg_dataset=val_dataset,
        plaid_dataset=None,
        seed=cfg.model.preprocessing.get("val_seed", None),
        type="val",
    )
    preprocessing_end = time.time()
    logger.info(
        f"Preprocessing time: {str(datetime.timedelta(seconds=(preprocessing_end - preprocessing_start)))}"
    )

    # training
    logger.info("\n--------\nTraining\n--------")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    best_val_loss = float("inf")
    best_model_epoch = 0
    best_model_path = os.path.join(log_dir, "best_model.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lbda = cfg.training.lbda
    logger.info(f"Using lbda: {lbda}")
    logger.info(f"Using device: {device}")
    epochs = cfg.training.num_epochs
    model.to(device)

    output_fields_names = train_dataset[0].output_fields_names
    output_scalars_names = train_dataset[0].output_scalars_names

    train_start = time.time()
    for epoch in range(epochs):
        # train loop
        epoch_train_loss = 0
        epoch_train_field_losses = torch.zeros(len(output_fields_names))
        epoch_train_scalar_losses = torch.zeros(len(output_scalars_names))

        model.train()
        for batch_id, batch in enumerate(train_loader):
            local_batch_size = batch.num_graphs
            field_losses, scalar_losses = model.compute_loss(batch)
            field_loss, scalar_loss = field_losses.mean(), scalar_losses.mean()
            loss = lbda * field_loss + (1 - lbda) * scalar_loss

            # recording losses
            epoch_train_field_losses += field_losses.detach().cpu() * (
                local_batch_size / len(train_dataset)
            )
            epoch_train_scalar_losses += scalar_losses.detach().cpu() * (
                local_batch_size / len(train_dataset)
            )
            epoch_train_loss += loss.item() * (local_batch_size / len(train_dataset))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for n, fn in enumerate(output_fields_names):
            tb_logger.scalars.add(
                f"train/loss/{fn}", epoch_train_field_losses[n].item(), epoch
            )
        for n, sn in enumerate(output_scalars_names):
            tb_logger.scalars.add(
                f"train/loss/{sn}", epoch_train_scalar_losses[n].item(), epoch
            )
        tb_logger.scalars.add("train/loss", epoch_train_loss, epoch)

        # validation loop
        epoch_val_loss = 0
        epoch_val_field_losses = torch.zeros(len(output_fields_names))
        epoch_val_scalar_losses = torch.zeros(len(output_scalars_names))

        model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(val_loader):
                local_batch_size = batch.num_graphs
                field_losses, scalar_losses = model.compute_loss(batch)
                field_loss, scalar_loss = field_losses.mean(), scalar_losses.mean()
                loss = lbda * field_loss + (1 - lbda) * scalar_loss

                # recording losses
                epoch_val_field_losses += field_losses.detach().cpu() * (
                    local_batch_size / len(val_dataset)
                )
                epoch_val_scalar_losses += scalar_losses.detach().cpu() * (
                    local_batch_size / len(val_dataset)
                )
                epoch_val_loss += loss.item() * (local_batch_size / len(val_dataset))

        for n, fn in enumerate(output_fields_names):
            tb_logger.scalars.add(
                f"val/loss/{fn}", epoch_val_field_losses[n].item(), epoch
            )
        for n, sn in enumerate(output_scalars_names):
            tb_logger.scalars.add(
                f"val/loss/{sn}", epoch_val_scalar_losses[n].item(), epoch
            )
        tb_logger.scalars.add("val/loss", epoch_val_loss, epoch)
        logger.info(
            f"Epoch {epoch:>{len(str(epochs))}}: Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}"
        )

        # save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            best_model_epoch = epoch
            best_model_time = time.time()
    train_end = time.time()
    logger.info("")
    logger.info(
        f"Training time: {str(datetime.timedelta(seconds=(train_end - train_start)))}"
    )
    logger.info(
        f"Saved best model at epoch {best_model_epoch} with loss {best_val_loss:.5f}"
    )
    logger.info(
        f"Training time to reach the best model epoch: {str(datetime.timedelta(seconds=(best_model_time - train_start)))}"
    )

    # loading the best model
    logger.info("Loading the best saved model")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    del train_dataset, val_dataset

    buffer = test_dataset
    # test dataset predictions
    test_dataset = model.preprocess(
        pyg_dataset=buffer,
        plaid_dataset=None,
        seed=cfg.model.preprocessing.get("test_seed", None),
        type="test",
    )

    predictions_dict = {}
    model.eval()
    first_samples = {}
    for id in test_ids:
        for _data in test_dataset:
            if _data.sample_id == id:
                first_samples[id] = _data

    import copy

    for id in test_ids:
        data = first_samples[id]

        with torch.no_grad():
            n_timesteps = len(data.timestep_list)
            instantaneous_predictions = torch.zeros(
                (data.x.shape[0], len(data.output_fields_names))
            )
            field_predictions_concat = torch.empty(
                (n_timesteps, data.x.shape[0], len(data.output_fields_names))
            )
            field_predictions_concat[0] = instantaneous_predictions
            buffer_list = []
            for n, ts in enumerate(data.timestep_list[:-1]):
                input_data = copy.deepcopy(data)
                input_data.x[:, -len(input_data.output_fields_names) :] = (
                    instantaneous_predictions
                )
                input_data.input_scalars = torch.tensor([[ts]], dtype=torch.float32)
                input_data.time = ts

                instantaneous_predictions, _ = model.predict(input_data)
                buffer = Data(
                    output_fields=instantaneous_predictions,
                    output_fields_names=output_fields_names,
                )
                buffer_list.append(buffer)
            unscaled_solutions = scaler.inverse_transform_prediction(buffer_list)
            for n, ts in enumerate(data.timestep_list[:-1]):
                field_predictions_concat[n + 1] = unscaled_solutions[n].output_fields
            predictions_dict[data.sample_id] = field_predictions_concat

    # creating submission
    reference = []
    for i, id in enumerate(test_ids):
        reference.append({})
        for n, fn in enumerate(output_fields_names):
            reference[i][fn] = predictions_dict[id][..., n].numpy()

    with open(os.path.join(log_dir, "reference.pkl"), "wb") as file:
        pickle.dump(reference, file)

    logger.info(
        "\n\
    ---------------------\n\
    ------Finished!------\n\
    ---------------------"
    )


if __name__ == "__main__":
    main()
