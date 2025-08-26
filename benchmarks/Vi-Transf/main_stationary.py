import datetime
import logging
import os
import time

import hydra
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data.utils import split_plaid_train_test, split_pyg_train_test
from src.evaluation.submission import create_submission


def seed_everything(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove all handlers associated with the root logger object (Hydra may add its own)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


@hydra.main(version_base=None, config_path="configs", config_name="rotor37.yaml")
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
    problem_definition, train_dataset, test_dataset = loader.load()

    # splitting datasets into train/val splits
    train_ids = plaid_train_dataset.get_sample_ids()
    train_ids, val_ids = train_test_split(train_ids, test_size=cfg.training.test_size)
    logger.info(f"Train_ids: {train_ids}")
    logger.info(f"Val_ids: {val_ids}")

    plaid_train_dataset, plaid_val_dataset = split_plaid_train_test(
        plaid_train_dataset, train_ids=train_ids, test_ids=val_ids
    )
    train_dataset, val_dataset = split_pyg_train_test(
        train_dataset, train_ids=train_ids, test_ids=val_ids
    )

    # scaling the pyg dataset
    train_dataset = scaler.fit_transform(train_dataset)
    val_dataset = scaler.transform(val_dataset)
    test_dataset = scaler.transform(test_dataset)

    # preprocessing the datasets
    train_dataset = model.preprocess(
        pyg_dataset=train_dataset,
        plaid_dataset=plaid_train_dataset,
        seed=cfg.model.preprocessing.get("train_seed", None),
        type="train",
    )

    val_dataset = model.preprocess(
        pyg_dataset=val_dataset,
        plaid_dataset=plaid_val_dataset,
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

    # test dataset predictions
    test_dataset = model.preprocess(
        pyg_dataset=test_dataset,
        plaid_dataset=plaid_test_dataset,
        seed=cfg.model.preprocessing.get("test_seed", None),
        type="test",
    )

    prediction_dataset = []
    model.eval()
    for data in tqdm(test_dataset, desc="Test", total=len(test_dataset)):
        with torch.no_grad():
            fields_predictions, scalars_predictions = model.predict(data)
            prediction_data = Data()
            prediction_data.fields_prediction = fields_predictions
            prediction_data.scalars_prediction = scalars_predictions

            prediction_data.output_fields_names = data.output_fields_names
            prediction_data.output_scalars_names = data.output_scalars_names
            prediction_data.sample_id = data.sample_id
            prediction_dataset.append(prediction_data)

    post_processed_dataset = model.postprocess(
        prediction_dataset, plaid_test_dataset, type="test"
    )
    prediction_dataset_unscaled = scaler.inverse_transform_prediction(
        post_processed_dataset
    )

    # creating submission
    logger.info("\n------\nSubmission\n------")
    create_submission(prediction_dataset_unscaled, plaid_test_dataset, save_dir=log_dir)

    logger.info("\n---------------------\n------Finished!------\n---------------------")


if __name__ == "__main__":
    main()
