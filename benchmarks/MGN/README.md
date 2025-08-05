# MeshGraphNet

This repository contains the implementation of the **MeshGraphNet (MGN)** used for predicting **steady** and **unsteady** flow problems.

---

## 📁 Project Structure

```
mgn/
├── configs/                   # JSON configuration files for different datasets
├── data.py                    # Data loading and preprocessing
├── main.py                    # Entry point for training steady flow problems
├── main_2d_elasto_ddp.py      # Entry point for training 2D Elasto-PlastoDynamics
├── model.py                   # Model architecture
├── train.py                   # Training logic
├── utils.py                   # Utilities
├── README.md                  # Project documentation
```

---

## 🚀 Launch Instructions

### ✅ Steady Flow Problems

Use `main.py` to train MGN on steady-state CFD datasets such as `2d_profile`, `tensile2d`, `rotor37`, `vkils59`, and `2d_multiscale`.

#### Example:
```bash
# Set configuration and runtime options
CONFIG_PATH="configs/2d_profile.json"
TARGET="Pressure"
RUN_NAME="mgn_Pressure"
SAVE_PATH="output"

# Launch training
python main.py \
  --config "$CONFIG_PATH" \
  --target_field "$TARGET" \
  --run_name "$RUN_NAME" \
  --save_path "$SAVE_PATH"
```

The `--target_field` argument defines the specific field (e.g., `U1`, `Pressure`, etc.) you want the model to predict. By default, the model will train on **all available fields** if this argument is omitted.

---

### 🔄 Unsteady Case: 2D Elasto-PlastoDynamics

Use `main_2d_elasto_ddp.py` to train on the 2D Elasto-PlastoDynamics problem using **Distributed Data Parallel (DDP)**.

#### Example:
```bash
srun python main_2d_elasto_ddp.py \
  --data_dir "$DATA_DIR" \
  --problem_dir "$PROBLEM_DIR" \
  --batch_size 16 \
  --epochs 100
```

Note: This script is compatible with Slurm-based HPC environments. If you're not using Slurm, replace `srun` with `python` as appropriate for your local setup.

---

## 🧩 Configuration Files

Located in the `configs/` folder, each `.json` file defines:
- Dataset paths
- Model hyperparameters
- Training/validation parameters

Examples:
- `2d_profile.json`
- `2d_multiscale.json`
- `tensile2d.json`
- `vkils59.json`
- `rotor37.json`

## ⚙️ Dependencies

Install the required Python libraries before running the scripts:
- torch
- dgl
- muscat
- plaid
- numpy
- sklearn
