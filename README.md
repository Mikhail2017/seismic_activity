# Seismic Activity — First Break Picking

Self-contained env for the Gradio viewer/utilities **and** the official `hardpicks` package (for later training).

## Setup

### Local machine (conda env with torch)

```bash
chmod +x setup_env.sh
./setup_env.sh
conda activate seismic_activity
```

### Lightning AI / cloud (PyTorch preinstalled, no extra envs)

```bash
chmod +x setup_lightning.sh
./setup_lightning.sh
# optional: skip import checks
SKIP_VERIFY=1 ./setup_lightning.sh
```

This installs `requirements-lightning.txt` into the **current** Python and does **not** install/replace `torch` / `torchvision`.

## Run the viewer

```bash
conda activate seismic_activity
python viewer.py
```

Default data directory: `/home/mika/data/seismic_activity/`.

Gathers are **shot × receiver-line** images (task description / hardpicks split).

## Package layout

- `seismic_utils/sites.py` — per-asset config (FB field, REC_PEG digit count)
- `seismic_utils/dataset.py` — native HDF5 I/O + line-gather indexing
- `seismic_utils/hardpicks_bridge.py` — reuse hardpicks `ShotLineGatherDataset` when available
- `seismic_utils/plotting.py` — gather plots + optional before/after/unlabeled overlay
- `seismic_utils/export_npz.py` — per line-gather NPZ export
- `viewer.py` — Gradio UI
- `plot_class_balance.py` — global class histogram

### hardpicks reuse

**Default backend is hardpicks** (official `ShotLineGatherDataset`). Native splitter is opt-in.

```bash
# default (requires hardpicks + torch)
python viewer.py
python -m seismic_utils.export_npz /path/to/asset.hdf5 -o /path/to/npz

# opt into native splitter
SEISMIC_BACKEND=native python viewer.py
python -m seismic_utils.export_npz ... --backend native
```

## Export NPZ gathers

```bash
python -m seismic_utils.export_npz /path/to/asset.hdf5 -o /path/to/npz
```

## Training notebook

Adapted from hardpicks `fbp_train_with_api.ipynb`:

```bash
conda activate seismic_activity   # or Lightning Studio kernel after setup_lightning.sh
jupyter notebook examples/local/fbp_train_with_api.ipynb
```

Knobs at the top of the notebook: `SITE_NAME`, `MAX_EPOCHS`, `BATCH_SIZE`, `DATA_DIR`.
After `fit`, it prints per-epoch train/valid metrics, saves `train_valid_curves.png`, and re-validates the best checkpoint.
