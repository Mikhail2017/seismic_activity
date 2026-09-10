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

Browsing defaults to the **native** index/loader (fast). Predictions run on the
already-loaded gather (no full hardpicks HDF5 open). Force official hardpicks browsing
with `SEISMIC_BACKEND=hardpicks`.

**Prediction mode:** open *Prediction model*, point at a `best*.ckpt` (or `output/train_*/`), click **Load model**, then set **Pick display** to `prediction` or `both` (lime = model, yellow/red = reference).

## Package layout

- `seismic_utils/sites.py` — per-asset config (FB field, REC_PEG digit count)
- `seismic_utils/dataset.py` — native HDF5 I/O + line-gather indexing
- `seismic_utils/hardpicks_bridge.py` — reuse hardpicks `ShotLineGatherDataset` when available
- `seismic_utils/plotting.py` — gather plots + optional before/after/unlabeled overlay
- `seismic_utils/predict.py` — load FBPUNet checkpoint + per-gather FB prediction
- `seismic_utils/export_npz.py` — per line-gather NPZ export
- `seismic_utils/npz_parser.py` — fast hardpicks-compatible dataset from NPZ
- `viewer.py` — Gradio UI (reference / prediction / both pick display)
- `plot_class_balance.py` — global class histogram

### hardpicks reuse

**Default browse backend is native** (fast). Prediction opens hardpicks on demand.
Force official hardpicks loads with `SEISMIC_BACKEND=hardpicks`.

```bash
# default (fast native browse; hardpicks opens on first prediction)
python viewer.py
python -m seismic_utils.export_npz /path/to/asset.hdf5 -o /path/to/npz

# force hardpicks for browsing too (slower open)
SEISMIC_BACKEND=hardpicks python viewer.py
python -m seismic_utils.export_npz ... --backend native
```

## Export NPZ gathers

```bash
python -m seismic_utils.export_npz /path/to/asset.hdf5 -o /path/to/npz
```

## Train from NPZ (fast I/O)

After export, use the NPZ parser instead of live HDF5 reads:

```python
from seismic_utils.npz_parser import create_npz_parser

train_parser = create_npz_parser(
    "Lalor",
    npz_root="/path/to/npz",
    prefix="train",
    site_params={
        "normalize_samples": True,
        "augmentations": [{"type": "flip"}],
        "subset": {"eval_ratio": 0.15, "use_eval_split": False},
    },
    segm_class_count=1,
)
```

See `examples/local/fbp_train_with_api.ipynb` (`DATA_BACKEND = "npz"`).

## Training notebook

Adapted from hardpicks `fbp_train_with_api.ipynb`:

```bash
conda activate seismic_activity   # or Lightning Studio kernel after setup_lightning.sh
jupyter notebook examples/local/fbp_train_with_api.ipynb
```

Knobs at the top of the notebook: `SITE_NAME`, `MAX_EPOCHS`, `BATCH_SIZE`, `DATA_DIR`.
After `fit`, it prints per-epoch train/valid metrics, saves `train_valid_curves.png`, and re-validates the best checkpoint.
