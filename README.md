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
- `seismic_utils/fbp_eval_report.py` — validation report (scalars, Plotly HTML, gather PNGs)
- `models/` — local FBPUNet (cloned from hardpicks) used by `train/fbp_train.py`
- `train/fbp_train.py` — training CLI (folds, multi-GPU, living `report/` log)
- `configs/train.yaml` — default training recipe (epochs, batch, patience, save_top_k, loss/LR)
- `TRAINING.md` — actual training pipeline (this repo)
- `original_hardpicks.md` — paper / hardpicks search recipe
- `train/fbp_eval.py` — checkpoint validation / prediction report CLI
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

See `train/fbp_train.py` or the notebook (`DATA_BACKEND = "npz"`).

## Training

Full description of **this repo’s** trainer: [`TRAINING.md`](TRAINING.md)
(data, folds, augs, loss, schedule, metrics). The paper / hardpicks search
recipe is [`original_hardpicks.md`](original_hardpicks.md).

CLI — TensorBoard + CSV logs, living report, best checkpoint on `valid/HitRate1px`.
Defaults come from [`configs/train.yaml`](configs/train.yaml) (**20** epochs, **16**
gathers per GPU, patience **4**, train augmentations). Override loop knobs with
`--config` or CLI flags; edit `augmentations:` in the YAML (empty list = none).

```bash
conda activate seismic_activity   # or Lightning Studio kernel after setup_lightning.sh
python train/fbp_train.py --fold A --model resnet34
python train/fbp_train.py --config configs/train.yaml --fold A
python train/fbp_train.py --fold A --patience 0          # no early stop
python train/fbp_train.py --fold A --save-top-k -1       # checkpoint after every epoch
python train/fbp_train.py --fold A --ckpt-dir output/train_foldA_resnet34
python train/fbp_train.py --sites Brunswick,Halfmile --backend npz
python train/fbp_train.py --list-models
python train/fbp_train.py --list-folds

# watch metrics
tensorboard --logdir output/train_foldA_resnet34/tensorboard
```

Useful flags: `--config`, `--model` / `--model-config`, `--loss`, `--lr`,
`--lr-step`, `--batch-size`, `--epochs`, `--patience`, `--save-top-k`, `--ckpt` / `--ckpt-dir`, `--num-workers`,
`--npz-root`, `--output-dir`, `--encoder-weights`, `--no-final-validate`.

Model presets (`--model`): `resnet18` (default), `resnet34`, `resnet50`, `efficientnet-b0`,
`efficientnet-b4`, `vanilla`, or any SMP encoder name. Full hyperparam overrides via
`--model-config path.yaml` (merged on top of the preset). Resolved config is written to
`output/.../model_config.yaml`.

The trainer is the **local** `models.fbp.unet.FBPUNet` (cloned from hardpicks). Optional
ImageNet (etc.) backbone init: `--encoder-weights imagenet`.

Notebook equivalent:

```bash
jupyter notebook examples/local/fbp_train_with_api.ipynb
```

Both write under `output/train_<sites>/`: best checkpoint (`valid/HitRate1px`),
`epoch_metrics.csv`, and `train_valid_curves.png`. Live training notes go to
`report/<run>_<YYYYMMDD_HHMMSS>/`.

## Site folds

`--fold` is a **whole-site holdout** (no intra-site `--eval-ratio` split). It is
mutually exclusive with `--sites`. Sites in this repo: **Brunswick, Halfmile,
Lalor, Sudbury**. Hardpicks also has Kevitsa/Matagami; those files are not here,
so fold **E** cannot run.

Folds A–D are leave-one-site-out (3 train / 1 valid). Fold A matches hardpicks
`foldA.yaml` exactly; B–D are the same rotation with Kevitsa/Matagami dropped.
F–K match the hardpicks YAMLs that already omit Kevitsa (2 train / 1 valid).

```bash
python train/fbp_train.py --fold A --model resnet34
python train/fbp_train.py --list-folds
```

Accepts `A`, `foldA`, `fold_a`. The same `--fold` is used at eval time so the
validation site matches training.

| Fold | Train | Valid |
| --- | --- | --- |
| A | Lalor, Brunswick, Sudbury | Halfmile |
| B | Lalor, Brunswick, Halfmile | Sudbury |
| C | Halfmile, Lalor, Sudbury | Brunswick |
| D | Sudbury, Halfmile, Brunswick | Lalor |
| E | — | unavailable (needs Matagami/Kevitsa) |
| F | Halfmile, Brunswick | Sudbury |
| G | Brunswick, Sudbury | Halfmile |
| H | Halfmile, Lalor | Brunswick |
| I | Sudbury, Halfmile | Lalor |
| J | Lalor, Brunswick | Sudbury |
| K | Brunswick, Sudbury | Halfmile |

G and K have the same split; they exist as separate letters because hardpicks
defines both YAMLs.

## Validation / prediction report

After training, score a checkpoint on the fold (or site) validation split and write
headline metrics, error histograms, offset slices, and worst/typical gather overlays:

```bash
python train/fbp_eval.py --ckpt-dir output/train_foldA_resnet34 --fold A --backend hdf5 --rmse-above 7
python train/fbp_eval.py --ckpt-dir /path/to/weights/baseline/foldA --fold A --backend hdf5 --data-dir /tmp/data/
python train/fbp_eval.py --ckpt output/train_foldA_resnet34/best-epoch=013-step=015232.ckpt --fold A --backend npz
```

`--fold` uses the same whole-site holdout as training (see **Site folds** above).
`--sites` evaluates those sites; add `--eval-ratio 0.15` to reuse the intra-site
holdout from `fbp_train.py`.

Writes `report/eval_<label>_<encoder>_<YYYYMMDD_HHMMSS>/`:

- `report.md` / `index.html` — headline HR@1–9, MAE, median/P90, RMSE, MBE (samples and ms)
- `stats.html` — Plotly error histogram, CDF, HR/MAE vs offset
- `worst.html` / `typical.html` — matplotlib gather overlays (PNG)
- `traces.parquet` (or `traces.csv.gz`) — per-trace predictions and errors
- `metrics.json`, `gathers.csv`, `offset_bins.csv`

Open `index.html` in a browser. Plotly HTML loads Plotly from CDN.
