# Training pipeline (`train/fbp_train.py`)

This is what **this repo** actually trains. The paper / hardpicks search recipe is
[`original_hardpicks.md`](original_hardpicks.md). Differences are listed at the end.

Launch:

```bash
conda activate seismic_activity
python train/fbp_train.py --fold A --model resnet34
python train/fbp_train.py --config configs/train.yaml --fold A
python train/fbp_train.py --fold A --patience 0          # train all --epochs
python train/fbp_train.py --fold A --ckpt output/train_foldA_resnet34/best-epoch=013-step=015232.ckpt
python train/fbp_train.py --fold A --ckpt-dir output/train_foldA_resnet34
python train/fbp_train.py --sites Brunswick,Halfmile --backend npz
python train/fbp_train.py --list-folds
python train/fbp_train.py --list-models
```

Checkpointing and early stopping both watch **`valid/HitRate1px`**. Live notes go to
`report/train_<label>_<model>_<YYYYMMDD_HHMMSS>/`. Weights, `model_config.yaml`, and a
copy of the recipe go to `output/train_<label>_<model>/`.

Loop / split / loss / LR defaults live in [`configs/train.yaml`](configs/train.yaml)
(`--config`). CLI flags override the recipe. `--fold`, `--sites`, paths, GPUs, and
`--ckpt` stay on the command line. Precedence for `loss` / `lr` / `lr_step` /
`encoder_weights`: recipe YAML → `--model-config` → CLI.

After fit, score the best checkpoint with [`train/fbp_eval.py`](train/fbp_eval.py)
(same `--fold` or `--sites`).

## Data

Sites in this repo: **Brunswick, Halfmile, Lalor, Sudbury**. There is no Kevitsa
or Matagami.

A training example is one **shot × receiver-line gather** (a 2D image: time ×
receivers). HDF5 (`--backend hdf5`) uses hardpicks parsers and the combined
bad-gather reject list when present. NPZ (`--backend npz`, default) reads
pre-exported gathers from `<data-dir>/npz` (faster I/O).

Preprocessing (hardpicks `ShotLineGatherPreprocessor` defaults):

- Per-trace amplitude normalization to \([-1, 1]\) (abs-max).
- Offset channels: shot–receiver / **3000 m**, neighbor distances / **50 m**.
- HDF5 traces are converted to fp16 in the parser (NPZ keeps the exported dtype).

The U-Net input is **4 channels**: normalized amplitude plus the three offset
maps, tiled along time. First-break prior maps are **off**.

## Splits

`--fold` and `--sites` are mutually exclusive. There is **no test site**.

### `--fold` (whole-site holdout)

Whole surveys are held out. `--eval-ratio` is ignored. Fold A train/valid sites
match hardpicks `foldA.yaml`; B–D drop Kevitsa/Matagami; F–K match the hardpicks
YAMLs that already omit Kevitsa. Fold **E** cannot run here.

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

G and K are the same split (hardpicks defines both letters).

### `--sites` (intra-site holdout)

Each listed site is split with `--eval-ratio` (default **0.15**) of shot IDs
**and** line IDs; valid is the union. Default if neither flag is set:
`Brunswick,Halfmile`.

Train and valid gather keys are asserted disjoint.

## Augmentations

Applied on **train only**. Validation is never augmented.

- **crop** — random time crop to 512 or 1024 samples (`max_crop_fraction=0.333`)
- **kill** — each trace zeroed with probability **0.08**
- **drop_and_pad** — resample line length to \(\{64, 128, 256, 512\}\) traces
  (`full_snap=true`, `max_drop_ratio=0.50`)
- **flip** — horizontal flip along the receiver axis

The collate function pads a minibatch to a common **power-of-two** size so the
U-Net can stack gathers. Padded traces are marked invalid (`rec_ids == -1`).

## Model

Local [`models.fbp.unet.FBPUNet`](models/fbp/unet.py) (cloned from hardpicks).
Fully convolutional encoder–decoder with skip connections.

`--model` presets: `resnet18` (default), `resnet34`, `resnet50`,
`efficientnet-b0`, `efficientnet-b4`, `vanilla`, or any SMP encoder name.
`--encoder-weights imagenet` (etc.) initializes the backbone; default is train
from scratch. `--model-config path.yaml` merges on top of the preset.

Segmentation is **binary** (first break vs not). Masks are generated with
`segm_class_count=1` for both train and valid parsers.

Loss:

- `--loss crossentropy` (default)
- `--loss dice`

Pick at inference: sample index of the highest first-break probability on each
trace (`segm_first_break_prob_threshold=0`).

## Optimization

| Knob | Default |
| --- | --- |
| Recipe file | [`configs/train.yaml`](configs/train.yaml) (`--config`) |
| Optimizer | Adam, weight decay \(10^{-6}\) |
| Learning rate | preset-specific (resnet18/34: `0.002136`); override with `--lr` |
| Scheduler | `StepLR`, \(\gamma=0.1\), `--lr-step` **10** (choices 5 / 10 / 20). Step 20 with `--epochs 20` does not decay during the run. |
| Batch size | **16 per GPU**. Global batch ≈ `16 × devices`. |
| Max epochs | **20** |
| Early stopping | patience **4** on `valid/HitRate1px`. `--patience 0` disables it. |
| Seed | `0` (`--seed`) |
| Precision | `32` (`--precision 16-mixed` is available) |

Best checkpoint: `output/.../best-epoch=…-step=….ckpt` (highest `valid/HitRate1px`).
After fit, the trainer re-validates that checkpoint unless `--no-final-validate`.

Resume a previous run (restores weights, optimizer, scheduler, and epoch):

```bash
python train/fbp_train.py --fold A --model resnet34 --ckpt path/to/best.ckpt
python train/fbp_train.py --fold A --ckpt-dir output/train_foldA_resnet34
```

Use the same `--model` (and architecture flags) as the original run. Only
`best-*.ckpt` is saved (`save_top_k=1`), so resume is from the best HR@1
checkpoint, not necessarily the last epoch. Raise `--epochs` if you need a
higher cap than the original run. `--output-dir` of the original experiment
keeps new `best-*.ckpt` files in the same folder.

Multi-GPU: batch size stays per GPU. Prefer
`torchrun --nproc_per_node=N train/fbp_train.py --fold A --devices N`.

## Metrics

Logged every validation epoch (and written into the living report):

- HitRate @ 1, 3, 5, **7, 9** px (absolute error in samples, `< buffer`)
- MAE, RMSE, MBE (samples)
- GatherCoverage (trace coverage / TC)

**Monitor / early stop / ModelCheckpoint:** `valid/HitRate1px` only.

[`train/fbp_eval.py`](train/fbp_eval.py) recomputes those scalars on the valid
split, plus millisecond conversions, offset bins, and worst/typical gather
overlays. Use the same `--fold` as training.

## Outputs

```
output/train_<sites-or-fold>_<model>/
  best-epoch=…-step=….ckpt
  model_config.yaml
  train_recipe.yaml
  data_split.yaml
  tensorboard/  csv_logs/
  epoch_metrics.csv  train_valid_curves.png

report/train_<label>_<model>_<YYYYMMDD_HHMMSS>/
  report.md  train_steps.csv  valid_epochs.jsonl  final.json
```

## Versus `original_hardpicks.md`

Aligned with the paper/hardpicks fold YAML: line gathers, abs-max + offset
normalization, crop / kill / drop_and_pad / flip, Adam + wd, StepLR, HR@1
selection, validation metrics including HR@7/9 / RMSE / coverage, default
batch 16 and 20 epochs with patience 4.

Not in this trainer:

- **Orion** 50-trial hyperparameter search (one config per run; use `--loss` /
  `--model` / `--lr` / `--lr-step` explicitly)
- **Ternary** (before / first-break / after) heads — this trainer is binary only
- **10 random seeds** on an unseen **test** site (Kevitsa is not in this dataset;
  `--fold A` valid is Halfmile, the original *validation* site)
- Fold **E** / five-site train–valid–test
- EfficientNet-B2 and `[1024,…]` decoder as named presets (possible via
  `--model` / `--model-config`)
- Mixed-precision default (fold YAMLs used 16; this CLI defaults to 32)
