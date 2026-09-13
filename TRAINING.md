`--model` presets: `resnet18` (default), `resnet34`, `resnet34-horizon`,
`resnet50`, `efficientnet-b0`, `efficientnet-b4`, `vanilla`, `meneses`, or any SMP encoder
name. `resnet34-horizon` is ResNet34 plus the linear-moveout first-break prior
channel (5th input). A `*-horizon` suffix on any preset/SMP encoder does the
same. Before/after masks: `--picker before_after`, YAML `picker: before_after`,
or `model: resnet34-before-after` (combinable with `-horizon`).
`--encoder-weights imagenet` (etc.) initializes the backbone; default is
train from scratch. `--model-config path.yaml` merges on top of the preset.

## Site folds

`--fold` is a **whole-site holdout**: every gather from a listed train site
goes to training, and every gather from the valid site goes to validation.
There is **no** intra-site `--eval-ratio` split when `--fold` is set.
`--fold` and `--sites` are mutually exclusive.

Sites in this repo: **Brunswick**, **Halfmile**, **Lalor**, **Sudbury**.
Hardpicks also defines Kevitsa/Matagami; those files are not here, so fold
**E** cannot run.

```bash
python train/fbp_train.py --fold A
python train/fbp_train.py --list-folds
```

Accepts `A`, `foldA`, `fold_a`. Use the same `--fold` with `train/fbp_eval.py`
so the validation site matches training.

### Leave-one-site-out (3 train / 1 valid)

A–D rotate the four local sites. Fold A matches hardpicks `foldA.yaml`
exactly; B–D are the same rotation with Kevitsa/Matagami dropped.

| Fold | Train | Valid |
| --- | --- | --- |
| A | Lalor, Brunswick, Sudbury | Halfmile |
| B | Lalor, Brunswick, Halfmile | Sudbury |
| C | Halfmile, Lalor, Sudbury | Brunswick |
| D | Sudbury, Halfmile, Brunswick | Lalor |
| E | — | unavailable (needs Matagami/Kevitsa) |

### Hardpicks YAMLs that already omit Kevitsa (2 train / 1 valid)

F–K copy the hardpicks fold YAMLs that never included Kevitsa.

| Fold | Train | Valid |
| --- | --- | --- |
| F | Halfmile, Brunswick | Sudbury |
| G | Brunswick, Sudbury | Halfmile |
| H | Halfmile, Lalor | Brunswick |
| I | Sudbury, Halfmile | Lalor |
| J | Lalor, Brunswick | Sudbury |
| K | Brunswick, Sudbury | Halfmile |

G and K are the same split; both letters exist because hardpicks ships both
YAMLs.

### `--sites` (no named fold)

If you pass `--sites Brunswick,Halfmile` instead of `--fold`, those sites are
used for **both** train and valid, and `eval_ratio` (default `0.15` from the
recipe YAML) holds out a random subset of shots/lines inside the sites.

Defined in `SITE_FOLDS` / `UNAVAILABLE_FOLDS` in `train/fbp_train.py`.

## Validation and checkpoint consistency

HDF5 training/evaluation uses the tracked local `OwnedMetadataGatherDataset`.
It caches raw metadata but copies arrays on every read, before cleaning,
cropping, or normalization. This prevents cumulative offset normalization and
permanent loss of cropped labels. Restart running Python processes to use it.
Old runs using shared cached metadata should be retrained for a clean baseline;
revalidating their weights cannot undo corrupted training inputs.

Lightning controls train/eval mode normally. The before/after evaluator must be
restored on fresh checkpoint loads; forcing BatchNorm eval repeatedly is not a
substitute for repeatable input data. New checkpoints record validation metrics,
and post-fit validation raises if any metric differs beyond numerical tolerance.

Mean cross-entropy validation loss is weighted by non-ignored target pixels
(class weights included), rather than averaging batch means. Other losses use
gather-count weighting and can still depend on batch composition. Distributed
validation reduces error sums, squared-error sums, hits and counts globally,
excluding duplicates added by the non-shuffled distributed sampler. Use
`build_loaders` for distributed validation: it supplies original dataset indices.

Standalone evaluation defaults to the saved training batch size (4 for legacy
checkpoints without that metadata). An explicit different batch size warns:
batch-dependent spatial padding can change CNN predictions, even in eval mode.
Before/after picks exclude time padding. The legacy equal-window-sum smoother
is retained for checkpoint compatibility; it does **not** guarantee a contiguous
run of `smooth_threshold` samples. Changing that selection rule is a separate
model/decoder experiment, not a validation consistency fix.

## Resume versus weights-only initialization

`--ckpt` / `--ckpt-dir` resumes optimizer, epoch and the manually managed scheduler.
Saved task, architecture, loss, optimizer/scheduler and training-data settings
must match. Increasing `--epochs` is allowed for StepLR, but not for a fixed-horizon
warmup/cosine schedule. This preserves scheduler progression; it does not promise
bit-identical stochastic augmentation or worker RNG sequences after restarting.

Legacy checkpoints without scheduler state cannot be exact resumes. Use
`--init-ckpt /absolute/path/to/model.ckpt` for explicit weights-only initialization
with a new optimizer/scheduler. The picker must match; this is fine-tuning, not
continuation of the old run.

## Run artifacts

Default output directories include a unique run ID. Explicit `--output-dir`
must be empty; previous checkpoints/configs are never intentionally overwritten.
For distributed launches, `torchrun --standalone` supplies a shared run ID;
static/custom launchers without an elastic run ID must set a unique
`SEISMIC_RUN_ID` shared by all ranks. Clear that variable when
starting another run in the same Python process/environment.

The checkpoint callback writes `best_checkpoint.json`, pointing to the actual
highest-scoring file, including when `--save-top-k -1` retains every epoch.
Directory-based loading follows this manifest. Legacy directories with multiple
checkpoints require an explicit `--ckpt`; modification time is not a score.
CSV summaries use the active logger's file, not a lexicographic version search.

The recipe on the training machine matters: `--picker before_after --fold A`
alone does not select ResNet34, HDF5, four epochs, or batch size 32. Specify those
flags or preserve the resolved recipe/model config when reproducing a run.

## Lateral pick cleaner (eval only, no retrain)

After decode, isolated pick crashes (line-end dives, single-trace jumps) are
flagged against a robust pick-vs-offset fit and replaced from neighboring
anchors. **Predictions are rewritten; labels are not.**
Do not enable hardpicks `auto_fill_missing_picks` on the valid parser.

Same checkpoint as a normal eval, plus `--lateral-clean`:

```bash
python train/fbp_eval.py --picker before_after \
  --ckpt output/train_foldA_resnet18-before-after_.../best-epoch=019-step=043500.ckpt \
  --fold A --backend hdf5 --data-dir /tmp/data \
  --lateral-clean
```

Compare the new `report/eval_…` to the run without the flag (headline MAE / RMSE
and the worst-gather PNGs). Expect RMSE/P90 on tail crashes to drop; systematic
early picks on a whole gather (`01` / `02` style) stay unchanged.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--lateral-clean` | off | Enable the post-process |
| `--lateral-window` | 15 | Traces in the local-median window |
| `--lateral-max-dev` | 15 | Flag if \|pick − median\| exceeds this many samples |
| `--lateral-max-flag-frac` | 0.30 | Skip the gather if this fraction of valid picks would be flagged |
| `--lateral-min-anchors` | 3 | Unflagged picks required before interpolation |

`metrics.json` records `lateral_clean.n_replaced`. `report.md` shows **Lateral clean: on**.

This is not a substitute for `--smooth-threshold` (that only changes the
before/after decoder). Use the cleaner for along-line outliers after decode.

## Minimal annotations (Meneses self-training)

Site-specific replica of Meneses et al. 2026: train on ~1% labelled gathers
(paper counts 148 / 54 / 120 / 43, then 75/25 train/val), score the remaining
99% against **manual** picks. This path is **not** compatible with `--fold`,
`before_after`, `-horizon`, or GeoNorm.

```bash
python train/fbp_self_train.py --config configs/minimal_annotations.yaml \
    --sites Halfmile --ablation combined --seed 0
python train/fbp_self_train.py --smoke
```

`--smoke` is Halfmile, 8 labelled gathers, 1 epoch, 2 outer steps (one QC draw).

`--ablation` is `control` | `windowed` | `weighted` | `combined` | `iterative`.
Static ablations run 25 epochs. `iterative` is 15 × 5 epochs with offset-bin
2σ QC (20 bins, keep gathers with ≥85% surviving picks), 200 unlabeled gathers
per step, and weight resets after iterations 5 and 10.

Recipe defaults: `--model meneses` (4-scale 64→512 U-Net, BN + LeakyReLU 0.01,
scratch), FB-window vs background (`picker: fbpunet`), WBCE weight 100, no
augmentation, per-trace z-score then int16 quantize, batch pad to a multiple of
16 with amplitude 1. Window width is ±10 ms (5 samples at 2 ms, 10 at Lalor 1 ms).
Picks are the per-trace argmax of the FB logit; a trace is unpicked if background
wins everywhere.

Headline metrics on the 99% pool match `fbp_eval.py`: HitRate, MAE/RMSE, coverage,
\(W_{pred}(x)\) / \(W_{total}(x)\) for \(x \in \{0,2,5,10\}\), and MAE on predicted
traces. \(W_{total}(10)\) is the paper’s end-to-end number. After the last fit the
run writes `report/index.html` (worst + typical gather galleries, Plotly stats,
`metrics.json`, trace table) next to `split.json`. Reuse a split with `--split-json`.

## Regression validation

Run from the repository root in the `seismic_activity` environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

Plugin autoload is disabled to avoid unrelated ROS pytest plugins on the local
machine. Tests cover HDF5 cache ownership, cropping, persistent workers, sanity
validation, multi-epoch checkpoint/fresh-eval parity, scheduler resume, CLI reports,
artifact selection, and two-process CPU/Gloo validation with uneven shards.
