`--model` presets: `resnet18` (default), `resnet34`, `resnet34-horizon`,
`resnet50`, `efficientnet-b0`, `efficientnet-b4`, `vanilla`, or any SMP encoder
name. `resnet34-horizon` is ResNet34 plus the linear-moveout first-break prior
channel (5th input). A `*-horizon` suffix on any preset/SMP encoder does the
same. Before/after masks: `--picker before_after`, YAML `picker: before_after`,
or `model: resnet34-before-after` (combinable with `-horizon`).
`--encoder-weights imagenet` (etc.) initializes the backbone; default is
train from scratch. `--model-config path.yaml` merges on top of the preset.

## Recipe YAML: time window and augmentations

`configs/train.yaml` (override with `--config`) owns loop knobs and preprocess.

**Linear time window** (`linear_time_window`) is a train **and** validation
preprocess, not an augmentation. It fits a two-slope first-arrival trend from
labeled picks (min-offset, max-left, max-right), shifts each trace so the trend
sits at `half_window_samples`, and crops to `2 * half_window_samples`. Unlabeled
gathers use `unlabeled_fallback: skip` or constant-velocity LMO. Predictions are
unshifted (`original_idx = warped_idx + sample_time_shift`) in `predict.py`.
Prefix `crop` chops from t=0 and fights this window — omit `crop` when the
window is enabled.

**Train-only augs** (validation never uses this list):

| `type` | Role |
|---|---|
| `crop` | Keep the start of the gather; drop late samples |
| `kill` | Zero traces; `invalidate_labels: true` marks them don't-care |
| `drop_and_pad` | Snap trace count; `drop_edges_next: false` keeps far offsets |
| `flip` | Reverse the receiver axis (p=0.5) |
| `rebalance_offsets` | Drop a fraction of near-offset traces (far traces kept) |
| `polarity` | Multiply random traces by −1 |
| `noise` / `resample_*` | Existing hardpicks ops |

Suggested order with far-offset emphasis: `rebalance_offsets` → `kill`/`polarity`
→ `drop_and_pad` (`drop_edges_next: false`) → `flip`.

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

## Regression validation

Run from the repository root in the `seismic_activity` environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

Plugin autoload is disabled to avoid unrelated ROS pytest plugins on the local
machine. Tests cover HDF5 cache ownership, cropping, persistent workers, sanity
validation, multi-epoch checkpoint/fresh-eval parity, scheduler resume, CLI reports,
artifact selection, and two-process CPU/Gloo validation with uneven shards.
