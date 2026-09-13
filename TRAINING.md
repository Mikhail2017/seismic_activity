`--model` presets: `resnet18` (default), `resnet34`, `resnet34-horizon`,
`resnet50`, `efficientnet-b0`, `efficientnet-b4`, `vanilla`, or any SMP encoder
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

## Opt-in single-transition before/after decoder (no retraining required)

`--before-after-decoder legacy|change_point` selects how two-class logits become
picks; `--picker before_after` still selects the training task. The default for
old checkpoints with no decoder metadata is **legacy**, unchanged.

`change_point` scores all boundaries using the before probability before the
boundary and the after probability from the boundary onward. It uses float32
log probabilities and cumulative sums, excludes time padding, and does not use
labels. An interior boundary must strictly beat both all-before and all-after
explanations. Otherwise the result is `0` (no pick), with a NaN probability.
Tied interior minima select the earliest boundary. Non-finite real logits
invalidate that trace. After probability at a returned pick is not calibrated
confidence in the boundary time.

Compare the same weights and batch size without lateral cleaning first:

```bash
python /home/mika/dev/seismic_activity/train/fbp_eval.py \
  --ckpt-dir /home/mika/dev/seismic_activity/fold_results/train_fold_A \
  --fold A --backend hdf5 --data-dir /tmp/data \
  --before-after-decoder legacy

python /home/mika/dev/seismic_activity/train/fbp_eval.py \
  --ckpt-dir /home/mika/dev/seismic_activity/fold_results/train_fold_A \
  --fold A --backend hdf5 --data-dir /tmp/data \
  --before-after-decoder change_point
```

Use the data directory on your machine. Evaluation overrides do not rewrite the
checkpoint. `metrics.json` and `report.md` record the active decoder;
`smooth_threshold` is null in change-point evaluation reports because
`--smooth-threshold` applies **only to legacy**. Then repeat with identical
`--lateral-clean` settings if desired. Compare missing-pick rate as well as
MAE/RMSE and large-error tails; reduced coverage is not an accuracy improvement.
Fresh inference is required unless full logits were saved; per-trace pick tables
cannot be re-decoded.

For future training, the same CLI flag or YAML `before_after_decoder: change_point`
is saved in model/checkpoint hyperparameters and used during validation. Model
config overrides recipe YAML; CLI overrides both. Exact resume rejects a decoder
change (old missing metadata equals legacy); weights-only initialization permits
it. This changes validation/checkpoint selection, not cross-entropy gradients.

The Python prediction APIs `predict_first_breaks_ms` and
`predict_first_breaks_ms_from_shot_gather` accept
`before_after_decoder="change_point"`; omitted means the checkpoint's decoder.
The viewer uses the saved decoder through these APIs (no new UI override).

This decoder can reject small false early after-regions, but cannot reliably fix
a network that confidently supports a wrong sustained early boundary. No
lateral smoothing, persistence-window rule, or new confidence threshold is
introduced by this option.

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

## Geometry-conditioned U-Net (GeoNorm)

`geonorm.md` maps per-trace offset/elevation into the existing FBPUNet. Opt-in
via recipe `geonorm:` or `--geonorm A|B|C|D` (default **A**, unchanged baseline).

| Ablation | Offset/elev input channels | GeoNorm |
| --- | --- | --- |
| A | no | no |
| B | yes | no |
| C | no | yes (every 2D norm → GroupNorm + per-trace scale/shift) |
| D | yes | yes |

`δx`/`δz` are min-max normalized on **training gathers only** and written next
to the checkpoint (`geom_stats.yaml`). Flip/drop/pad keep `geom_features`
aligned with traces. Eval restores the same stats:

```bash
python train/fbp_train.py --fold A --picker before_after --geonorm D
python train/fbp_eval.py --ckpt-dir output/train_foldA_resnet18-before-after-geomD_... --fold A
```

GeoNorm modulates the **trace** axis of `(B, C, traces, time)` — not time.
Existing `use_dist_offsets` channels stay independent of ablation B's two maps.

## Lateral pick cleaner (eval only, no retrain)

After decode, isolated pick crashes (line-end dives, single-trace jumps) are
flagged against a robust pick-vs-offset fit and replaced from neighboring
anchors. **Predictions are rewritten; labels are not.**
Do not enable hardpicks `auto_fill_missing_picks` on the valid parser.

Same checkpoint as a normal eval, plus `--lateral-clean`:

```bash
python train/fbp_eval.py --picker before_after \
  --ckpt output/train_foldA_resnet18-before-after-geomd_.../best-epoch=019-step=043500.ckpt \
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

## Regression validation

Run from the repository root in the `seismic_activity` environment:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests
```

Plugin autoload is disabled to avoid unrelated ROS pytest plugins on the local
machine. Tests cover HDF5 cache ownership, cropping, persistent workers, sanity
validation, multi-epoch checkpoint/fresh-eval parity, scheduler resume, CLI reports,
artifact selection, and two-process CPU/Gloo validation with uneven shards.
