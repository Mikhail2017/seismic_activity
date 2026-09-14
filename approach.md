# Cross-site first-break picking: approach and evaluation

This report describes the production first-break picking pipeline used on the Hardpicks land-reflection surveys. The task is to recover, for every trace in a shot × receiver-line gather, the sample index of the first arriving energy. The operational goal is **cross-site generalization**: a model trained on three surveys must pick correctly on a completely unseen fourth survey.

The method follows the Hardpicks U-Net baseline of St-Charles et al. [1, 2], recasts picking as before/after segmentation, and adds geometry-conditioned normalization (GeoNorm) adapted from GeoFormer [3]. Training converts logits to picks with the **legacy** before/after smoother (window 50 samples) and selects the checkpoint on held-out HR@1. Evaluation then applies lateral cleaning; the leave-one-site-out table in section 10 was additionally decoded with `change_point`. The classical baseline is STA-LTA-OS [4].

---

## 1. Hardpicks seismic data format

The public Hardpicks benchmark [1, 2] contains 3D land reflection surveys acquired with explosive sources and parallel geophone lines over crystalline hard-rock mining sites in Canada. Each survey is stored as a SEG-Y-derived HDF5 file with traces under `TRACE_DATA/DEFAULT`.

| Field | Role |
| --- | --- |
| `data_array` | Raw 32-bit trace amplitudes, one row per receiver |
| `SHOTID` / `SHOT_PEG` | Shot identity |
| `REC_PEG` | Receiver peg; the receiver-line ID is `REC_PEG // 10^d` (`d = 3` except Halfmile, `d = 4`) |
| `CHANNEL` | Acquisition order along the line |
| `REC_X`, `REC_Y` | Receiver coordinates (scaled by `abs(COORD_SCALE)` when present) |
| `OFFSET` | Shot–receiver offset |
| `SAMP_RATE` | Sampling interval in microseconds |
| First-break field | Manual pick time in milliseconds (`SPARE1` except Lalor, which uses `SPARE2`) |

Unlabeled picks are stored as `0` or `-1` and mapped to NaN. A **line gather** — one shot intersected with one receiver line, traces ordered by `CHANNEL` — is the training example. It is treated as a 2D image: **vertical axis = time samples**, **horizontal axis = receivers along the line**. Full 3D shot gathers are *not* fed to the network as a single image; each parallel receiver line is a separate example.

The four surveys used here (Kevitsa / Matagami from the five-site Geophysics paper [2] is not in this dataset):

| Site | HDF5 file | Sampling | Trace length | Line gathers (eval) | Traces | Labeled traces | Label coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brunswick | `Brunswick_orig_1500ms_V2.hdf5` | 2 ms | 751 | 18,443 | 4,490,714 | 3,731,549 | 83.1% |
| Halfmile | `Halfmile3D_add_geom_sorted.hdf5` | 2 ms | 751 | 5,496 | 1,094,863 | 992,919 | 90.7% |
| Lalor | `Lalor_raw_z_1500ms_norp_geom_v3.hdf5` | **1 ms** | 1001 | 12,040 | 2,027,587 | 1,119,014 | 55.2% |
| Sudbury | `preprocessed_Sudbury3D.hdf` | 2 ms | 1001 | 4,313 | 711,155 | 200,338 | 28.2% |

Across the four sites this is about 40k line gathers and 8.3M traces. Annotations in Hardpicks cover 72.3% of traces in gathers judged valid by an expert [1]; they start from classical picker suggestions and are then manually corrected. Lalor’s original picks followed an onset convention and were shifted to the trough convention used on the other sites [2].

---

## 2. Task, annotation classes, and loss

First-break picking is cast as **binary semantic segmentation** of the gather, not as detecting a one-pixel “first-break” class. For trace $r$ with expert pick sample $b_r$:

$$
y_{k,r} = 0 \text{ if } k < b_r \text{ (before the first arrival)}, \quad y_{k,r} = 1 \text{ if } k \ge b_r \text{ (at or after the first arrival)}.
$$

The pick is recovered later as the **before → after change point** along time. This avoids the extreme imbalance of a single positive sample per trace [1, 2]. The first-break buffer used when building the mask is `0`: the class boundary sits exactly on the annotated sample.

A third label is used only as a mask, never as a class the model must predict:

- **`ignore_index = -1`** (`DONTCARE_SEGM_MASK_LABEL`): unlabeled traces, dummy padded receivers, and time samples dropped by a training crop. These pixels are excluded from the loss and from pick-error metrics.

The training objective is **pixel-wise multiclass cross-entropy** (`torch.nn.CrossEntropyLoss`) over the two real classes, with `ignore_index=-1`. For valid pixels $\Omega$:

$$
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{|\Omega|} \sum_{i \in \Omega} \sum_{c \in \{0,1\}} y_{i,c} \log p_{i,c}.
$$

Raw logits are passed to the loss (no extra softmax). There is no class weighting and no Dice term in the production recipe.

---

## 3. Metrics

All pick-error metrics are computed **only on traces with a valid reference annotation**. Units are **samples** (pixels along the time axis) unless a millisecond conversion is stated. Signed error is $e = \hat{b} - b$ (prediction minus reference): **negative = early**, **positive = late**.

Hit rate uses a **strict** inequality, matching both Hardpicks [1] and the evaluator:

$$
\mathrm{HR@}\delta = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\big[|e_i| < \delta\big], \qquad \delta \in \{1,3,5,7,9\}.
$$

For integer picks, **HR@1 is an exact sample match**. Additional scalars:

| Metric | Definition | Desired |
| --- | --- | --- |
| MAE | mean of $\lvert e\rvert$ | lower |
| RMSE | $\sqrt{\mathrm{mean}(e^2)}$ | lower |
| P90 / P95 | 90th / 95th percentile of $\lvert e\rvert$ | lower |
| Median AE | median of $\lvert e\rvert$ | lower |
| MBE | mean of $e$ | near zero |
| GatherCoverage | fraction of eligible traces that receive a pick | high, without dropping hard traces |

Millisecond MAE/MBE use each site’s sampling interval (1 ms at Lalor, 2 ms elsewhere). Checkpoint selection and early stopping use **`valid/HitRate1px` on the held-out site**, not training loss.

---

## 4. Data preparation for training

Training and evaluation read the four HDF5 files directly (`backend: hdf5`). Metadata handling is the repeatable `owned-hdf5-metadata-v1` path: transforms must not rewrite cached raw offsets or labels across accesses.

**Per gather, before the model sees it:**

1. Load the shot × receiver-line gather and its first-break times.
2. Convert pick times to sample indices; unlabeled traces stay at the ignore label.
3. Build the 2-class segmentation mask from those indices (section 2).
4. Provide geometry: shot–receiver offset and neighbour-receiver distances (`provide_offset_dists`), plus the GeoNorm $(\delta x, \delta z)$ vectors (section 7).
5. **Per-trace absolute-maximum normalization** of amplitudes,

$$
\tilde{x}_{k,r} = \frac{x_{k,r}}{\max(\max_j |x_{j,r}|, \varepsilon)},
$$

so each live trace lies in approximately $[-1, 1]$. Dead traces remain near zero.

6. **Training only:** apply the augmentations in section 8, in that order, *before* normalization and mask generation. Validation is never augmented.
7. **Batch collation** pads gathers in a batch to a common power-of-two $(T, L)$. Dummy receivers and padded time samples are ignore-labeled and are excluded from loss and metrics.

GeoNorm min–max statistics for $\delta x$ and $\delta z$ are computed on **training traces of that fold only** and written next to the checkpoint (`geom_stats.yaml`). Validation and later evaluation reuse the same stats; they are not recomputed on the held-out site.

---

## 5. Model architecture

The network is the local **FBPUNet**: a fully convolutional U-Net whose encoder is an ImageNet-initialized **ResNet18** (five encoder stages) and whose decoder is the vanilla Hardpicks decoder with channels **`[256, 128, 64, 32, 16]`**. Because it is fully convolutional, it accepts variable gather sizes (after power-of-two padding).

**Input.** A batched tensor of shape $(B, C, L, T)$ — **height = receivers along the line**, **width = time samples** (the opposite of how gathers are plotted). Under configuration D this is **six channels**:

- Channel 0: normalized waveform.
- Channels 1–3: Hardpicks `offset_distances` (shot–receiver offset and neighbour-receiver distances), each **constant along time**.
- Channels 4–5: GeoNorm $\delta x$ and $\delta z$ maps, also constant along time.

First-break prior masks are off.

**Encoder.** SMP ResNet18, `depth=5`, ImageNet weights, first convolution adapted to $C$ seismic channels rather than RGB. Skip feature maps are kept at each scale.

**Decoder.** Five blocks. Each block:

1. Learned $2 \times 2$ transposed convolution (stride 2) to upsample.
2. Concatenate the matching encoder skip (U-Net skip connection).
3. Two $3 \times 3$ convolutions, each followed by normalization and ReLU.

Decoder attention is off. There is no extra mid-block (`mid_block_channels: 0`); the deepest encoder map goes straight into the decoder.

**Head.** A $1 \times 1$ convolution produces **two logits per pixel** (before / after). Softmax is applied only when converting logits to probabilities.

**GeoNorm.** Under configuration D, every 2D Batch/Instance/affine GroupNorm in the encoder and decoder is replaced by geometry-conditioned GeoNorm (section 7), so scale and shift vary across receivers and are constant along time.

**Optimization (recipe).** Adam, learning rate $0.002136$ (ResNet18 preset; `lr: null` in the recipe), weight decay $10^{-6}$, batch size 16 gathers, 32-bit precision. StepLR drops the learning rate by $10\times$ after 10 epochs ($0.0002136$). Maximum 20 epochs; early stopping if validation HR@1 does not improve for 4 checks; a single best checkpoint is kept.

---

## 6. Site-wise folds (cross-site generalization)

St-Charles et al. argue that a random shuffle of gathers from all surveys is the wrong split for this problem [1]:

> shuffling and splitting across the gathers of all surveys may lead to misleading results if the trained models manage to overfit to the spatial or subsurface characteristics of particular regions.

The entire purpose of the benchmark — and of this pipeline — is to test whether a model trained on sites A and B can pick on a completely unseen site C.

The original Hardpicks folds used five sites and a train / valid / test triple, including Kevitsa. This dataset has the four public surveys only, so the production split is **leave-one-site-out** on those four: three whole surveys for training, the remaining whole survey for validation and checkpoint selection. There is no extra untouched test set and no intra-site `eval_ratio` split. Results are reported **per fold**.

| Fold | Train | Held-out validation | Train gathers | Valid gathers | Best epoch |
| --- | --- | --- | ---: | ---: | ---: |
| A | Lalor, Brunswick, Sudbury | **Halfmile** | ~34.8k | 5,496 | 11 |
| B | Lalor, Brunswick, Halfmile | **Sudbury** | 35,979 | 4,313 | 19 |
| C | Halfmile, Lalor, Sudbury | **Brunswick** | 21,849 | 18,443 | 19 |
| D | Sudbury, Halfmile, Brunswick | **Lalor** | 28,252 | 12,040 | 13 |

Fold A matches Hardpicks fold A with Kevitsa dropped. B–D are the same rotation so that each site is held out once. Fold D (Lalor held out) is the hardest transfer: Lalor is the only 1 ms survey, so a model trained only on 2 ms data must generalize across sampling rate as well as geology and acquisition [1, 2].

---

## 7. GeoNorm

GeoFormer conditions a transformer on per-trace acquisition geometry [3]. The same idea is applied here to the U-Net: **feature statistics should depend on offset and elevation**, not only on a global norm that mixes near and far traces.

For each trace $i$,

$$
\delta x_i = \sqrt{(s_x - r_x)^2 + (s_y - r_y)^2}, \qquad \delta z_i = z_r - z_s.
$$

Both coordinates are min–max scaled to $[0,1]$ with the **training-site** statistics of that fold (example, fold A: $\delta x \in [0.10, 7188]$ m, $\delta z \in [-104, 112]$ m over 7.23M training traces). Those stats travel with the checkpoint and must not be swapped across folds.

A shared MLP maps $\mathbf{g}_i = (\delta x_i, \delta z_i)$ to a 256-d embedding (Linear–GELU–Linear–GELU, $2 \to 256 \to 256$). That embedding is computed once per forward pass.

**GeoNorm** replaces standard affine normalization. Feature maps are $\mathbf{x} \in \mathbb{R}^{B \times C \times L \times T}$ (traces as height, time as width):

1. GroupNorm **without** affine parameters (preferred `groups=8`, reduced if $C$ is not divisible by 8).
2. A linear layer predicts per-trace scale $\boldsymbol{\gamma}_i$ and shift $\boldsymbol{\beta}_i$ from the geometry embedding.
3. If the trace axis has been downsampled ($L_{\mathrm{feat}} \neq L$), $\boldsymbol{\gamma}, \boldsymbol{\beta}$ are resampled with adaptive average pooling — never cropped.
4. Modulation is broadcast **down time, across channels, varying across traces**:

$$
\mathrm{GeoNorm}(\mathbf{x})_{:,:,i,:} = (1 + \boldsymbol{\gamma}_i) \odot \bar{\mathbf{x}}_{:,:,i,:} + \boldsymbol{\beta}_i.
$$

The conditioning linear layer is **zero-initialized**, so GeoNorm starts as plain GroupNorm.

**Configuration D** (the production setting) turns **both** mechanisms on:

| | Geometry as extra input channels | GeoNorm in every block |
| --- | :---: | :---: |
| A (baseline) | | |
| B | yes | |
| C | | yes |
| **D (used)** | **yes** | **yes** |

The extra input channels are an offset map and an elevation map, each constant along time. Combined with the waveform and the Hardpicks offset-distance channels, they give the first convolution direct access to geometry, while GeoNorm continues to modulate deeper features. Geometry features are attached **after** training augmentations from the (already reordered) shot/receiver coordinates, so flip and drop/pad stay aligned with waveform and labels.

---

## 8. Data augmentation

Augmentations run **only on training gathers**, in the order below, *before* amplitude normalization and mask generation. Waveform, labels, and shot/receiver coordinates stay aligned; GeoNorm features are computed afterwards from those coordinates.

| Operation | Parameters | Effect |
| --- | --- | --- |
| Time crop | `low_sample_count: 512`, `high_sample_count: 1024`, `max_crop_fraction: 0.333` | Keep the record start; drop a bounded tail. Labels outside the crop are ignored **for that sample only**. |
| Kill traces | `prob: 0.08` | Independently zero each trace’s amplitudes. Labels are **kept** (`invalidate_labels` is off). Was off, because of negative impact on results |
| Drop and pad | `target_trace_counts: [64, 128, 256, 512]`, `full_snap: true`, `max_drop_ratio: 0.50` | Snap receiver-line length to a target: drop traces (prefer unlabeled, then edges) or pad with dummy receivers. If reaching the nearest smaller target would drop more than half the line, pad up to the next larger target instead. Was off, because of negative impact on results |
| Receiver flip | probability $0.5$ | Reverse receiver order, with labels and coordinates reversed the same way. |

Validation and evaluation apply none of these.

---

## 9. Evaluation post-processing

Training-time checkpoint selection converts logits to picks with the decoder stored in the checkpoint. With the production recipe that decoder is **`legacy`** (window 50 samples). **Final evaluation** reuses those weights and that decoder, then applies lateral cleaning. It does not re-select the checkpoint.

### 9.1 Pick decoder (legacy)

Each trace’s predicted class map (argmax over the two logits) is scanned with the Hardpicks-style smoother `fb_smooth_result`. Consecutive after-class (`1`) onsets are compared by summing a 50-sample window; the pick is the first onset whose window sum matches the next. An empty or unstable run is left unpicked (`0`). This rule does not require a contiguous after-class run, and is kept as the training default for checkpoint compatibility.

The alternative **`change_point`** decoder is an **evaluation-only override**: it scores every before→after boundary $k$ as $C(k) = -\sum_{t<k}\log P_{\mathrm{before}}(t) - \sum_{t\ge k}\log P_{\mathrm{after}}(t)$ (equivalently a prefix sum of log-odds) and takes the earliest interior minimum that strictly beats both endpoints. The comparison table in section 10 was scored with this override. Training never used it for checkpoint selection.

### 9.2 Lateral cleaning

Picks are then cleaned **across receivers** (not in time, and not by filling labels). Isolated traces that disagree with their neighbours are replaced. Production settings:

| Setting | Value |
| --- | --- |
| Lateral window | 15 traces |
| Maximum deviation | 15.0 samples |
| Maximum flagged fraction | 0.3 |
| Minimum anchors | 3 |

If too large a fraction of a gather would be flagged, or too few reliable neighbours exist, the gather is left unchanged. `n_replaced` counts substituted **predictions**, not proven label errors.

On Brunswick (fold C) this matters a lot: raw-evaluator RMSE is 40.9 samples; after lateral cleaning it is 11.5, while HR@1 barely moves (0.807 → 0.805). Headline tables always use the **cleaned** metrics.

Dummy and unlabeled traces remain excluded from scoring.

---

## 10. Results: learned model vs STA-LTA-OS

The learned model is ResNet18 + GeoNorm D. Training selected checkpoints with the **legacy** smoother. The numbers below are those checkpoints evaluated with **`change_point` + lateral cleaning** on each held-out site. The classical baseline is adaptive **STA-LTA with outlier statistics** (STA-LTA-OS) of Jones and van der Baan [4], implemented in `seismic_utils/sta_lta.py` and scored on the same four full sites.

STA-LTA-OS fits a two-state EM on the Hilbert envelope of each live trace (pre-break = null, post-break = outlier), then places the onset with a short-window picker. Production parameters match the paper defaults scaled to these sampling rates: **Th = 1.3**, **Lw = 0.50 s**, **Sw = 0.05 s**. Dead traces are left unpicked. No neural checkpoint is used.

### 10.1 Headline comparison (validation site of each fold)

| Held-out site | Fold | Method | HR@1 | HR@3 | HR@5 | MAE | RMSE | P90 | MBE | Coverage | Traces scored |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Halfmile | A | **U-Net + GeoNorm D** | **0.706** | **0.917** | **0.955** | **0.81** | **4.31** | **2** | 0.18 | **0.999** | 992,919 |
| Halfmile | — | STA-LTA-OS | 0.060 | 0.259 | 0.418 | 77.03 | 130.66 | 245 | −63.5 | 0.637 | 992,919 |
| Sudbury | B | **U-Net + GeoNorm D** | **0.696** | **0.951** | **0.973** | **0.82** | **5.30** | **1** | 0.50 | **1.000** | 200,338 |
| Sudbury | — | STA-LTA-OS | 0.096 | 0.312 | 0.442 | 33.47 | 60.92 | 107 | −4.79 | 0.545 | 200,338 |
| Brunswick | C | **U-Net + GeoNorm D** | **0.805** | **0.947** | **0.961** | **1.11** | **11.46** | **1** | −0.26 | **0.993** | 3,731,549 |
| Brunswick | — | STA-LTA-OS | 0.085 | 0.360 | 0.554 | 70.97 | 149.74 | 297 | −61.1 | 0.681 | 3,731,549 |
| Lalor | D | **U-Net + GeoNorm D** | **0.576** | **0.761** | **0.812** | **2.63** | **8.20** | **8** | −0.42 | **1.000** | 1,119,014 |
| Lalor | — | STA-LTA-OS | 0.014 | 0.074 | 0.130 | 61.02 | 125.36 | 245 | 8.28 | 0.725 | 1,119,014 |

Training-time HR@1 on the same held-out site (checkpoint selection, no lateral cleaning) was 0.709 / 0.697 / 0.808 / 0.576 for folds A–D. Cleaned eval HR@1 is essentially the same; the decoder/cleaning gap is small except for RMSE on Brunswick, where lateral cleaning removes heavy tails.

**Reading the table.** The U-Net is an order of magnitude more accurate than STA-LTA-OS on every site: HR@1 rises from a few percent to 58–80%, MAE falls from tens of samples to about 1–3, and coverage is essentially complete. STA-LTA-OS is strongly early on Halfmile and Brunswick (MBE −63 and −61 samples) and leaves 27–46% of traces unpicked. The learned model’s residual difficulty is **Lalor** (fold D): HR@1 = 0.576, P90 = 8 samples, consistent with transferring from 2 ms training data onto a 1 ms survey [1]. Even there, HR@5 is 0.81 versus 0.13 for STA-LTA-OS.

### 10.2 Offset behaviour (learned model)

Errors grow with offset, as expected for weaker far-offset first arrivals, but stay small on three of four sites.

| Site (fold) | Near-offset HR@1 | Far-offset HR@1 | Near MAE | Far MAE |
| --- | ---: | ---: | ---: | ---: |
| Halfmile (A) | 0.82 | 0.50 | 0.59 | 1.53 |
| Sudbury (B) | 0.71 | 0.70 | 0.64 | 0.95 |
| Brunswick (C) | 0.92 | 0.43 | 0.13 | 6.77 |
| Lalor (D) | 0.69 | 0.41 | 1.27 | 4.42 |

Brunswick’s far-offset bin (mid-offset ~1.9 km) is the main source of its RMSE; lateral cleaning reduces but does not eliminate those outliers. Sudbury stays flat across offset, with the caveat that only 28% of traces are labeled.

### 10.3 Typical vs worst gathers

Gallery selection ranks gathers by P90 absolute error (worst) or by median MAE among the remainder (typical).
**Typical learned-model gathers** (median error, representative of deployed behaviour):

| Site | Gather | Shot | MAE (samples) | RMSE | HR@1 | Labeled traces | Figure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Halfmile | 4533 | 20261246 | 0.61 | 1.85 | 0.72 | 179 | [png](images/typical_halfmile_g4533_s20261246.png) |
| Halfmile | 2891 | 20181317 | 0.61 | 1.92 | 0.83 | 179 | [png](images/typical_halfmile_g2891_s20181317.png) |
| Sudbury | 5609 | 598 | 0.44 | 0.96 | 0.72 | 50 | [png](images/typical_sudbury_g5609_s598.png) |
| Brunswick | 12956 | 211081 | 0.31 | 1.44 | 0.86 | 235 | [png](images/typical_brunswick_g12956_s211081.png) |
| Brunswick | 16765 | 271072 | 0.31 | 0.92 | 0.82 | 235 | [png](images/typical_brunswick_g16765_s271072.png) |
| Lalor | 13428 | 246166 | 1.99 | 3.92 | 0.54 | 103 | [png](images/typical_lalor_g13428_s246166.png) |

![Halfmile typical gather 4533](images/typical_halfmile_g4533_s20261246.png)

![Halfmile typical gather 2891](images/typical_halfmile_g2891_s20181317.png)

![Sudbury typical gather 5609](images/typical_sudbury_g5609_s598.png)

![Brunswick typical gather 12956](images/typical_brunswick_g12956_s211081.png)

![Brunswick typical gather 16765](images/typical_brunswick_g16765_s271072.png)

![Lalor typical gather 13428](images/typical_lalor_g13428_s246166.png)

On Halfmile, Sudbury, and Brunswick a typical gather is sub-sample to ~0.6-sample MAE with HR@1 around 0.7–0.86. A typical Lalor gather is worse (~2 samples MAE) but still on a different scale from STA-LTA.

**Worst learned-model gathers** (heavy tails, not the mean):

| Site | Gather | Shot | MAE | RMSE | HR@1 | Labeled traces | Likely issue | Figure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Halfmile | 2375 | 20141438 | 23.6 | 38.1 | 0.24 | 55 | Sparse labels, large residual streak | [png](images/worst_halfmile_g2375_s20141438.png) |
| Sudbury | 10705 | 1063 | 134.2 | 137.2 | 0.00 | 23 | Almost unlabeled line; systematic miss | [png](images/worst_sudbury_g10705_s1063.png) |
| Sudbury | 4546 | 505 | 104.7 | 105.0 | 0.00 | 21 | Same pattern | [png](images/worst_sudbury_g4546_s505.png) |
| Brunswick | 585 | 11053 | 151.5 | 300.4 | 0.31 | 132 | Far-offset / noisy gather; drives RMSE | [png](images/worst_brunswick_g585_s11053.png) |
| Brunswick | 807 | 11078 | 136.7 | 229.8 | 0.00 | 278 | Complete miss on a well-labeled gather | [png](images/worst_brunswick_g807_s11078.png) |
| Lalor | 478 | 202130 | 101.8 | 161.2 | 0.40 | 25 | Sparse labels + large bias | [png](images/worst_lalor_g478_s202130.png) |
| Lalor | 1585 | 206127 | 80.2 | 157.0 | 0.39 | 109 | Cluster of failures | [png](images/worst_lalor_g1585_s206127.png) |

![Halfmile worst gather 2375](images/worst_halfmile_g2375_s20141438.png)

![Sudbury worst gather 10705](images/worst_sudbury_g10705_s1063.png)

![Sudbury worst gather 4546](images/worst_sudbury_g4546_s505.png)

![Brunswick worst gather 585](images/worst_brunswick_g585_s11053.png)

![Brunswick worst gather 807](images/worst_brunswick_g807_s11078.png)

![Lalor worst gather 478](images/worst_lalor_g478_s202130.png)

![Lalor worst gather 1585](images/worst_lalor_g1585_s206127.png)

Worst cases concentrate on (i) gathers with very few labeled traces, where a handful of bad picks dominate MAE, and (ii) a small number of Brunswick/Lalor shots where the arrival is weak or the hyperbola is broken. Lateral cleaning replaced 13k / 5k / 123k / 107k traces on folds A–D; the Brunswick and Lalor counts show that post-processing is doing real work on those sites, but it cannot invent a first break that the network never saw and it's a general issue with such validation, because gathers on different sites could be unique at the very level.

**STA-LTA-OS on the same kind of example** is not in the same regime. A *typical* Brunswick STA-LTA gather already has MAE ≈ 57 samples and HR@1 ≈ 0.04–0.09 ([g3463 shot 51106](images/sta_lta_typical_brunswick_g3463_s51106.png), [g934 shot 11094](images/sta_lta_typical_brunswick_g934_s11094.png)). A *worst* Brunswick STA-LTA gather has MAE 150–340 samples ([g585 shot 11053](images/sta_lta_worst_brunswick_g585_s11053.png), MAE 342; [g18199 shot 291091](images/sta_lta_worst_brunswick_g18199_s291091.png), MAE 293). Gather 585 is a worst example for **both** methods; the U-Net MAE there is 152 versus 342 for STA-LTA, and the U-Net still picks about a third of traces to within one sample.

![STA-LTA typical Brunswick gather 3463](images/sta_lta_typical_brunswick_g3463_s51106.png)

![STA-LTA typical Brunswick gather 934](images/sta_lta_typical_brunswick_g934_s11094.png)

![STA-LTA worst Brunswick gather 585](images/sta_lta_worst_brunswick_g585_s11053.png)

![STA-LTA worst Brunswick gather 18199](images/sta_lta_worst_brunswick_g18199_s291091.png)

---

## 11. Summary

The production pipeline is a ResNet18 U-Net trained as before/after segmentation with GeoNorm D, Hardpicks-style site folds, crop / kill / drop-and-pad / flip augmentations, a **legacy** pick decoder at training time, and lateral cleaning at evaluation. The reported leave-one-site-out numbers additionally decode with `change_point`. On every held-out Hardpicks survey the learned model outperforms STA-LTA-OS by a wide margin in hit rate, error, and coverage. Remaining errors are concentrated at far offset, on Lalor’s 1 ms sampling, and on a thin tail of pathological gathers — which is the correct failure mode for a cross-site test, rather than an in-survey shuffle that can leak local geology into both train and test [1].
Unfortunately produced metrics are still far from desired  values in top articles and approach should  be improved.

---

## References

1. P.-L. St-Charles, B. Rousseau, J. Ghosn, J.-P. Nantel, G. Bellefleur, and E. Schetselaar, “A multi-survey dataset and benchmark for first break picking in hard rock seismic exploration,” in *Fourth Workshop on Machine Learning and the Physical Sciences (NeurIPS)*, 2021.

2. P.-L. St-Charles, B. Rousseau, J. Ghosn, G. Bellefleur, and E. Schetselaar, “A deep learning benchmark for first break detection from hardrock seismic reflection data,” *Geophysics*, vol. 89, no. 1, pp. WA279–WA294, 2024, doi: [10.1190/geo2022-0741.1](https://doi.org/10.1190/geo2022-0741.1).

3. T. Gao and J. Ma, “GeoFormer: Geometry-aware transformer and its application to 5D first-arrival picking,” arXiv:2608.25668, 2026.

4. N. A. Jones and M. van der Baan, “Adaptive STA–LTA with outlier statistics,” *Bulletin of the Seismological Society of America*, 2015. (STA-LTA-OS; Th = 1.3, Lw = 0.50 s, Sw = 0.05 s).
