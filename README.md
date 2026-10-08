# lane-width-road-elevation

**基於車道寬度之單目前方道路起伏估測** — Lane-Width-Based Monocular Estimation of the Road Elevation Profile Ahead

Monocular road pitch estimation from a single camera. The pipeline runs five sequential stages over one image and returns a **continuous pitch(z) curve** — the road's pitch angle in degrees as a function of depth ahead of the camera — using only a pretrained semantic segmentation model and classical projective geometry. No depth sensor required.

> **On the name:** formerly `mono3D-two-plane-geo` (renamed 2026-10-06). That name came from the first model, an explicit near/far pair of planes, which WWH-9 replaced with a continuous pitch(z) curve — there is no longer a knee or a pair of planes. The new name keeps what has not changed since the start: the lane width is the ruler, and the road's elevation profile ahead is the output. PRs #1–#21, their branches and the Linear project still carry the old name.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Calibration and the Evidence Guards](#calibration-and-the-evidence-guards)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
  - [1. Single-Image Inference (with visualization)](#1-single-image-inference-with-visualization)
  - [2. Batch Inference on a Dataset](#2-batch-inference-on-a-dataset)
  - [3. Unit Tests](#3-unit-tests)
  - [4. OpenLane as a Second Dataset](#4-openlane-as-a-second-dataset)
- [Ground Truth](#ground-truth)
- [Configuration Reference](#configuration-reference)
- [Critical Conventions](#critical-conventions)

---

## How It Works

The pipeline lives in `libs/inference/pipeline.py::infer_one()`, which chains the five stages:

```
road_segmentation → line_segmentation → lane_segmentation → lane_fitting → pitch_estimation
```

| Stage | File | What it does |
|---|---|---|
| road_segmentation | `libs/inference/road_segmentation.py` | PIDNet-L semantic segmentation → binary road mask (Cityscapes class 0) |
| line_segmentation | `libs/inference/line_segmentation.py` | ELSED line detection with a perspective-aware length threshold (interpolated by segment mid-y) |
| lane_segmentation | `libs/inference/lane_segmentation.py` | Near-to-far continuity tracking: seed the innermost lines at the bottom, track upward band by band with a local line model, re-seed past junctions |
| lane_fitting | `libs/inference/lane_fitting.py` | Inner-chain extraction → sub-pixel edge refinement on the **unmasked** image → one continuous gap-bridged lane curve per side |
| pitch_estimation | `libs/inference/pitch_estimation.py` | Sample lane widths from the two curves, inverse-perspective them to depth, then a continuous pitch(z) |

**Geometry:**

```
depth  z    = f_x × w_real / w_px          # w_px = inner-edge-to-inner-edge pixel width
Y_3d        = -z × (y_pixel - cy) / f_y    # cy = image_height / 2
pitch(z)    = arctan( slope of Y_3d over a local z-window )
```

The default estimator is `windowed`: for each sampled depth it takes a Theil-Sen slope over a `±max(1 m, 0.15·z)` window, which makes the spatial resolution of the output explicit. `method: spline` switches to a global weighted `UnivariateSpline` with a Theil-Sen MAD prefilter.

**Callers.** `utils/batch_inference_road_lane_segmentation.py` calls `infer_one()` directly. `utils/inference_road_lane_segmentation.py` deliberately inlines the same stages so it can draw each intermediate product — if you change `pipeline.py`, that second copy needs the same change.

---

## Calibration and the Evidence Guards

`f_x`, `f_y` and `camera_height` are **required**. Under the flat-ground pinhole model every lateral threshold has an exact pixel form at row `y`, so association tolerance, slope gates and model memory are all *derived* rather than hand-tuned. Three evidence guards ride on the same projection:

| Guard | Where | What it rejects |
|---|---|---|
| Paint segment gate | `paint_evidence.filter_paint_segments`, before tracking | ELSED segments that are not a bright ridge of bounded width — shadow boundaries, kerbs. The tracker then re-seeds on the true painted line |
| Evidence-break truncation | `paint_evidence.truncate_at_evidence_break`, after refinement | The tail of a chain once the photometric "is this paint?" test fails for a sustained run — this is what crest occlusion looks like |
| Depth-continuity truncation | `lane_fitting.truncate_at_depth_jump` | A jump in paired-row depth that exceeds both a continuity gate and 1.5× the local-plane extrapolation. Real paint beyond a crest is a *disconnected* section and must never be joined to the near chain |

Their thresholds are module constants with the derivations in the docstrings, not config values.

**The tracker does not need the lane width** (since 2026-09-15). Every threshold in `lane_segmentation.py` is a lateral *distance*; `w_real` only ever converted "a fraction of a lane" into metres, so stages 1–3 now run on a road whose width they do not know. `paint_evidence` never needed it either — its one geometric quantity is the row depth `f_y·h/(y-cy)`, and the width it was handed was never read.

Three of the five lateral constants are gone, each on measured evidence. The ablation pushes one gate's distance to 1e6 m (its pixel window then exceeds the image, so it can reject nothing) and diffs the pitch-curve hashes over CARLA (120 frames) and OpenLane Tier A (168 frames / 16 segments):

| Constant | Outcome | Evidence |
|---|---|---|
| ROI corridor | deleted | 288/288 frames bit-identical without it |
| seed-window outer bound | deleted | 288/288 bit-identical, and not one of 764 seeds moved over 552 frames picked for a *dashed* ego-lane marking — the case it was meant to cover |
| cross-lane cap | now **measured** per frame | `_measure_lane_width_m` reads each side's lateral offset at its own seed row and sums them, so neither side is extrapolated; one side alone gives 2×. Reads 3.317 m on CARLA against a 3.3226 m GT-implied truth, and tracks OpenLane's real per-segment spread (2.77–3.84). Unmeasurable ⇒ cap off, never a guessed width |
| slope gate | kept | The one lateral scale that *cannot* be measured instead of assumed — it runs in `_segment_info`, before any line has been found |
| association tolerance | open | Its justification is unsettled: ELSED endpoint noise is a *pixel*-domain quantity (already covered by `_TOL_PX_FLOOR`), confusion with the next lane is a *metre*-domain one, and the two imply opposite scaling with depth |

> ⚠ **Why the deleted pair was inert is not the obvious reason.** It is *not* that innermost-first selection makes an outer bound redundant — the band loop returns at the first band holding any candidate, so a bound that empties a band does change where the seed comes from. Both were written as `px_max_at(3.25, y)`, i.e. against `z_min` with its ±15° grade slack, so in flat-road metres they admit `3.25·z_at/z_min`. The slack starts at 6 m: on CARLA that is 8.6 m of real lateral distance at z=10; on OpenLane, whose bottom image row already sees 6.85 m, the bound was never a 3.25 m barrier anywhere in the image.

> ⚠ **The failure they were meant to prevent is real, and predates their removal.** Of the 552 dashed frames, 28 of 512 (5.5 %) seed on the *adjacent* lane and measure a 5–12 m "lane" — identically with and without the bound, and 23 of the 28 already produce no pitch at all. The fix needs a bound derived from the flat-road depth rather than the grade-padded one (WWH-21); re-adding these two would not cover it.

> ⚠ **The slope gate is the cautionary one.** On CARLA it looks removable — 6 frames touched, range slightly *better* without it. On OpenLane, disabling it costs 2 whole frames and 14.65 m of median range. Those three Town03 routes have almost no junctions, and **a gate that rejects stop lines and crosswalks cannot be evaluated on roads that have none.**

> A hand-tuned fallback for un-calibrated cameras (`min_slope` / `lane_band_tolerance` / `roi_near`) was removed on 2026-08-27: every caller supplied the full calibration, so it was a second implementation nothing ran and no test covered. Its thresholds were fitted to one dataset anyway — a genuinely different camera needs them re-derived, which is what the geometry path does on its own.

---

## Project Structure

```
lane-width-road-elevation/
├── config/
│   ├── inference_road_lane_segmentation.yaml   # Main config for all inference
│   └── train_road_segmentation.yaml            # Config for the (legacy) Resnet101 training path
├── libs/
│   ├── inference/
│   │   ├── pipeline.py                         # infer_one(): the five stages
│   │   ├── geometry.py                         # CameraGeometry: the pinhole model shared by the stages
│   │   ├── road_segmentation.py                # PIDNet loader + masking
│   │   ├── line_segmentation.py                # ELSED line detection (perspective-aware)
│   │   ├── lane_segmentation.py                # Near-to-far continuity tracking
│   │   ├── paint_evidence.py                   # Photometric "is this actually paint?" guards
│   │   ├── lane_fitting.py                     # Inner chain → sub-pixel refine → lane curves
│   │   └── pitch_estimation.py                 # Width sampling → depth → continuous pitch(z)
│   ├── road_profile_gt.py                      # Measured road-profile ground truth
│   └── visualization/
│       ├── lane_visualization.py               # Masks, segments, lanes, curves, per-step dumps
│       ├── pitch_visualization.py              # Per-frame pitch / Y_3d profile plots
│       ├── profile_mae_visualization.py        # Per-frame MAE plot (batch)
│       └── route_profile_visualization.py      # Whole-route terrain profile plot (batch)
├── carla_module/
│   ├── get_carlaDataset.py                     # Dataset collector (images + measurements + road profile)
│   ├── pick_route.py                           # Top-down route picking tool
│   ├── project_lane_gt.py                      # Project lane GT into the image
│   └── verify_carla_geometry.py                # Calibration checks against the simulator
├── utils/
│   ├── env_setup.py                            # Must be called before any C extension import
│   ├── inference_road_lane_segmentation.py     # Single-image inference + all visualizations
│   └── batch_inference_road_lane_segmentation.py  # Batch inference → CSV + plots
├── tests/
│   ├── synthetic.py                            # Synthetic pinhole scenes with analytically-known answers
│   ├── test_geometry.py                        # The projection model
│   ├── test_pitch_estimation.py                # Metric stage against known grades
│   ├── test_paint_evidence.py                  # Paint guards' edge cases
│   └── test_depth_jump.py                      # Depth-continuity guard's edge cases
├── openlane_module/                            # OpenLane → this project's dataset format, + the tools
│   │                                           # that measure what that data can verify. See its README
│   ├── convert_openlane.py                     # Annotations → images/ + measurements.csv + road_profile.csv
│   ├── frame_tags.py                           # Per-frame variables: weather, case tags, curvature, GT width
│   ├── frame_check.py                          # Which coordinate frame the GT `xyz` is in
│   ├── gt_width_noise.py                       # Noise floor of the lane-width GT itself (~12 mm)
│   ├── measure_paint_inset.py                  # Measures the paint inset directly, instead of assuming it
│   └── nearfield_feasibility.py                # Whether the near-field anchor fits a high camera's image
├── debug/                                      # Diagnostic and prototype scripts (gitignored, not part of the pipeline)
├── scripts/
│   └── setup_elsed.py                          # Clone + patch the ELSED C++ extension
├── docs/papers/                                # Reference papers and design drawios
├── pidnet_models/                              # PIDNet model definitions
├── pidnet_pretrained_model/                    # Pretrained weights (not tracked in git)
│   └── PIDNet_L_Cityscapes_test.pt
├── outputs/                                    # Inference results, CSVs, plots
├── main.py                                     # Thin wrapper around single-image inference
├── .env.example                                # Template for machine-specific paths
└── pyproject.toml                              # Dependencies (managed by uv)
```

---

## Prerequisites

- Python **3.12** (exact version required)
- [`uv`](https://github.com/astral-sh/uv) as the package manager
- A C++ compiler (for building the ELSED extension):
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
  - **Windows**: Visual Studio Build Tools with MSVC
  - **Linux**: `gcc` / `g++`
- **Windows only**: A system-installed OpenCV SDK (not the PyPI wheel) — path set via `.env`

For CARLA data collection, additionally:
- A running CARLA simulator server (tested with 0.9.16)
- The matching CARLA Python `.whl` installed manually

---

## Installation

**Step 1 — Copy and fill in the environment file**

```bash
cp .env.example .env
```

Edit `.env` with your machine-specific paths. On macOS/Linux, you can leave the Windows-only fields empty:

```dotenv
# Windows only — paths to OpenCV DLLs and CMake config for building pyelsed
OPENCV_BIN_PATH=
OPENCV_DIR=

# Path to CARLA .whl (required only for CARLA work)
CARLA_WHL_PATH=
```

**Step 2 — Patch and prepare the ELSED C++ source**

This must be done before `uv sync`, because `pyproject.toml` references the local `elsed_src/` directory as the source for `pyelsed`.

```bash
python scripts/setup_elsed.py
```

This script clones the ELSED repo into `elsed_src/`, pins pybind11 to v2.13.6 (required for Python 3.12), and applies Windows/MSVC compatibility patches.

**Step 3 — Install all dependencies (this also compiles pyelsed)**

```bash
uv sync
```

**Step 4 — Place pretrained PIDNet weights**

Download `PIDNet_L_Cityscapes_test.pt` and place it at:

```
pidnet_pretrained_model/PIDNet_L_Cityscapes_test.pt
```

**Step 5 — (Optional) Install the CARLA Python package**

Only needed for data collection. Install the `.whl` that matches your CARLA server version:

```bash
uv pip install $CARLA_WHL_PATH
```

---

## Running the Project

> **Important:** All commands must be run from the **project root directory**. The config file and the relative paths in the code depend on this.

### 1. Single-Image Inference (with visualization)

Runs the five stages on one image and saves every intermediate product to `outputs/`.

```bash
python -m utils.inference_road_lane_segmentation
# python main.py is a thin wrapper around exactly this
```

Set the target image in `config/inference_road_lane_segmentation.yaml`:

```yaml
input:
  image_path: "inference_datasets/<dataset>/images/000500.png"
```

Output files (base name comes from `visualization.save_path`):

| File | Content |
|---|---|
| `result_overlay.png` | Road mask overlaid on the image |
| `result_line_segments.png` | All detected ELSED line segments |
| `result_lanes_seg.png` | The tracked inner left/right lane segments |
| `result_lane_fits.png` | The two continuous lane curves |
| `result_pitch_profile.png` | Predicted pitch(z) against GT |
| `result_y3d_profile.png` | Predicted road height Y_3d(z) against GT |
| `outputs/lane_fit_step/` | Per-step dump of the lane-fitting stage |
| `outputs/pitch_est_step/` | Per-step dump of the pitch-estimation stage |

Per-stage timing, the GT source, and the frame's pitch MAE are printed to stdout. GT is read from the image's own dataset, so no separate GT file has to be prepared.

---

### 2. Batch Inference on a Dataset

Runs inference on every frame in a dataset's `measurements.csv` and writes the results back out.

```bash
python -m utils.batch_inference_road_lane_segmentation
```

Set the paths in `config/inference_road_lane_segmentation.yaml`:

```yaml
input:
  image_batch_path: "inference_datasets/<dataset>/images"
csv_io:
  measurements_csv: "inference_datasets/<dataset>/measurements.csv"
  output_dir: "outputs/measurements.csv"
  problem_csv: "outputs/problem.csv"
  problem_mae_threshold: 2.0
```

Produces:

| Output | Content |
|---|---|
| `csv_io.output_dir` | The input CSV plus `z_visible_min`, `z_visible_max`, `profile_mae` per frame |
| `outputs/problem.csv` | Frames with `profile_mae` above `problem_mae_threshold`, worst first |
| `outputs/profile_mae_<dataset>.png` | Per-frame MAE over the route |
| `outputs/route_profile_<dataset>.png` | Whole-route terrain profile (analytic vs mesh height, their difference, per-frame MAE aligned to distance). Only when the dataset has a `road_profile.csv` |

The plot filenames carry the dataset name so that running several datasets back to back does not overwrite them; the CSV paths still follow the config.

Two flags override the config, so several datasets can be run back to back without editing it between runs (and without a second copy of the pipeline):

```bash
python -m utils.batch_inference_road_lane_segmentation --dataset inference_datasets/<dataset> --tag baseline
```

One dataset directory is one **continuous sequence**: the lane width measured on one frame is carried to the next ones only within it (see the near-field section below).

| Flag | Overrides |
|---|---|
| `--dataset <DIR>` | `input.image_batch_path` and `csv_io.measurements_csv` |
| `--tag <suffix>` | Output filename suffix, so back-to-back runs do not overwrite each other |

The output CSV records where each frame's lane width came from: `w_real_used`, `w_real_status` (`measured` / `held` / `last_resort` / `no_anchor`), `w_real_reason` (why this frame could not be measured itself) and `w_real_hold_m` / `w_real_hold_frames` (how old a held width is). The run ends with a tally of the statuses and lists the `last_resort` frames (pitch on the **assumed** width) and the `no_anchor` frames (no pitch).

Frames that produce no output are skipped and their ids printed. **A skipped frame is not necessarily a bug** — the method needs two inner lane edges, so intersections and unmarked crests legitimately give nothing, and the paint guards will abstain rather than measure a kerb.

---

### 3. Unit Tests

```bash
uv run --no-sync python -m pytest
```

> ⚠ Use `--no-sync`. A plain `uv run` (or `uv sync`, or `uv add`) re-resolves the environment and replaces the CUDA build of torch with the CPU wheel the lockfile pins. If that happens, restore it with:
> ```bash
> uv pip install "torch==2.10.0+cu126" "torchvision==0.25.0+cu126" --index-url https://download.pytorch.org/whl/cu126
> ```

The tests are pure geometry — no images, no weights, no GPU, under two seconds. They cover two things:

- **The projection model and the metric stage**, against synthetic pinhole scenes whose answer is known analytically (a road of constant grade must read back as that grade). `tests/synthetic.py` builds the scenes from the forward model the pipeline inverts.
- **The three evidence guards' edge cases**, each of which cost a full three-dataset sweep to find: the leading-drop exception, the run-length rule, the fixed bright-peak window, and the large-row-gap case that must *not* be read as a depth jump. The docstrings name the frames.

They do **not** measure accuracy. That is what the batch MAE sweep is for.

---

### 4. OpenLane as a Second Dataset

Every threshold in this pipeline was originally fitted on three CARLA routes through one Town03 map. `openlane_module/` converts [OpenLane](https://github.com/OpenDriveLab/OpenLane) (3D lane annotations over the Waymo Open Dataset) into the same three-file dataset format, so a gate can be tested on roads with junctions, real paint and a different camera mount. **The pipeline and the GT module are unchanged** — every compatibility problem is solved inside the converter.

```bash
python openlane_module/convert_openlane.py --openlane <DIR> --out <OUT> --no-images
python openlane_module/frame_tags.py        # per-frame variables -> CSV, for picking strata
```

That second camera is what exposed thresholds silently written against *this* mount — the near-field calibration window contained zero rows on a 2.1 m camera, and the two deleted lateral bounds above were never a barrier anywhere in its image. It also supplies the junction frames that showed the slope gate is *not* removable.

`openlane_module/README.md` has the per-file table, the three design decisions and the licence terms. ⚠ **CC BY-NC-SA + Waymo Non-Commercial: images and derived images must not be committed.**

---

## Ground Truth

GT is not a pipeline stage — it is the reference the runners score against, and it lives in its own `ground_truth` config section.

A collected dataset carries two files: `measurements.csv` (per frame: the camera transform, vehicle distance, body pitch) and `road_profile.csv` (the road surface ahead, in world coordinates). `libs/road_profile_gt.py` projects the profile into the camera frame with `v = P_world − cam_world`, `z_gt = v·forward`, `h_gt = v·up` — no offset constant, no arc-length conversion, and it never goes through the vehicle's attitude.

`road_profile.csv` has two height columns and `ground_truth.height_source` picks between them:

| Value | Column | Meaning |
|---|---|---|
| `auto` (default) | `z_mesh` if present, else `z` | Falls back for pre-WWH-14 datasets |
| `analytic` | `z` | The OpenDRIVE analytic centreline: smooth, without the implemented surface detail |
| `mesh` | `z_mesh` | A downward ray-cast onto the road mesh at collection time |

`mesh` is the better reference: the collector's road surface deviates from the analytic centreline in a few localised sections and the camera demonstrably sees those deviations. Asking for `mesh` on a dataset that lacks the column raises rather than silently falling back. The runners print `GT source:` so you can see which one was used.

> ⚠ Do not re-litigate the analytic/mesh choice with **absolute-height** MAE — it is dominated by a per-frame constant offset. The evidence is written up in the `libs/road_profile_gt.py` module docstring and the config comments.

Datasets with no `road_profile.csv` fall back to a legacy GT that reconstructs the road ahead from the vehicle's own later body pitch.

---

## Configuration Reference

All inference parameters live in `config/inference_road_lane_segmentation.yaml`. The sections map 1:1 onto the pipeline stages, plus `ground_truth`, `visualization` and `csv_io`. The file itself carries long comments recording where each calibrated value came from; this is only a map.

```yaml
model:
  device: "cuda"                # "cpu" or "cuda"
  model_name: "pidnet_l"
  weight_path: "pidnet_pretrained_model/PIDNet_L_Cityscapes_test.pt"

input:
  image_path: "..."             # single-image inference target
  image_batch_path: "..."       # directory for batch inference
  resize_size: [512, 1024]      # [height, width] — PIL swaps internally

road_segmentation:              # no parameters: PIDNet-L uses argmax, and no
                                # morphological cleanup is applied to the mask

line_segmentation:
  min_segment_length_near: 65   # min segment length (px) at the bottom of the image
  min_segment_length_far: 0     # ... and at the top; interpolated linearly by mid-y

lane_segmentation:              # thresholds are derived from pitch_estimation's calibration
  track_bands: 16               # continuity-tracking band count (clamped to >= 16 internally)

lane_fitting:
  num_samples: 80               # width-sample fallback when samples_per_meter is unset
  samples_per_meter: 6          # geometry mode: z-uniform width-sample density, per metre of visible depth

pitch_estimation:
  f_x: 512                      # horizontal focal length (px, after resize)
  f_y: 455                      # vertical focal length (px, after resize)
  last_resort_lane_width: 3.25  # assumed width, ONLY before a sequence's first measurement (flagged); was `w_real`
  camera_height: 1.08           # camera mount height above the road (m); also enables geometry mode
  camera_forward_offset: 1.5    # camera mount offset ahead of the vehicle origin (m), for GT distance alignment
  method: windowed              # "windowed" (default) or "spline"

ground_truth:
  height_source: "auto"         # "auto" | "analytic" | "mesh"

visualization:
  alpha: 0.4
  save_path: "outputs/result.png"

csv_io:
  measurements_csv: "inference_datasets/<dataset>/measurements.csv"
  output_dir: "outputs/measurements.csv"
  problem_csv: "outputs/problem.csv"
  problem_mae_threshold: 2.0
```

> On 2026-09-18 (stage C) `w_real` was renamed `last_resort_lane_width` and demoted to a last resort — the width is measured, see below; the constant is used only before a sequence's first measurement, and flagged. What follows is why the constant cannot be trusted as a parameter.
>
> ⚠ **`w_real = 3.25` is specific to this road's marking layout** — double yellow on the left, single white on the right, on a 3.5 m lane. Because `w_real` is inner-edge to inner-edge, each side is inset from the boundary-centre width by a different amount (left 0.1875, right 0.0625, total 0.25). On the same 3.5 m lane: single+single → 3.375, double+single → **3.25**, double+double → 3.125. Re-derive it per road; do not carry 3.25 to another map or another lane. And determine it from height/geometry, never by minimising MAE — MAE trades the z scale against pitch error and its optimum is systematically pulled low. Measured on 2026-09-15 over all three routes: the GT-implied truth is **3.3226 m** (`w_px·z_gt/f_x`, 2,011 frames) while profile MAE is minimised at **3.25**, and using the true width costs +0.020 MAE (+8.5%) on every route. The gap is not noise — a wrong `w_real` is cancelling a second systematic error somewhere else, the same pattern as the 2026-07-27 finding where it cancelled a missing camera forward offset.

**The lane width is measured, not assumed.** Lane width really does vary per road, so a single constant is wrong somewhere: the GT-projected truth is 3.29 / 3.55 / 3.31–3.39 across `full_road`'s three `road_id`s, and a fixed 3.25 is an 8 % depth-scale error on the widest of them. `NearfieldWidthCalibrator` reads it per frame off the near-field ground plane, anchored on the known camera height:

| Piece | How it is set |
|---|---|
| Depth window | Derived from `f_y·h` — the nearer of a curvature bound and a row-count bound. The old fixed z ∈ [2, 5] m was a property of *this* 1.08 m camera; on OpenLane's 2.1 m mount that window contains zero rows, so the self-calibration silently never ran |
| Estimate | The Theil-Sen **intercept** `w_real_z0`, which is free of a camera pitch θ0 *as long as the window lies on the wheel plane*. Per-section error against GT: 46 mm on CARLA and 35 mm on OpenLane, against 108 / 173 mm for the fixed constant |
| Gate | Row count, z-span, residual MAD, and **\|θ0\| ≤ 1°**. A level-mounted camera sits within about 1° of the wheel plane (OpenLane extrinsics −0.3…+0.8°), so a larger θ0 means the window is on a grade change — and there the intercept is *not* θ0-free: the width error runs about 4 % per degree of θ0 (per-frame correlation −0.90 on CARLA, −0.48 on OpenLane; WWH-27). The 1° gate rejects 17 % of the frames the old 3° bound passed on OpenLane and 38–43 % on the hilly CARLA routes. (An older 0.3° gate rejected 93 % of frames on a high camera) |
| Adoption | From a sequence's third passing frame on, the width is the **median of the last 40 passes** (≈ 33 m of travel on OpenLane, ≈ 8 m on CARLA's 0.125 m frame spacing): one frame's error is mostly independent of the next one's. Before that, a run of passing frames (≥ 2 frames and ≥ 0.5 m), so a lone clean-looking frame at a curvature inflection is not trusted; the run survives one failed frame, not two. Pooling + the 1° gate: OpenLane ego-lane near-range Z-error 6.04 → 4.57 cm, CARLA width error median 5.1 → 0.5 % (WWH-27) |
| No measurement | User decision, 2026-09-18. Within a sequence a rejected frame reuses the last adopted value however old it is, and reports its age. Only before the sequence's first adoption is `last_resort_lane_width` used, with the frame marked `last_resort`; without that key the frame outputs no pitch (`no_anchor`) |

It used to be an opt-in flag, off by default, because the width gets measurably better while profile MAE gets 8–15 % *worse* — exactly the z-scale-against-pitch trade the warning above describes. The user ruled (2026-09-15) that adapting to real road widths comes first and the MAE cause is chased separately, so it is now the only source of the width.

> ⚠ The pooled median is slow to follow a real width change (a lane change, a widening road): a new width takes over only once it is the majority of the last 40 passes. The one width change at hand, CARLA `full_road` (3.29 / 3.55 m between roads), gives the same error with 20 or 40 passes; a lane change is untested.
>
> ⚠ Holding without a bound means a held width goes stale when the road changes under it. OpenLane segment 13356 widens from 3.07 to 3.79 m (inner edges) over 22 m; it is measured once, at 2.71 m against 3.11 true (−13 %), and only three later frames pass the quality gate, each two or more failures apart, so that reading is held to the end (−29 % by then). No adoption rule fixes it — the frames that would have are rejected by the quality gate itself.
>
> ⚠ **Known limitation, not fixed (user decision 2026-10-08): a held width rests on whatever few frames happened to pass.** OpenLane segment 13336… (133368) is a jammed on-ramp: the car creeps ~0.1 m per frame (191 frames cover 20 m) with a dashed left line. The only passes before frame 146 are frames 6–10 — four readings within 0.9 m of travel, at θ0 0.7–1.0° (the window sits where the road starts to climb) — and they read 3.31–3.38 m against a true 3.50 (−5 %). From frame 15 to 134 the gap between two dashes sits inside the 1.9 m near-field window, so the left side has no paint there and nothing is measured; the Kalman filter, treating the four readings as independent, holds 3.33 m with a claimed ±1.5 % for 97 frames. Near-range height error over that stretch is ~10 cm, and 3–4 cm with the same lines at the true width. When paint returns (θ0 0.1–0.4°, readings 3.4–3.6 m) the filter weights the old readings equally and takes ~45 frames to reach −1.3 %. Tried: inflating the measurement noise with θ0 only speeds the catch-up, not the hold; measuring the window on model-inferred rows across the gap is worse (8.1 → 9.6 cm — the inferred line bends, θ0 1.6–2.1°). Diagnosis script: `debug/zerror/seg133368.py` (local).
>
> A run survives one failed frame (2026-09-18). Before that an unbroken run was required, and segment 17612 — whose near field has no rows every other frame — never completed one and output nothing for all 30 frames, while its readings were good; it now adopts 2.99 m against 2.88. Allowing any earlier pass within the window instead was tried and rejected: on CARLA's 0.125 m frame spacing it let an inflection reading through and the width error p90 rose 6.2 → 10.1 %.

> ⚠ The gate rejects a *noisy* fit, not a *confidently wrong* one. On OpenLane's San Francisco tram-track segment the tracker locks onto the rails: they are straight, parallel and bright, so the residual MAD (0.001 m) beats a healthy frame's (0.002 m) while the width reads 1.58 m against a true 3.17 m. That failure has to be caught upstream, in line selection.

---

## Critical Conventions

- **`setup_env()` must be the very first call** in any entry-point script, before any `import cv2`, `import pyelsed`, or `import carla`. It loads `.env` and registers OpenCV DLL paths on Windows. See `utils/env_setup.py`.
- **Image format**: the pipeline expects **RGB** throughout. Only convert to BGR with `cv2.cvtColor` immediately before `cv2.imwrite()`.
- **Coordinate system**: OpenCV convention — origin at top-left, y increases downward. Left lane lines have **negative** slope, right lane lines have **positive** slope.
- **`resize_size` is `[height, width]`** in the YAML, but PIL's `image.resize()` expects `(width, height)`. The swap is handled inside `predict_road()` — do not swap it again.
- **`w_real` is inner-edge to inner-edge**, not CARLA's `waypoint.lane_width` (which is boundary-centre to boundary-centre).
- **`pipeline.py` has a second copy**: `utils/inference_road_lane_segmentation.py` inlines the same stages to draw intermediates. Changes to the pipeline must be made in both.
