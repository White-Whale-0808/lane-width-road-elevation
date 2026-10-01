# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monocular road pitch angle estimation using two-plane geometry. The pipeline processes single camera images through 5 sequential stages to estimate the road's pitch angle in degrees.

## Setup

```bash
# Requires Python 3.12, uses uv as package manager
cp .env.example .env        # Machine-specific paths (CARLA wheel, OpenCV dirs)
uv sync                     # Install dependencies
python scripts/setup_elsed.py  # Build local patched pyelsed (ELSED C++ extension)
```

PIDNet weights must be placed in `pidnet_pretrained_model/PIDNet_L_Cityscapes_test.pt`.

## Common Commands

```bash
# Single image inference with visualization
python -m utils.inference_road_lane_segmentation

# Batch inference (CSV in/out; also writes outputs/profile_mae_<dataset>.png and
# outputs/route_profile_<dataset>.png — the whole-route terrain profile)
python -m utils.batch_inference_road_lane_segmentation

# Batch overrides: another dataset dir (one dir = one continuous sequence),
# and a suffix so several runs do not overwrite each other
python -m utils.batch_inference_road_lane_segmentation \
    --dataset <DIR> --tag <SUFFIX>

# Unit tests — pure geometry, ~2 s, no images/weights/GPU.
# ⚠ --no-sync is required: a plain `uv run` swaps the CUDA torch for the CPU wheel
uv run --no-sync python -m pytest

# Same as the single-image command above — main.py is a thin wrapper around it
python main.py
```

All inference config lives in `config/inference_road_lane_segmentation.yaml`.
`openlane_module/` converts the OpenLane dataset into this project's format and
holds the tools that measure what that data can verify — see its own README
(including the official 3D-lane scorer and the ego-lane Z-error evaluation, WWH-26).
`baselines/latr/` runs the public LATR (ICCV 2023) OpenLane weights locally with no
compiled ops, as the Z-error comparison baseline — see its README.

## Pipeline Architecture

Entry point: `libs/inference/pipeline.py::infer_one()`

```
road_segmentation → line_segmentation → lane_segmentation → lane_fitting → pitch_estimation
```

| Stage | File | What it does |
|---|---|---|
| road_segmentation | `libs/inference/road_segmentation.py` | PIDNet-L semantic segmentation → binary road mask |
| line_segmentation | `libs/inference/line_segmentation.py` | ELSED line detection with adaptive length threshold (perspective-aware) |
| lane_segmentation | `libs/inference/lane_segmentation.py` | Near-to-far continuity tracking: seed innermost lanes at bottom, track upward band-by-band by predicted-x association. Thresholds (association tolerance, seed window, slope gates, model memory) are **derived from projection geometry** — camera intrinsics + height are required |
| lane_fitting | `libs/inference/lane_fitting.py` | Inner-chain extraction (`inner_chain_points`: shadowing → dense per-row inner envelope → fragments split at gaps / x-jumps → junction-consistency purge keeping the largest fragment group) → sub-pixel edge refinement on the **unmasked** image (`refine_inner_points`: nearest qualifying gradient peak ±3 px, parabola sub-pixel) → continuous gap-bridged lane curve per side (`lane_curve`) |
| pitch_estimation | `libs/inference/pitch_estimation.py` | `estimate_pitch_from_curves`: sample lane widths from the two curves (z-uniform), inverse perspective → depth, then continuous pitch(z) via local z-window Theil-Sen slopes (`method: windowed`, default — window ±max(1 m, 0.15·z) is the explicit spatial resolution) or the global weighted spline (`method: spline`) |

The pipeline also runs the WWH-15 evidence guards:

- `libs/inference/paint_evidence.py` — photometric "is this actually paint?"
  checks (a marking is a bright ridge of bounded width). `filter_paint_segments`
  drops non-paint ELSED segments before tracking (kills shadow boundaries so the
  tracker re-seeds on the true line); `truncate_at_evidence_break` cuts the
  refined inner chain at the first sustained failure (crest-occlusion tails).
  Thresholds are module constants with derivations in the docstring, not config.
- `lane_fitting.truncate_at_depth_jump` — depth-continuity guard on paired-row
  z: a jump exceeding both the continuity gate and 1.5× the local-plane
  extrapolation marks a hidden interval (real paint beyond a crest is a
  DISCONNECTED section and is never joined to the near chain). Depth is in
  **lane widths** (`f_x/width`), so it takes no `w_real`; the old 3.0 m floor
  is carried over as 3.0/3.25 lane widths (same cut on every one of 488
  OpenLane Tier A frames).

`lane_segmentation.py` is a single unified tracker (the older slope-dependent
`_positive_angle.py` / `_negative_angle.py` variants have been removed).
`f_x`, `f_y` and `camera_height` are **required** — every threshold is computed
from the pinhole model (see the module docstring). The hand-tuned fallback for
un-calibrated cameras (`min_slope` / `lane_band_tolerance` / `roi_near` /
`roi_far`) was removed 2026-08-27: no caller ever reached it.

**It does NOT take `w_real`** (since 2026-09-15). Every threshold there is a
lateral distance, and the lane width only ever converted "a fraction of a lane"
into metres, so stages 1–3 now run on a road whose width they do not know.
Three of the five lateral constants are gone, each on measured evidence rather
than judgement (ablation: push a gate's distance to 1e6 m, diff the pitch-curve
hashes over CARLA 120 frames + OpenLane Tier A 168 frames). Two survive,
`_TOL_X_M` and `_SLOPE_GATE_X_M`, and both are still hand-picked.

| Constant | Outcome |
|---|---|
| ROI corridor, seed-window outer bound | **deleted** — 288/288 frames bit-identical, and not one of 764 seeds moved over 552 dashed-marking frames |
| cross-lane cap | **measured per frame** (`_measure_lane_width_m`): each side's lateral offset at its own seed row, summed. Unmeasurable ⇒ cap off, never a guessed width |
| slope gate | **kept** — the one lateral scale that cannot be measured instead of assumed; it runs in `_segment_info`, before any line has been found |

⚠ Two traps behind that table, both written up with the numbers in the README:
the deleted pair was inert because of the ±15° grade slack in `z_min`, **not**
because innermost-first selection makes an outer bound redundant; and the slope
gate looks removable on CARLA only because those routes have no junctions —
**a gate that rejects stop lines and crosswalks cannot be evaluated on roads
with none.** The failure the deleted pair was meant to prevent is real, predates
their removal, and needs a flat-road bound instead (WWH-21).

## Critical Conventions

- **`setup_env()` must be called before any C extension imports** (cv2, pyelsed, carla). See `utils/env_setup.py`.
- **Image format**: Pipeline expects RGB throughout. Use `cv2.cvtColor` to BGR only for `cv2.imwrite()`.
- **Coordinate system**: OpenCV convention (origin top-left, y-down). Left lane has negative slope, right lane positive.
- **`resize_size` in config is `[height, width]`**, but PIL expects `(width, height)` — the swap is handled in `predict_road()`.

## Config Structure (YAML)

Config sections map 1:1 to pipeline stages:
- `road_segmentation` — (currently empty, PIDNet uses argmax not threshold)
- `line_segmentation` — `min_segment_length_near`, `min_segment_length_far`
- `lane_segmentation` — `track_bands` only (continuity-tracking band count, clamped to >= 16 internally; independent of `lane_fitting.num_bands`). All other thresholds are derived from `pitch_estimation`'s `f_x` / `f_y` / `camera_height` — **not** from `w_real`, which this stage no longer takes
- `lane_fitting` — `samples_per_meter` (geometry mode: z-uniform width-sample density in pitch_estimation, per meter of visible depth), `num_samples` (width-sample fallback when `samples_per_meter` is unset). `inner_chain_points` itself has no density tunables: shadowing + dense per-row inner envelope + fragment/junction purge, every kept row becomes a point — `lane_curve` (continuous gap-bridged polyline per side) is the model and pitch_estimation resamples it. `w_real` is inner-edge-to-inner-edge and **road-specific** (3.25 = this dataset's double-yellow-left / single-white-right on a 3.5 m lane; re-derive it per marking combination — the config comment has the table). This stage no longer takes it at all (stage C, 2026-09-18); width sampling and pitch live in pitch_estimation (`sample_widths_from_curves` / `estimate_pitch_from_curves`)
- `pitch_estimation` — `f_x`, `f_y`, `camera_height`, `camera_forward_offset` (`camera_height` feeds the lane_segmentation geometry), `method` (`windowed` default = local z-window Theil-Sen, no global residual filter; `spline` = global weighted UnivariateSpline with Theil-Sen MAD prefilter). `last_resort_lane_width` (3.25; **was `w_real` until 2026-09-18**, renamed so it cannot pass for a measured parameter). The lane width is measured per frame from the near-field ground plane (`NearfieldWidthCalibrator`, one instance per continuous sequence = one dataset dir), resolved by `resolve_lane_width` in three tiers (stage C, user decision 2026-09-18): **measured** → **held** (a rejected frame reuses the sequence's last measurement however old; age in `w_real_hold_m` / `w_real_hold_frames`) → **last_resort** (only when the sequence has measured nothing yet; the constant is used and the frame is flagged `w_real_status = last_resort`). Remove the key / set null and those frames output no pitch instead (`no_anchor`). Cause of every non-measured frame in `w_real_reason`; the batch prints the tally. A lone image (`infer_one` without a calibrator, single-image runner) has no hold tier. The depth window is derived from `f_y·h`, the estimate is the θ0-free Theil-Sen intercept `w_real_z0`, and a quality gate (row count / z-span / residual MAD) rejects noisy fits; the old fixed z∈[2,5] m window and |θ0|≤0.3° gate were properties of *this* camera and are gone (WWH-19, 2026-09-12). Lane width really does vary per road: GT-projected truth is 3.29/3.55/3.31–3.39 on full_road and 3.34 vs 3.25 on the two hill routes, and the estimate tracks it to 35–52 mm against 108–173 mm for a fixed constant. It was off by default until stage C because the width gets measurably *better* while profile MAE gets 8–15 % *worse* — the same z-scale-against-pitch trade the `w_real` note warns about; the user ruled on 2026-09-15 to accept that and chase the MAE cause separately. ⚠ The gate rejects noise, not a confidently wrong fit: on OpenLane's tram-track segment the residual MAD beats a healthy frame's while the width reads 1.47–1.58 m against a true 3.17 — that has to be caught upstream. Adoption needs a run of passes spanning ≥ 0.5 m that survives **one** failed frame (a flickering near field, OpenLane 17612, never completed an unbroken run; accepting any pass within z_hi instead was replayed and rejected — CARLA width p90 6.2 → 10.1 %). ⚠ With an unbounded hold, a held width goes stale when the road changes: OpenLane 13356 widens 3.07 → 3.79 m, is measured once at 2.71 (−13 %) and holds it to the end (−29 %), because later frames fail the quality gate itself.
- `ground_truth` — **not a pipeline stage**: this is the reference the runners score against. `height_source` picks which column of `road_profile.csv` supplies the profile height — `auto` (default: `z_mesh` when present, else `z`), `analytic` (`z`, the OpenDRIVE centreline) or `mesh` (`z_mesh`, the downward ray-cast). Asking for `mesh` on a pre-WWH-14 dataset raises; only `auto` falls back. The collector's road surface deviates from the analytic centreline in a few localised sections and the camera sees it, so `mesh` is the better reference — see the module docstring in `libs/road_profile_gt.py` for the evidence, and **do not re-litigate it with absolute-height MAE**, which is dominated by a per-frame constant offset.
