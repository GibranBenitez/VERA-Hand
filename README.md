# VERA-Hand: <ins>V</ins>erified <ins>E</ins>valuation of <ins>R</ins>eliability <ins>A</ins>cross <ins>Hand</ins> Pose Estimators

[__VERA-Hand: A Verified Multidimensional Benchmark for 2D Hand Pose Estimation in Third-Person Dynamic Gesture Videos__](https://arxiv.org/abs/2610.32517)
<br>
[Gibran Benitez-Garcia](https://gibranbenitez.github.io), and Hiroki Takahashi
<br>
___Accepted to [WACV 2027](https://wacv.thecvf.com/Conferences/2027)___

[Project page](https://gibranbenitez.github.io/VERA_Hand/) ·
[Data (Google Drive)](https://drive.google.com/drive/folders/1JphSAR9u4K0ZWUHfGbAC8JIoPuw7jIKF) ·
[Annotation viewer](https://gibranbenitez.github.io/VERA_Hand/viewer/index.html)

VERA-Hand provides manually verified, frame-level 2D hand annotations (21 keypoints, MediaPipe topology, with a
left/right label) for every visible hand in 10,768 frames sampled from 52 third-person dynamic-gesture videos of the
[IPN Hand dataset](https://gibranbenitez.github.io/IPN_Hand/) (13 subjects, 10,690 hands). Estimators are evaluated along
four axes: **detection**, **localization**, **identification** and **temporal stability**, plus the joint **Pose-F1**.

This repository contains the verified annotations, the blur labels and the evaluation code. Images and
estimator predictions are distributed through Google Drive (see [Data](#data)).

---

## Repository layout

```
VERA-Hand/
  README.md
  requirements.txt
  vera_supervised.py            Frame-level supervised metrics (detection, localization, identification)
  vera_temporal.py              Temporal-stability metrics (jitter, velocity energy, gaps)
  metrics/                      Internal package used by both scripts
  annotations/
    vera_hand_gt.json           Verified ground truth (10,768 frames)
    blur_labels.csv             Motion-blur class for every annotated hand
  examples/
    RTMW-wb_raw.json            One prediction file, for a quick sanity check
  gt_imgs/                      (not in the repo) benchmark frames, from VERA-Hand_images.zip
  predictions/                  (not in the repo) estimator predictions, from VERA-Hand_predictions.zip
```

---

## Data

The repository keeps only the essentials. Everything else is in the
[Google Drive folder](https://drive.google.com/drive/folders/1JphSAR9u4K0ZWUHfGbAC8JIoPuw7jIKF):

| Where | File | Contents |
|-------|------|----------|
| This repo | `annotations/vera_hand_gt.json` | Verified annotations for the 10,768 benchmark frames |
| This repo | `annotations/blur_labels.csv` | Blur class (sharp / medium / blurry) per annotated hand |
| This repo | `examples/RTMW-wb_raw.json` | Example prediction file (RTMW-wb, raw) |
| Google Drive | `VERA-Hand_images.zip` | **Not needed for evaluation.** The 10,768 benchmark frames (`.jpg`, 640×480), named exactly as the keys of the annotation JSON; extracts to `gt_imgs/` |
| Google Drive | `VERA-Hand_predictions.zip` | **For reproducing the paper numbers.** Predictions of the evaluated estimators, in `bench_frames/` and `full_videos/` (format below); extracts to `predictions/` |
| Google Drive | `vera_hand_gt.json` | Copy of the annotation file, for convenience |

The images are needed only to visualize the annotations or to run your own estimator. The metrics in this repository
operate on JSON files only. Extract both archives in the repository root.

### Images

The frames were extracted from the **original IPN Hand videos (640×480)** with:

```bash
ffmpeg -i <video>.avi -vf fps=30 -q:v 2 <video>_%06d.jpg
```

These frames are **not** the 320×240 frames distributed with the original IPN Hand release. They are the original
frames, so faces are **not blurred** (only the frames shown in the online annotation viewer are blurred).

### Annotation format

`annotations/vera_hand_gt.json` is a dictionary keyed by frame filename. Each value is the list of visible hands in
that frame (empty list when no hand is visible):

```json
{
  "1CM1_1_R_#217_000024.jpg": [
    {
      "is_right": 1,
      "hand_pts": [[251.0, 473.0], [281.0, 459.0], "... 21 [x, y] points in total ..."],
      "bb_size": 289.64
    }
  ]
}
```

- `is_right`: `1` for the right hand, `0` for the left hand.
- `hand_pts`: 21 keypoints in pixel coordinates (frame size 640×480, i.e. width 640, height 480), in MediaPipe order
  (0 = wrist, 1–4 thumb, 5–8 index, 9–12 middle, 13–16 ring, 17–20 little finger).
- `bb_size`: hand size used to normalize distances.
- Frame names are `<video>_<frame index>.jpg`, for example `1CM1_1_R_#217` + `000024`.

The benchmark has 261 frames without hands, 10,324 frames with one hand and 183 frames with two hands.

### Blur labels

`annotations/blur_labels.csv` has one row per annotated hand with the columns `frame_name`, `subject_id`, `video_id`,
`slot` (`left` / `right`), `blur_score`, `flow_magnitude_mean`, `flow_magnitude_p90`, `n_pixels_bbox` and
`blur_class` (`sharp`, `medium` or `blurry`). Blur is assigned per hand from optical-flow magnitude and hand-region
sharpness (see Section C.1 of the supplementary material).

---

## Installation

Python ≥ 3.9 with NumPy and pandas:

```bash
pip install -r requirements.txt
```

---

## Quick start

Run the supervised evaluation on the example prediction file (a few seconds):

```bash
python vera_supervised.py \
    --preds-path examples/RTMW-wb_raw.json \
    --gt-path annotations/vera_hand_gt.json \
    --blur-tags annotations/blur_labels.csv \
    --out-dir outputs
```

Expected values in `outputs/RTMW-wb_raw_general-both-agnostic.csv`:

| det_f1 | pck | pose_f1 | swap_rate | cross_hand_overlap |
|--------|-----|---------|-----------|--------------------|
| 0.932 | 0.560 | 0.477 | 0.31% | 0.91% |

These are the RTMW-wb values reported in Tables 3 and 4 of the paper.

---

## Evaluating your own estimator

Write one JSON per estimator with the same structure as the annotations: a dictionary keyed by frame filename, whose
value is the list of predicted hands.

```json
{
  "1CM1_1_R_#217_000024.jpg": [
    { "is_right": 1, "hand_pts": [[x0, y0], [x1, y1], "...", [x20, y20]] }
  ],
  "1CM1_1_R_#217_000025.jpg": []
}
```

- `is_right` is `1` for the right hand and `0` for the left hand.
- `hand_pts` holds 21 keypoints in the MediaPipe order, in pixel coordinates of a 640×480 frame. A keypoint may be
  `null` if the model did not predict it.
- Frames without detections carry an empty list. Every benchmark frame should have an entry.

Then point `--preds-path` to your file. The estimator name in the output is the file name without `.json`.

---

## Reproducing the paper numbers

Download `VERA-Hand_predictions.zip` from Google Drive and extract it in the repository root, so that you get:

```
predictions/
  bench_frames/                One JSON per estimator and variant  ->  vera_supervised.py
    RTMW-wb_raw.json, RTMW-wb_clean.json, Sapiens2-1B_raw.json, Sapiens2-1B_clean.json, ...
  full_videos/                 One folder per estimator and variant, one JSON per full video  ->  vera_temporal.py
    RTMW-wb_raw/<video>.json, ...
```

`raw` are the predictions of the off-the-shelf pipeline. `clean` are the predictions after the training-free post-hoc
cleanup described in Section E of the supplementary material.

### Supervised metrics

```bash
python vera_supervised.py \
    --preds-path predictions/bench_frames/ \
    --gt-path annotations/vera_hand_gt.json \
    --blur-tags annotations/blur_labels.csv \
    --breakdown --out-dir outputs
```

When `--preds-path` is a folder, every JSON inside it is one estimator, and the three output files have one row per
estimator. Values you should obtain (Tables 3 and 4 of the paper):

| Estimator | Det-F1 | PCK@0.10 | Pose-F1 | Swap | Cross-hand overlap |
|-----------|--------|----------|---------|------|--------------------|
| RTMW-wb (raw) | 0.932 | 0.560 | 0.477 | 0.31% | 0.91% |
| RTMW-wb (clean) | 0.951 | 0.553 | 0.477 | 0.28% | 0.79% |
| Sapiens2-1B (raw) | 0.666 | 0.491 | 0.361 | 0.03% | 85.44% |
| Sapiens2-1B (clean) | 0.940 | 0.490 | 0.406 | 0.03% | 0.05% |

### Temporal metrics

Temporal stability is computed on predictions over the complete videos, so `full_videos/` holds one JSON per full
video. The frames of the complete videos are not part of this release; they come from the
[IPN Hand dataset](https://gibranbenitez.github.io/IPN_Hand/).

```bash
python vera_temporal.py \
    --preds-path predictions/full_videos/ \
    --hand right --region all \
    --out-dir outputs
```

---

## Script reference

### `vera_supervised.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--preds-path` | yes | | A prediction JSON, or a folder of prediction JSONs (one per estimator) |
| `--gt-path` | yes | | Ground-truth JSON (`annotations/vera_hand_gt.json`) |
| `--alpha` | | `0.10` | PCK and Pose-F1 distance threshold (fraction of hand size) |
| `--beta` | | `0.50` | Minimum PCK for a hand to count in Pose-F1 |
| `--hand` | | `both` | Hand side to report: `left`, `right` or `both` (macro average of left and right) |
| `--blur-tags` | | | `annotations/blur_labels.csv`; omit to skip the blur analysis |
| `--detection-mode` | | `label_agnostic` | `label_agnostic`: a hand counts as detected regardless of the predicted side, and wrong-side errors are left to the identification axis. `label_aware`: detection uses the handedness-aware matching |
| `--breakdown` | | off | Add precision and recall for Det-F1 and Pose-F1, and MPJPE in pixels |
| `--only-gt-frames` | | off | Ignore frames that are in the predictions but not in the GT |
| `--out-dir` | | `outputs` | Output directory |
| `--log-level` | | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

Outputs, written to `--out-dir` as `{name}_{type}-{hand}[-agnostic].csv`, where `{name}` is the JSON name (single
file) or the folder name (folder of JSONs), `{hand}` is the value of `--hand`, and `-agnostic` is appended in the
default `label_agnostic` mode:

- **`_general`**: one row per estimator.

  | Column | Description |
  |--------|-------------|
  | `det_f1` | Detection F1 |
  | `pck` | PCK at `--alpha`, macro average of left and right |
  | `pose_f1` | Pose-F1 at (`--alpha`, `--beta`), macro average of left and right |
  | `swap_rate` | Fraction of frames with a left/right label swap |
  | `cross_hand_overlap` | Fraction of frames where predictions overlap across hands |
  | `duplicate_hands_rate` | Fraction of frames with duplicate same-side predictions |
  | `mpjpe` | Normalized MPJPE (macro average of left and right) |
  | `det_precision`, `det_recall`, `pose_precision`, `pose_recall`, `mpjpe_pixels` | Only with `--breakdown` |

- **`_region`**: PCK by keypoint group: `pck_wrist`, `pck_palm`, `pck_fingertips`, `pck_whole`.
- **`_blur`** (requires `--blur-tags`): Det-F1 and Pose-F1 per blur class: `sharp_det_f1`, `medium_det_f1`,
  `blurry_det_f1`, `sharp_pose_f1`, `medium_pose_f1`, `blurry_pose_f1`.

### `vera_temporal.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--preds-path` | yes | | Folder with per-video JSONs (one estimator) or a folder of such folders (several estimators) |
| `--hand` | | `right` | `left` or `right`. Temporal metrics are per hand, `both` is not supported |
| `--region` | | `all` | Keypoint subset: `all`, `base` (wrist and MCP joints) or `tips` (fingertips) |
| `--out-dir` | | `outputs` | Output directory |
| `--log-level` | | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

The mode is detected automatically: if `--preds-path` contains JSON files directly, it is a single estimator and the
CSV is named after that folder; if it contains sub-folders with JSON files, each sub-folder is an estimator and the
CSV is named after the parent folder.

Output `{name}_temporal-{hand}_{region}.csv`, one row per estimator:

| Column | Description |
|--------|-------------|
| `estimator` | Estimator name (folder name) |
| `Gaps` | Total number of detection-gap events over all videos |
| `Jacc` | Median acceleration jitter over videos, times 100 |
| `Jjerk` | Median jerk jitter over videos, times 100 |
| `Ev` | Median velocity energy over videos, times 100 |

---

## License

The benchmark code shared in this repository is licensed under the [MIT license](LICENSE).

The annotations and data of VERA-Hand (the frames and the annotation, blur-label and prediction files) are derived from
the IPN Hand dataset and inherit its license: they are licensed under a
[Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).

---

## Citation

If you find useful the VERA-Hand benchmark for your research, please cite the paper:

```bibtex
@article{benitezgarcia2026verahand,
  title         = {{VERA-Hand}: A Verified Multidimensional Benchmark for 2D Hand Pose Estimation in Third-Person Dynamic Gesture Videos},
  author        = {Benitez-Garcia, Gibran and Takahashi, Hiroki},
  journal       = {arXiv preprint arXiv:2610.32517},
  year          = {2026},
  eprint        = {2610.32517},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

The frames and annotations come from IPN Hand, so please also cite the
[IPN Hand paper](https://github.com/GibranBenitez/IPN-hand#citation).
