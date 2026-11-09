#!/usr/bin/env python3
"""The sixth protocol dimension: temporal metrics.
"""

from __future__ import annotations

import argparse
import csv, glob
import json
import logging
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .utils import (
    N_HAND_KEYPOINTS,
    is_right_to_label,
    normalize_hand_pts,
)

logger = logging.getLogger("temporal_metrics")

# Keypoint subsets. "all" is the 21 hand keypoints; "base" is the
# structural anchor (wrist + 5 MCPs); "tips" the fingertips.
_SUBSETS: Dict[str, Tuple[int, ...]] = {
    "all": tuple(range(N_HAND_KEYPOINTS)),
    "base": (0, 5, 9, 13, 17),
    "tips": (4, 8, 12, 16, 20),
}

# Frame-name pattern: trailing integer index before .jpg.
_FRAME_IDX_RE = re.compile(r"_(\d+)\.jpg$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preds-dir", required=True,
                   help="Directory with the 52 prediction JSONs of one "
                        "estimator/variant.")
    p.add_argument("--estimator-name", required=True)
    p.add_argument("--variant-label", required=True,
                   help="e.g. 'raw' or 's0s1'. Output label only.")
    p.add_argument("--detection-min-kp", type=int, default=5,
                   help="K: min valid keypoints for a slot to count as "
                        "detected in a frame (default 5).")
    p.add_argument("--jitter-min-windows", type=int, default=10,
                   help="N: min valid windows a keypoint's series needs "
                        "before it enters the mean-over-joints, within "
                        "each subset (default 10).")
    p.add_argument("--frame-dims", nargs=2, type=int, metavar=("H", "W"),
                   default=[480, 640])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Frame parsing / ordering
# ---------------------------------------------------------------------------


def _frame_index(frame_name: str) -> int:
    """Trailing integer of a frame name; -1 if none (sorts first)."""
    m = _FRAME_IDX_RE.search(frame_name)
    return int(m.group(1)) if m else -1


def load_video_json(path: str) -> Dict[str, list]:
    with open(path) as f:
        return json.load(f)


def predicted_slots_per_frame(
    frame_entries: list,
    frame_w: int,
    frame_h: int,
) -> Dict[str, Optional[list]]:
    """Return {'left': kps|None, 'right': kps|None} for one frame, with
    coordinates normalized to [0, 1] by image size.

    Takes the first hand of each handedness (the project's "primary per
    slot" convention). ``normalize_hand_pts`` returns pixel coordinates
    (and applies the off-frame filter); we then divide by (W, H) so all
    downstream finite differences are image-relative.
    """
    out: Dict[str, Optional[list]] = {"left": None, "right": None}
    if not isinstance(frame_entries, list):
        return out
    for hand in frame_entries:
        if not isinstance(hand, dict):
            continue
        ir = hand.get("is_right")
        if ir is None:
            continue
        label = is_right_to_label(ir).lower()  # 'right' / 'left'
        if label not in out or out[label] is not None:
            continue
        px = normalize_hand_pts(
            hand.get("hand_pts", []), frame_w, frame_h,
            apply_out_of_frame_filter=True,
        )
        out[label] = _to_image_relative(px, frame_w, frame_h)
    return out


def _to_image_relative(
    kps: Sequence,
    frame_w: int,
    frame_h: int,
) -> List[Optional[Tuple[float, float]]]:
    """Divide each present keypoint by (W, H) -> coords in [0, 1]."""
    out: List[Optional[Tuple[float, float]]] = []
    for kp in kps:
        if kp is None:
            out.append(None)
        else:
            out.append((float(kp[0]) / frame_w, float(kp[1]) / frame_h))
    return out


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------


def n_valid_kp(kps: Optional[Sequence]) -> int:
    if kps is None:
        return 0
    return sum(1 for k in kps if k is not None)


def is_detected(kps: Optional[Sequence], min_kp: int) -> bool:
    return n_valid_kp(kps) >= min_kp


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------


def gap_lengths_for_slot(detected_flags: List[bool]) -> List[int]:
    """Lengths of runs of non-detected frames BETWEEN the first and last
    detection. Leading/trailing absence is excluded.

    ``detected_flags`` is ordered by frame index.
    """
    detected_idx = [i for i, d in enumerate(detected_flags) if d]
    if len(detected_idx) < 2:
        return []  # no interior region to host a gap
    first, last = detected_idx[0], detected_idx[-1]
    gaps: List[int] = []
    run = 0
    for i in range(first + 1, last):
        if detected_flags[i]:
            if run > 0:
                gaps.append(run)
                run = 0
        else:
            run += 1
    # No trailing flush needed: the loop stops before `last`, which is a
    # detected frame by construction, so any open run was already closed
    # at the frame just before `last`. But to be safe with edge shapes:
    if run > 0:
        gaps.append(run)
    return gaps


# ---------------------------------------------------------------------------
# Temporal differential metrics (J_acc, J_jerk, E_v)
# ---------------------------------------------------------------------------


def _diff_series_per_keypoint(
    slot_kps_by_frame: List[Optional[list]],
    detected_flags: List[bool],
    order: str,
) -> List[List[float]]:
    """Per-keypoint list of finite-difference magnitudes across windows.

    ``order`` in {'vel', 'acc', 'jerk'} selects the stencil:
        vel  (2-frame pair t, t+1):
             d = pos_{t+1} - pos_t
        acc  (3-frame window t-1, t, t+1):
             d = pos_{t+1} - 2 pos_t + pos_{t-1}
        jerk (5-frame window t-2..t+2, direct 5-point central):
             d = pos_{t+2} - 2 pos_{t+1} + 2 pos_{t-1} - pos_{t-2}

    A window contributes a keypoint's magnitude only when the slot is
    detected in ALL frames of the window AND the keypoint is valid
    (non-None) in all of them. Coordinates are already image-relative
    [0, 1], so the magnitudes are image-relative (no further
    normalization).

    Returns a list of length 21; entry i is the list of magnitudes for
    keypoint i over all valid windows.
    """
    n = len(slot_kps_by_frame)
    per_kp: List[List[float]] = [[] for _ in range(N_HAND_KEYPOINTS)]

    if order == "vel":
        rng = range(0, n - 1)
        offsets = (0, 1)
    elif order == "acc":
        rng = range(1, n - 1)
        offsets = (-1, 0, 1)
    elif order == "jerk":
        rng = range(2, n - 2)
        offsets = (-2, -1, 0, 1, 2)
    else:
        raise ValueError(f"unknown order: {order}")

    for t in rng:
        # All frames in the window must be detected.
        if not all(detected_flags[t + o] for o in offsets):
            continue
        frames = [slot_kps_by_frame[t + o] for o in offsets]
        if any(fr is None for fr in frames):
            continue

        for i in range(N_HAND_KEYPOINTS):
            pts = [fr[i] if i < len(fr) else None for fr in frames]
            if any(p is None for p in pts):
                continue
            if order == "vel":
                p0, p1 = pts
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
            elif order == "acc":
                p_prev, p_cur, p_next = pts
                dx = p_next[0] - 2.0 * p_cur[0] + p_prev[0]
                dy = p_next[1] - 2.0 * p_cur[1] + p_prev[1]
            else:  # jerk: p[t+2] - 2p[t+1] + 2p[t-1] - p[t-2]
                p_m2, p_m1, _p_0, p_p1, p_p2 = pts
                dx = p_p2[0] - 2.0 * p_p1[0] + 2.0 * p_m1[0] - p_m2[0]
                dy = p_p2[1] - 2.0 * p_p1[1] + 2.0 * p_m1[1] - p_m2[1]
            per_kp[i].append(math.hypot(dx, dy))

    return per_kp


def _metric_over_subset(
    per_kp: List[List[float]],
    subset_idx: Sequence[int],
    jitter_min_windows: int,
) -> Tuple[float, int]:
    """Mean over time then over valid joints of a per-keypoint magnitude
    series, restricted to ``subset_idx``.

    A keypoint enters the mean-over-joints only if its series has at
    least ``jitter_min_windows`` windows. Returns (metric, n_kp_used).
    The metric is the mean over the surviving keypoints of each
    keypoint's temporal mean. NaN (with n_kp_used=0) if no keypoint in
    the subset survives.
    """
    per_kp_means: List[float] = []
    for i in subset_idx:
        series = per_kp[i]
        if len(series) >= jitter_min_windows:
            per_kp_means.append(float(np.mean(series)))
    if not per_kp_means:
        return float("nan"), 0
    return float(np.mean(per_kp_means)), len(per_kp_means)


def _count_valid_windows(
    slot_kps_by_frame: List[Optional[list]],
    detected_flags: List[bool],
    order: str,
) -> int:
    """Number of windows where the slot is detected in all frames of the
    stencil (control count; subset-independent)."""
    n = len(slot_kps_by_frame)
    if order == "acc":
        rng = range(1, n - 1)
        offsets = (-1, 0, 1)
    elif order == "jerk":
        rng = range(2, n - 2)
        offsets = (-2, -1, 0, 1, 2)
    else:
        raise ValueError(order)
    count = 0
    for t in rng:
        if not all(detected_flags[t + o] for o in offsets):
            continue
        if any(slot_kps_by_frame[t + o] is None for o in offsets):
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------


def process_video(
    video_label: str,
    frames_json: Dict[str, list],
    frame_w: int,
    frame_h: int,
    detection_min_kp: int,
    jitter_min_windows: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Process one video JSON. Returns (wide_rows, gap_rows) for both
    slots."""
    frame_names = sorted(frames_json.keys(), key=_frame_index)

    # Pre-extract per-frame slot keypoints.
    left_kps: List[Optional[list]] = []
    right_kps: List[Optional[list]] = []
    for fname in frame_names:
        slots = predicted_slots_per_frame(frames_json[fname], frame_w, frame_h)
        left_kps.append(slots["left"])
        right_kps.append(slots["right"])

    wide_rows: List[Dict[str, object]] = []
    gap_rows: List[Dict[str, object]] = []

    for slot, kps_by_frame in (("left", left_kps), ("right", right_kps)):
        detected_flags = [is_detected(k, detection_min_kp) for k in kps_by_frame]
        n_detected = int(sum(detected_flags))

        # Per-keypoint magnitude series for each order.
        vel_series = _diff_series_per_keypoint(kps_by_frame, detected_flags, "vel")
        acc_series = _diff_series_per_keypoint(kps_by_frame, detected_flags, "acc")
        jerk_series = _diff_series_per_keypoint(kps_by_frame, detected_flags, "jerk")

        n_valid_accel = _count_valid_windows(kps_by_frame, detected_flags, "acc")
        n_valid_jerk = _count_valid_windows(kps_by_frame, detected_flags, "jerk")

        rec: Dict[str, object] = {"video": video_label, "slot": slot}

        # Three metrics x three subsets.
        n_kp_by_subset: Dict[str, int] = {}
        for metric_name, series in (("Jacc", acc_series),
                                    ("Jjerk", jerk_series),
                                    ("Ev", vel_series)):
            for subset_name, subset_idx in _SUBSETS.items():
                val, n_kp = _metric_over_subset(
                    series, subset_idx, jitter_min_windows
                )
                rec[f"{metric_name}_{subset_name}"] = val
                # Use J_acc as the canonical source for the per-subset
                # effective-keypoint count.
                if metric_name == "Jacc":
                    n_kp_by_subset[subset_name] = n_kp

        # Gaps (unchanged).
        gaps = gap_lengths_for_slot(detected_flags)
        if gaps:
            arr = np.asarray(gaps, dtype=float)
            gap_mean = float(arr.mean())
            gap_median = float(np.median(arr))
            gap_p90 = float(np.percentile(arr, 90))
            gap_max = float(arr.max())
        else:
            gap_mean = gap_median = gap_p90 = gap_max = float("nan")

        rec.update({
            "n_gaps": len(gaps),
            "gap_mean": gap_mean,
            "gap_median": gap_median,
            "gap_p90": gap_p90,
            "gap_max": gap_max,
            "n_detected_frames": n_detected,
            "n_valid_accel_windows": n_valid_accel,
            "n_valid_jerk_windows": n_valid_jerk,
            "n_kp_all": n_kp_by_subset.get("all", 0),
            "n_kp_base": n_kp_by_subset.get("base", 0),
            "n_kp_tips": n_kp_by_subset.get("tips", 0),
        })

        wide_rows.append(rec)
        for g in gaps:
            gap_rows.append({
                "video": video_label,
                "slot": slot,
                "gap_length": int(g),
            })

    return wide_rows, gap_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_WIDE_FIELDS = [
    "estimator", "variant", "video", "slot",
    "Jacc_all", "Jacc_base", "Jacc_tips",
    "Jjerk_all", "Jjerk_base", "Jjerk_tips",
    "Ev_all", "Ev_base", "Ev_tips",
    "n_gaps", "gap_mean", "gap_median", "gap_p90", "gap_max",
    "n_detected_frames", "n_valid_accel_windows", "n_valid_jerk_windows",
    "n_kp_all", "n_kp_base", "n_kp_tips",
]
_GAP_FIELDS = ["estimator", "variant", "video", "slot", "gap_length"]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)

    h, w = args.frame_dims

    json_files = sorted(p.split("\\")[-1] for p in glob.glob(os.path.join(args.preds_dir, "*.json")) if not p.endswith(".scores.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {args.preds_dir}")
    logger.info(
        "%s (%s): %d video JSONs; K=%d, jitter_min_windows=%d",
        args.estimator_name, args.variant_label, len(json_files),
        args.detection_min_kp, args.jitter_min_windows,
    )

    all_wide: List[Dict[str, object]] = []
    all_gaps: List[Dict[str, object]] = []

    for jf in json_files:
        video_label = os.path.splitext(jf)[0]
        path = os.path.join(args.preds_dir, jf)
        try:
            frames_json = load_video_json(path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", jf, e)
            continue
        if not isinstance(frames_json, dict) or not frames_json:
            logger.warning("Skipping %s: empty or non-dict JSON.", jf)
            continue

        wide_rows, gap_rows = process_video(
            video_label, frames_json, w, h,
            args.detection_min_kp, args.jitter_min_windows,
        )
        for r in wide_rows:
            r["estimator"] = args.estimator_name
            r["variant"] = args.variant_label
        for r in gap_rows:
            r["estimator"] = args.estimator_name
            r["variant"] = args.variant_label
        all_wide.extend(wide_rows)
        all_gaps.extend(gap_rows)

        logger.debug(
            "%s: left/right detected frames = %d/%d",
            video_label,
            wide_rows[0]["n_detected_frames"],
            wide_rows[1]["n_detected_frames"],
        )

    # Write the wide CSV.
    out_wide = os.path.join(
        args.output_dir,
        f"temporal_{args.estimator_name}_{args.variant_label}.csv",
    )
    with open(out_wide, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_WIDE_FIELDS)
        writer.writeheader()
        for r in all_wide:
            writer.writerow({k: r.get(k, "") for k in _WIDE_FIELDS})
    logger.info("Wrote %s (%d rows)", out_wide, len(all_wide))

    # Write the long gaps CSV.
    out_gaps = os.path.join(
        args.output_dir,
        f"temporal_{args.estimator_name}_{args.variant_label}_gaps.csv",
    )
    with open(out_gaps, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_GAP_FIELDS)
        writer.writeheader()
        for r in all_gaps:
            writer.writerow({k: r.get(k, "") for k in _GAP_FIELDS})
    logger.info("Wrote %s (%d rows)", out_gaps, len(all_gaps))

    return 0


if __name__ == "__main__":
    sys.exit(main())