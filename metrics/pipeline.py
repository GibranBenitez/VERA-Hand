"""pipeline.py — Compute-metrics pipeline for VERA-Hand release scripts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# metrics/ package — make available for re-export and internal use
# ---------------------------------------------------------------------------
from .detection import (  # noqa: E402
    TH_IOU,
    SlotMatchResult,
    match_slot,
    precision_recall_f1,
)
from .detection_agnostic import match_frame_agnostic  # noqa: E402
from .identification import detect_label_swaps  # noqa: E402
from .localization import (  # noqa: E402
    HandLocResult,
    compute_per_keypoint_errors,
    localize_hand,
)
from .utils import (  # noqa: E402
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    KeypointList,
    N_HAND_KEYPOINTS,
    compute_video_hand_size_means,
    is_right_to_label,
    normalize_hand_pts,
    padded_bbox,
    parse_frame_name,
    resolve_hand_size,
)
from .visibility import VisibilityCounters, per_frame_visibility  # noqa: E402
from .frame_categorization import categorize_frame  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-hand geometry helpers
# (extracted verbatim from hand_frame_stats.py — read-only canonical source)
# ---------------------------------------------------------------------------


def valid_points(hand_pts):
    """Return valid (x, y) tuples, ignoring None/null and malformed points."""
    pts = []
    for p in hand_pts:
        if (
            isinstance(p, list)
            and len(p) >= 2
            and p[0] is not None
            and p[1] is not None
        ):
            pts.append((float(p[0]), float(p[1])))
    return pts


def bbox_from_points(points):
    """Return bbox as (x1, y1, x2, y2), or None if no valid points."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_area(box):
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(box_a, box_b):
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return 0.0 if union <= 0 else inter / union


def _bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_diag(box):
    x1, y1, x2, y2 = box
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalized_center_distance(box_a, box_b):
    """Center distance normalized by the largest bbox diagonal."""
    if box_a is None or box_b is None:
        return math.inf
    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    scale = max(_bbox_diag(box_a), _bbox_diag(box_b))
    return math.inf if scale <= 0 else dist / scale


# ---------------------------------------------------------------------------
# Pose-F1 / GT_base configuration
# (extracted verbatim from compute_metrics.py — read-only canonical source)
# ---------------------------------------------------------------------------

POSE_F1_PAIRS: List[Tuple[float, float]] = [
    (0.02, 0.50),
    (0.05, 0.50),
    (0.10, 0.50),
]

GT_BASE_ALPHA: float = 0.02


def _pose_suffix(alpha: float, beta: float) -> str:
    a_str = f"{int(round(alpha * 100)):03d}"
    b_str = f"{int(round(beta * 100)):03d}"
    return f"{a_str}_{b_str}"


def _alpha_suffix(a: float) -> str:
    s = f"{a:.3f}".replace(".", "").rstrip("0")
    return s if s else "0"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Frame splitting
# (extracted verbatim from compute_metrics.py)
# ---------------------------------------------------------------------------


def split_hands_by_slot(
    raw_frame_value: Any,
    frame_w: int,
    frame_h: int,
    *,
    apply_out_of_frame_filter: bool,
    frame_name: str,
    is_gt: bool,
) -> Tuple[
    Optional[KeypointList],
    Optional[KeypointList],
    List[KeypointList],
    List[KeypointList],
    bool,
    bool,
]:
    primary_left: Optional[KeypointList] = None
    primary_right: Optional[KeypointList] = None
    extras_left: List[KeypointList] = []
    extras_right: List[KeypointList] = []
    gt_anomaly = False
    gt_empty = False

    if not isinstance(raw_frame_value, list):
        logger.warning(
            "Frame %s: value is not a list (%s); treating as empty.",
            frame_name, type(raw_frame_value).__name__,
        )
        return None, None, [], [], False, True

    if len(raw_frame_value) == 0:
        gt_empty = True
        if is_gt:
            # Benign and expected for benchmark frames deliberately added
            # with no visible hands (e.g. manual_gt_r2's no-hands frames).
            # Per-frame WARNING would flood warnings.log; the aggregate
            # count is already surfaced via n_frames_no_gt downstream.
            logger.debug("Frame %s: GT has an empty list of hands.", frame_name)
        return None, None, [], [], False, gt_empty

    n_left = 0
    n_right = 0
    for hand in raw_frame_value:
        if not isinstance(hand, dict):
            logger.warning("Frame %s: list entry is not a dict; ignored.", frame_name)
            continue
        is_right = hand.get("is_right")
        if is_right is None:
            logger.warning("Frame %s: hand without is_right field; ignored.", frame_name)
            continue
        label = is_right_to_label(is_right)
        kps = normalize_hand_pts(
            hand.get("hand_pts", []), frame_w, frame_h,
            apply_out_of_frame_filter=apply_out_of_frame_filter,
        )
        if label == "Right":
            n_right += 1
            if primary_right is None:
                primary_right = kps
            else:
                extras_right.append(kps)
        else:
            n_left += 1
            if primary_left is None:
                primary_left = kps
            else:
                extras_left.append(kps)

    if is_gt:
        n_total = n_left + n_right
        if n_total > 2:
            gt_anomaly = True
            logger.warning(
                "Frame %s: GT has %d hands (>2). Likely annotation error.",
                frame_name, n_total,
            )
        if n_left >= 2:
            gt_anomaly = True
            logger.warning(
                "Frame %s: GT has %d Left hands. Duplicate handedness in GT.",
                frame_name, n_left,
            )
        if n_right >= 2:
            gt_anomaly = True
            logger.warning(
                "Frame %s: GT has %d Right hands. Duplicate handedness in GT.",
                frame_name, n_right,
            )

    return primary_left, primary_right, extras_left, extras_right, gt_anomaly, gt_empty


# ---------------------------------------------------------------------------
# Main processing pipeline
# (extracted verbatim from compute_metrics.py)
# ---------------------------------------------------------------------------


def _safe_percentile(values: List[float], q: float) -> float:
    arr = np.array([v for v in values if v is not None and not math.isnan(v)])
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _safe_mean(values: List[float]) -> float:
    arr = np.array([v for v in values if v is not None and not math.isnan(v)])
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def process_all_frames(
    gt_raw: Dict[str, Any],
    pred_raw: Dict[str, Any],
    *,
    frame_w: int,
    frame_h: int,
    alphas: List[float],
    frames_dir: Optional[str],
    progress_every: int,
    detection_mode: str = "label_aware",
    only_gt_frames: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, VisibilityCounters], Dict[str, int]]:
    """``detection_mode``: "label_aware" (default, unchanged) or "both" —
    "both" runs the label-agnostic match ADDITIONALLY, writing "_la"
    suffixed columns alongside the untouched label-aware ones. Callers
    (vera_supervised.py) decide whether the reported det/pose-f1 columns
    come from the label-aware or label-agnostic side. See
    metrics/detection_agnostic.py for the matching policy.

    ``only_gt_frames``: opt-in, default False (unchanged behavior). When
    True, frame coverage is restricted to ``gt_frames`` only — pred-only
    frames are excluded entirely (never iterated, never scored) instead of
    being treated as empty-GT FP. Mirrors compute_metrics.py's
    ``--only-gt-frames``.
    """
    gt_frames = set(gt_raw.keys())
    pred_frames = set(pred_raw.keys())
    only_gt = gt_frames - pred_frames
    only_pred = pred_frames - gt_frames
    if only_gt_frames:
        all_frames = sorted(gt_frames)
    else:
        all_frames = sorted(gt_frames | pred_frames)

    if only_gt:
        logger.info(
            "Frames in GT but missing in pred: %d. Treated as empty pred "
            "(FN for slots with GT, TN otherwise).", len(only_gt),
        )
    if only_pred:
        if only_gt_frames:
            logger.info(
                "Frames in pred but missing in GT: %d. --only-gt-frames is "
                "active: these frames are EXCLUDED ENTIRELY (not scored as "
                "FP, not scored at all).", len(only_pred),
            )
        else:
            logger.info(
                "Frames in pred but missing in GT: %d. Treated as empty GT "
                "(FP for slots with a pred hand, TN otherwise).", len(only_pred),
            )

    frame_to_video: Dict[str, str] = {}
    frame_to_subject: Dict[str, str] = {}
    video_to_frames: Dict[str, List[str]] = defaultdict(list)
    parsed_ok = []
    for frame_name in all_frames:
        fid = parse_frame_name(frame_name)
        if fid is None:
            continue
        frame_to_video[frame_name] = fid.video_uid
        frame_to_subject[fid.video_uid] = fid.subject_id
        video_to_frames[fid.video_uid].append(frame_name)
        parsed_ok.append(frame_name)

    logger.debug(
        "Total frames (union): %d (in GT=%d, in pred=%d, parsed=%d)",
        len(all_frames), len(gt_raw), len(pred_raw), len(parsed_ok),
    )
    logger.debug("Videos detected: %d", len(video_to_frames))
    for i, (v, frames) in enumerate(sorted(video_to_frames.items()), start=1):
        logger.debug("  Video %d/%d: %s (%d frames)", i, len(video_to_frames), v, len(frames))

    gt_hands_per_frame: Dict[str, Dict[str, KeypointList]] = {}
    for frame_name in parsed_ok:
        if frame_name not in gt_raw:
            gt_hands_per_frame[frame_name] = {"Left": None, "Right": None}
            continue
        pl, pr, _el, _er, _ano, _emp = split_hands_by_slot(
            gt_raw[frame_name], frame_w, frame_h,
            apply_out_of_frame_filter=False,
            frame_name=frame_name, is_gt=True,
        )
        gt_hands_per_frame[frame_name] = {"Left": pl, "Right": pr}

    per_video_means, global_means = compute_video_hand_size_means(
        gt_hands_per_frame, frame_to_video
    )
    logger.debug(
        "Global hand_size means (px): Left=%.2f, Right=%.2f",
        global_means.get("Left", float("nan")),
        global_means.get("Right", float("nan")),
    )

    rows: List[Dict[str, Any]] = []
    visibility_per_video: Dict[str, VisibilityCounters] = defaultdict(VisibilityCounters)
    global_fallback_logged: set = set()
    summary = {
        "n_frames_total": len(all_frames),
        "n_frames_processable": 0,
        "n_frames_with_warnings": 0,
        "n_videos": len(video_to_frames),
        "n_frames_only_in_gt": len(only_gt),
        "n_frames_only_in_pred": len(only_pred),
        "only_gt_frames_active": only_gt_frames,
        "only_gt_frames_excluded": len(only_pred) if only_gt_frames else 0,
    }

    video_order = sorted(video_to_frames.items())
    n_processed = 0
    t0 = time.time()
    for vi, (video_uid, frames_in_video) in enumerate(video_order, start=1):
        logger.debug(
            "[Video %d/%d] Processing %s (%d frames).",
            vi, len(video_order), video_uid, len(frames_in_video),
        )
        for frame_name in frames_in_video:
            n_processed += 1
            if n_processed % progress_every == 0:
                pct = 100.0 * n_processed / max(1, len(parsed_ok))
                elapsed = time.time() - t0
                logger.debug(
                    "Processed %d frames (%.1f%% complete, %.1fs elapsed).",
                    n_processed, pct, elapsed,
                )
            row = _process_one_frame(
                frame_name=frame_name,
                gt_raw=gt_raw,
                pred_raw=pred_raw,
                frame_w=frame_w,
                frame_h=frame_h,
                alphas=alphas,
                frames_dir=frames_dir,
                gt_hands=gt_hands_per_frame.get(frame_name),
                video_uid=video_uid,
                subject_id=frame_to_subject[video_uid],
                per_video_means=per_video_means,
                global_means=global_means,
                vis_counter=visibility_per_video[video_uid],
                global_fallback_logged=global_fallback_logged,
                detection_mode=detection_mode,
            )
            rows.append(row)
            if row.get("frame_processable"):
                summary["n_frames_processable"] += 1

    df = pd.DataFrame(rows)
    return df, visibility_per_video, summary


def _process_one_frame(
    *,
    frame_name: str,
    gt_raw: Dict[str, Any],
    pred_raw: Dict[str, Any],
    frame_w: int,
    frame_h: int,
    alphas: List[float],
    frames_dir: Optional[str],
    gt_hands: Optional[Dict[str, KeypointList]],
    video_uid: str,
    subject_id: str,
    per_video_means: Dict[Tuple[str, str], float],
    global_means: Dict[str, float],
    vis_counter: VisibilityCounters,
    global_fallback_logged: set,
    detection_mode: str = "label_aware",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "frame_name": frame_name,
        "video_id": video_uid,
        "subject_id": subject_id,
        "frame_processable": True,
        "frame_has_gt": False,
        "frame_valid_for_pck": False,
        "frame_only_in_gt": frame_name not in pred_raw,
        "frame_only_in_pred": frame_name not in gt_raw,
    }

    gt_left = gt_hands["Left"] if gt_hands else None
    gt_right = gt_hands["Right"] if gt_hands else None

    if frame_name in gt_raw:
        _gl, _gr, _el, _er, gt_anomaly, gt_empty = split_hands_by_slot(
            gt_raw[frame_name], frame_w, frame_h,
            apply_out_of_frame_filter=False,
            frame_name=frame_name, is_gt=True,
        )
    else:
        gt_anomaly = False
        gt_empty = True

    pred_left, pred_right, extras_left, extras_right, _pano, _pemp = split_hands_by_slot(
        pred_raw.get(frame_name, []), frame_w, frame_h,
        apply_out_of_frame_filter=True,
        frame_name=frame_name, is_gt=False,
    )

    pred_left_list: List[KeypointList] = (
        ([pred_left] if pred_left is not None else []) + extras_left
    )
    pred_right_list: List[KeypointList] = (
        ([pred_right] if pred_right is not None else []) + extras_right
    )
    n_pred_left = len(pred_left_list)
    n_pred_right = len(pred_right_list)

    row["frame_has_gt"] = (gt_left is not None) or (gt_right is not None)
    row["gt_has_left"] = gt_left is not None
    row["gt_has_right"] = gt_right is not None
    row["gt_hands"] = int(gt_left is not None) + int(gt_right is not None)
    row["pred_has_left"] = n_pred_left > 0
    row["pred_has_right"] = n_pred_right > 0
    row["pred_hands"] = n_pred_left + n_pred_right
    row["gt_anomaly"] = gt_anomaly
    row["gt_empty"] = gt_empty

    res_l = match_slot(
        gt_left, pred_left_list, frame_w, frame_h,
        frame_name=frame_name, slot_label="Left",
    )
    res_r = match_slot(
        gt_right, pred_right_list, frame_w, frame_h,
        frame_name=frame_name, slot_label="Right",
    )
    row["det_left"] = res_l.outcome
    row["det_right"] = res_r.outcome
    row["n_phantom_left"] = res_l.n_phantoms
    row["n_phantom_right"] = res_r.n_phantoms
    row["matched_iou_left"] = res_l.matched_iou
    row["matched_iou_right"] = res_r.matched_iou
    row["duplicate_handedness"] = res_l.duplicate or res_r.duplicate
    row["gate_failed_lone_left"] = res_l.gate_failed_lone
    row["gate_failed_lone_right"] = res_r.gate_failed_lone

    pred_pool: list = []
    for kps in pred_left_list:
        pred_pool.append((0, kps))
    for kps in pred_right_list:
        pred_pool.append((1, kps))
    swaps = detect_label_swaps(
        gt_left, gt_right, pred_pool, frame_w, frame_h, frame_name=frame_name
    )
    row["label_swap_left"] = swaps.label_swap_left
    row["label_swap_right"] = swaps.label_swap_right
    row["cross_hand_left"] = swaps.cross_hand_left
    row["cross_hand_right"] = swaps.cross_hand_right

    hs_left, fb_left, fb_left_global = (None, False, False)
    hs_right, fb_right, fb_right_global = (None, False, False)
    if gt_left is not None:
        hs_left, fb_left, fb_left_global = resolve_hand_size(
            gt_left, video_uid, "Left", per_video_means, global_means, frame_name
        )
    if gt_right is not None:
        hs_right, fb_right, fb_right_global = resolve_hand_size(
            gt_right, video_uid, "Right", per_video_means, global_means, frame_name
        )

    for slot_label, fb_global in (("Left", fb_left_global), ("Right", fb_right_global)):
        if not fb_global:
            continue
        key = (video_uid, slot_label)
        if key in global_fallback_logged:
            continue
        global_fallback_logged.add(key)
        logger.warning(
            "Video %s, handedness %s: using global hand_size fallback "
            "(no per-video mean available).",
            video_uid, slot_label,
        )

    row["hand_size_left"] = hs_left if hs_left is not None else float("nan")
    row["hand_size_right"] = hs_right if hs_right is not None else float("nan")
    row["hand_size_fallback_left"] = fb_left
    row["hand_size_fallback_right"] = fb_right

    left_loc: Optional[HandLocResult] = None
    right_loc: Optional[HandLocResult] = None

    if res_l.outcome == "TP" and gt_left is not None and res_l.matched_pred is not None and hs_left:
        left_loc = localize_hand(gt_left, res_l.matched_pred, hs_left, alphas)
        vis_counter.update(gt_left, res_l.matched_pred)
    if res_r.outcome == "TP" and gt_right is not None and res_r.matched_pred is not None and hs_right:
        right_loc = localize_hand(gt_right, res_r.matched_pred, hs_right, alphas)
        vis_counter.update(gt_right, res_r.matched_pred)

    for label, loc in (("left", left_loc), ("right", right_loc)):
        for a in alphas:
            sfx = _alpha_suffix(a)
            key = f"pck_{label}_{sfx}"
            row[key] = loc.pck_per_alpha[a] if loc is not None else float("nan")
            row[f"{key}_n_correct"] = (
                loc.n_correct_per_alpha[a] if loc is not None else 0
            )
            row[f"{key}_n_eligible"] = (
                loc.n_eligible_per_alpha[a] if loc is not None else 0
            )
        row[f"mpjpe_{label}_pixels"] = loc.mpjpe_pixels if loc is not None else float("nan")
        row[f"mpjpe_{label}_normalized"] = loc.mpjpe_normalized if loc is not None else float("nan")
        if loc is not None:
            px_errs = [e for e in loc.per_kp_errors if e is not None]
            n_kp = len(px_errs)
            sum_px = float(sum(px_errs)) if n_kp else 0.0
            hs = loc.hand_size_used
            sum_norm = sum_px / float(hs) if (hs and hs > 0) else float("nan")
            row[f"mpjpe_{label}_pixels_sum"] = sum_px
            row[f"mpjpe_{label}_pixels_n_kp"] = n_kp
            row[f"mpjpe_{label}_normalized_sum"] = sum_norm
            row[f"mpjpe_{label}_normalized_n_kp"] = n_kp
        else:
            row[f"mpjpe_{label}_pixels_sum"] = 0.0
            row[f"mpjpe_{label}_pixels_n_kp"] = 0
            row[f"mpjpe_{label}_normalized_sum"] = 0.0
            row[f"mpjpe_{label}_normalized_n_kp"] = 0

    row["frame_valid_for_pck"] = (left_loc is not None) or (right_loc is not None)

    for alpha, beta in POSE_F1_PAIRS:
        sfx_ab = _pose_suffix(alpha, beta)
        for label, det_outcome, n_pred, loc in (
            ("left", res_l.outcome, n_pred_left, left_loc),
            ("right", res_r.outcome, n_pred_right, right_loc),
        ):
            tp = fp = fn = 0
            if det_outcome == "TP":
                if loc is None:
                    fn = 1
                    fp = 1 + max(0, n_pred - 1)
                else:
                    pck_val = loc.pck_per_alpha.get(alpha)
                    pck_ok = (
                        pck_val is not None
                        and not math.isnan(pck_val)
                        and pck_val >= beta
                    )
                    if pck_ok:
                        tp = 1
                        fp = max(0, n_pred - 1)
                    else:
                        fn = 1
                        fp = 1 + max(0, n_pred - 1)
            elif det_outcome == "FN":
                fn = 1
            elif det_outcome == "FP":
                fp = n_pred
            row[f"pose_tp_{label}_{sfx_ab}"] = tp
            row[f"pose_fp_{label}_{sfx_ab}"] = fp
            row[f"pose_fn_{label}_{sfx_ab}"] = fn

    # --- Label-agnostic Detection ONLY (rebuttal, --detection-mode both)
    # --------------------------------------------------------------------
    # Completely additive: only runs when explicitly requested, writes
    # ONLY new "_la"-suffixed columns, and never touches res_l/res_r,
    # left_loc/right_loc, or any pck_*/mpjpe_* column used by the
    # standard aggregates. Mirrors compute_metrics.py's implementation
    # verbatim — see metrics/detection_agnostic.py for the full matching
    # policy and the rationale for pooling FP at the frame level instead
    # of splitting it per slot.
    #
    # NOTE: a label-agnostic Pose-F1 variant was implemented and then
    # reverted after review (didn't hold up as a metric). Pose-F1 is
    # ALWAYS the original label-aware version (pose_tp/fp/fn_left/right_*,
    # pose_f1_left/right_*), regardless of --detection-mode. Only
    # Detection gets a label-agnostic variant.
    if detection_mode == "both":
        pred_pool_agnostic: List[KeypointList] = pred_left_list + pred_right_list
        agnostic = match_frame_agnostic(
            gt_left, gt_right, pred_pool_agnostic, frame_w, frame_h,
            frame_name=frame_name,
        )
        row["det_left_la"] = agnostic.left.status
        row["det_right_la"] = agnostic.right.status
        row["matched_iou_left_la"] = agnostic.left.matched_iou
        row["matched_iou_right_la"] = agnostic.right.matched_iou
        row["n_matched_la"] = agnostic.n_matched
        row["n_phantom_la"] = agnostic.n_phantom

    def _gt_base_flag(loc: Optional[HandLocResult]) -> bool:
        if loc is None:
            return False
        v = loc.pck_per_alpha.get(GT_BASE_ALPHA)
        if v is None or math.isnan(v):
            return False
        return v > 0.85

    row["GT_base_left"] = _gt_base_flag(left_loc)
    row["GT_base_right"] = _gt_base_flag(right_loc)

    if left_loc is not None and gt_left is not None and res_l.matched_pred is not None:
        v = per_frame_visibility(gt_left, res_l.matched_pred)
        row["visibility_recall_left"] = v["recall"]
        row["visibility_precision_left"] = v["precision"]
    else:
        row["visibility_recall_left"] = float("nan")
        row["visibility_precision_left"] = float("nan")
    if right_loc is not None and gt_right is not None and res_r.matched_pred is not None:
        v = per_frame_visibility(gt_right, res_r.matched_pred)
        row["visibility_recall_right"] = v["recall"]
        row["visibility_precision_right"] = v["precision"]
    else:
        row["visibility_recall_right"] = float("nan")
        row["visibility_precision_right"] = float("nan")

    cat = categorize_frame(
        frame_name=frame_name,
        gt_left=gt_left,
        gt_right=gt_right,
        hand_size_left=hs_left,
        hand_size_right=hs_right,
        frame_w=frame_w,
        frame_h=frame_h,
        frames_dir=frames_dir,
    )
    row["blur_score_image"] = cat.blur_score_image
    row["blur_score_left_hand"] = cat.blur_score_left_hand
    row["blur_score_right_hand"] = cat.blur_score_right_hand
    row["bbox_area_left_hand"] = cat.bbox_area_left_hand
    row["bbox_area_right_hand"] = cat.bbox_area_right_hand
    row["hands_close"] = cat.hands_close
    row["hand_near_edge_left"] = cat.hand_near_edge_left
    row["hand_near_edge_right"] = cat.hand_near_edge_right
    row["hand_near_face_left"] = cat.hand_near_face_left
    row["hand_near_face_right"] = cat.hand_near_face_right
    row["hand_face_distance_left"] = cat.hand_face_distance_left
    row["hand_face_distance_right"] = cat.hand_face_distance_right

    return row


# ---------------------------------------------------------------------------
# Aggregation
# (extracted verbatim from compute_metrics.py)
# ---------------------------------------------------------------------------


def _aggregate_group(
    df_group: pd.DataFrame,
    alphas: List[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n_frames_total"] = int(len(df_group))
    valid = df_group[df_group["frame_processable"] == True]  # noqa: E712
    out["n_frames_valid"] = int(len(valid))

    out["n_gt_hands_left"] = int(valid["gt_has_left"].sum()) if not valid.empty else 0
    out["n_gt_hands_right"] = int(valid["gt_has_right"].sum()) if not valid.empty else 0
    out["n_gt_hands_total"] = out["n_gt_hands_left"] + out["n_gt_hands_right"]
    out["n_frames_no_gt"] = int((valid["gt_hands"] == 0).sum()) if not valid.empty else 0
    out["n_frames_gt_anomaly"] = int(valid["gt_anomaly"].sum()) if not valid.empty else 0
    out["n_pred_hands_left"] = int(valid["pred_has_left"].sum()) if not valid.empty else 0
    out["n_pred_hands_right"] = int(valid["pred_has_right"].sum()) if not valid.empty else 0
    out["n_pred_hands_total"] = out["n_pred_hands_left"] + out["n_pred_hands_right"]
    out["n_frames_no_pred"] = int((valid["pred_hands"] == 0).sum()) if not valid.empty else 0
    out["n_frames_only_in_gt"] = int(valid["frame_only_in_gt"].sum()) if not valid.empty else 0
    out["n_frames_only_in_pred"] = int(valid["frame_only_in_pred"].sum()) if not valid.empty else 0

    for label in ("left", "right"):
        col = f"det_{label}"
        if valid.empty:
            tp = fp = fn = 0
        else:
            outcomes = valid[col].value_counts()
            tp = int(outcomes.get("TP", 0))
            fn = int(outcomes.get("FN", 0))
            fp = int(valid[f"n_phantom_{label}"].sum())
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        out[f"tp_{label}"] = tp
        out[f"fp_{label}"] = fp
        out[f"fn_{label}"] = fn
        out[f"precision_{label}"] = p
        out[f"recall_{label}"] = r
        out[f"f1_{label}"] = f1

    f1l = out["f1_left"]
    f1r = out["f1_right"]
    if math.isnan(f1l) and math.isnan(f1r):
        out["macro_f1"] = float("nan")
    elif math.isnan(f1l):
        out["macro_f1"] = f1r
    elif math.isnan(f1r):
        out["macro_f1"] = f1l
    else:
        out["macro_f1"] = (f1l + f1r) / 2.0

    tp_l = out.get("tp_left", 0)
    fp_l = out.get("fp_left", 0)
    fn_l = out.get("fn_left", 0)
    tp_r = out.get("tp_right", 0)
    fp_r = out.get("fp_right", 0)
    fn_r = out.get("fn_right", 0)
    _, _, micro_f1 = precision_recall_f1(tp_l + tp_r, fp_l + fp_r, fn_l + fn_r)
    out["micro_f1"] = micro_f1

    for a in alphas:
        sfx = _alpha_suffix(a)
        for label in ("left", "right"):
            colname = f"pck_{label}_{sfx}"
            if colname not in valid.columns or valid.empty:
                out[colname] = float("nan")
                continue
            vals = valid[colname].dropna()
            out[colname] = float(vals.mean()) if len(vals) > 0 else float("nan")

        v_l = out.get(f"pck_left_{sfx}", float("nan"))
        v_r = out.get(f"pck_right_{sfx}", float("nan"))
        if (
            isinstance(v_l, float) and math.isnan(v_l)
            or isinstance(v_r, float) and math.isnan(v_r)
        ):
            out[f"pck_both_{sfx}_macro"] = float("nan")
        else:
            out[f"pck_both_{sfx}_macro"] = (float(v_l) + float(v_r)) / 2.0

        nc_total = 0
        ne_total = 0
        for label in ("left", "right"):
            nc_col = f"pck_{label}_{sfx}_n_correct"
            ne_col = f"pck_{label}_{sfx}_n_eligible"
            if not valid.empty and nc_col in valid.columns and ne_col in valid.columns:
                nc_total += int(valid[nc_col].sum())
                ne_total += int(valid[ne_col].sum())
        out[f"pck_both_{sfx}_micro"] = (
            nc_total / ne_total if ne_total > 0 else float("nan")
        )

    for unit in ("pixels", "normalized"):
        per_hand_vals: Dict[str, List[float]] = {"left": [], "right": []}
        for label in ("left", "right"):
            colname = f"mpjpe_{label}_{unit}"
            vals = (
                valid[colname].dropna().tolist()
                if not valid.empty and colname in valid.columns
                else []
            )
            per_hand_vals[label] = vals
            out[f"mpjpe_{label}_{unit}_mean"] = _safe_mean(vals)
            out[f"mpjpe_{label}_{unit}_p50"] = _safe_percentile(vals, 50)
            out[f"mpjpe_{label}_{unit}_p90"] = _safe_percentile(vals, 90)
            out[f"mpjpe_{label}_{unit}_p95"] = _safe_percentile(vals, 95)

        for stat in ("mean", "p50", "p90", "p95"):
            l_val = out.get(f"mpjpe_left_{unit}_{stat}", float("nan"))
            r_val = out.get(f"mpjpe_right_{unit}_{stat}", float("nan"))
            if (
                isinstance(l_val, float) and math.isnan(l_val)
                or isinstance(r_val, float) and math.isnan(r_val)
            ):
                out[f"mpjpe_both_{unit}_{stat}_macro"] = float("nan")
            else:
                out[f"mpjpe_both_{unit}_{stat}_macro"] = (
                    float(l_val) + float(r_val)
                ) / 2.0

        sum_total = 0.0
        n_kp_total = 0
        for label in ("left", "right"):
            sum_col = f"mpjpe_{label}_{unit}_sum"
            n_col = f"mpjpe_{label}_{unit}_n_kp"
            if not valid.empty and sum_col in valid.columns and n_col in valid.columns:
                s_series = valid[sum_col].astype(float).fillna(0.0)
                n_series = valid[n_col].astype(float).fillna(0.0)
                mask = ~np.isnan(s_series.to_numpy())
                sum_total += float(s_series[mask].sum())
                n_kp_total += int(n_series[mask].sum())
        out[f"mpjpe_both_{unit}_mean_micro"] = (
            sum_total / n_kp_total if n_kp_total > 0 else float("nan")
        )

        pooled = list(per_hand_vals["left"]) + list(per_hand_vals["right"])
        out[f"mpjpe_both_{unit}_p50_micro"] = _safe_percentile(pooled, 50)
        out[f"mpjpe_both_{unit}_p90_micro"] = _safe_percentile(pooled, 90)
        out[f"mpjpe_both_{unit}_p95_micro"] = _safe_percentile(pooled, 95)

    if valid.empty:
        out["label_swap_sum"] = float("nan")
        out["label_swap_rate"] = float("nan")
    else:
        gt_left_frames = valid[valid["gt_has_left"] == True]   # noqa: E712
        gt_right_frames = valid[valid["gt_has_right"] == True]  # noqa: E712
        label_swap_left_sum = (
            float(gt_left_frames["label_swap_left"].sum())
            if not gt_left_frames.empty else float("nan")
        )
        label_swap_right_sum = (
            float(gt_right_frames["label_swap_right"].sum())
            if not gt_right_frames.empty else float("nan")
        )
        out["label_swap_sum"] = label_swap_left_sum + label_swap_right_sum
        out["label_swap_rate"] = (
            (label_swap_left_sum + label_swap_right_sum) / out["n_gt_hands_total"]
        )

    # Denominator: frames where GT has at least one hand present (same
    # convention as label_swap_rate / cross_hand_overlap) — see
    # compute_metrics.py for the full rationale.
    if valid.empty:
        out["duplicate_handedness_rate"] = float("nan")
    else:
        gt_present_frames = valid[
            (valid["gt_has_left"] == True) | (valid["gt_has_right"] == True)  # noqa: E712
        ]
        out["duplicate_handedness_rate"] = (
            float(gt_present_frames["duplicate_handedness"].mean())
            if not gt_present_frames.empty else float("nan")
        )

    for label in ("left", "right"):
        for kind in ("recall", "precision"):
            colname = f"visibility_{kind}_{label}"
            if valid.empty or colname not in valid.columns:
                out[f"{colname}_mean"] = float("nan")
            else:
                out[f"{colname}_mean"] = _safe_mean(valid[colname].dropna().tolist())

    if not valid.empty:
        out["n_hand_size_fallback_left"] = int(valid["hand_size_fallback_left"].sum())
        out["n_hand_size_fallback_right"] = int(valid["hand_size_fallback_right"].sum())
    else:
        out["n_hand_size_fallback_left"] = 0
        out["n_hand_size_fallback_right"] = 0

    for alpha, beta in POSE_F1_PAIRS:
        sfx_ab = _pose_suffix(alpha, beta)
        if valid.empty:
            tp_l = fp_l = fn_l = tp_r = fp_r = fn_r = 0
        else:
            tp_l = int(valid[f"pose_tp_left_{sfx_ab}"].sum())
            fp_l = int(valid[f"pose_fp_left_{sfx_ab}"].sum())
            fn_l = int(valid[f"pose_fn_left_{sfx_ab}"].sum())
            tp_r = int(valid[f"pose_tp_right_{sfx_ab}"].sum())
            fp_r = int(valid[f"pose_fp_right_{sfx_ab}"].sum())
            fn_r = int(valid[f"pose_fn_right_{sfx_ab}"].sum())

        out[f"pose_tp_left_{sfx_ab}"] = tp_l
        out[f"pose_fp_left_{sfx_ab}"] = fp_l
        out[f"pose_fn_left_{sfx_ab}"] = fn_l
        out[f"pose_tp_right_{sfx_ab}"] = tp_r
        out[f"pose_fp_right_{sfx_ab}"] = fp_r
        out[f"pose_fn_right_{sfx_ab}"] = fn_r

        p_l, r_l, f1_l = precision_recall_f1(tp_l, fp_l, fn_l)
        p_r, r_r, f1_r = precision_recall_f1(tp_r, fp_r, fn_r)
        out[f"pose_precision_left_{sfx_ab}"] = p_l
        out[f"pose_recall_left_{sfx_ab}"] = r_l
        out[f"pose_f1_left_{sfx_ab}"] = f1_l
        out[f"pose_precision_right_{sfx_ab}"] = p_r
        out[f"pose_recall_right_{sfx_ab}"] = r_r
        out[f"pose_f1_right_{sfx_ab}"] = f1_r

        p_m, r_m, f1_m = precision_recall_f1(
            tp_l + tp_r, fp_l + fp_r, fn_l + fn_r
        )
        out[f"pose_micro_precision_{sfx_ab}"] = p_m
        out[f"pose_micro_recall_{sfx_ab}"] = r_m
        out[f"pose_micro_f1_{sfx_ab}"] = f1_m

        if math.isnan(f1_l) and math.isnan(f1_r):
            macro_f1 = float("nan")
        elif math.isnan(f1_l):
            macro_f1 = f1_r
        elif math.isnan(f1_r):
            macro_f1 = f1_l
        else:
            macro_f1 = (f1_l + f1_r) / 2.0
        out[f"pose_macro_f1_{sfx_ab}"] = macro_f1

    # --- Label-agnostic Detection ONLY (rebuttal, --detection-mode both)
    # --------------------------------------------------------------------
    # Guarded purely by column presence: when the flag was off, these
    # "_la" columns never existed in the per-frame CSV, so this whole
    # block is a no-op and the output is byte-identical to before. Pooled
    # (micro) only — no left/right split, no macro. See
    # metrics/detection_agnostic.py docstring for the rationale.
    #
    # Pose-F1 has NO agnostic variant (reverted after review — see the
    # note in _process_one_frame). pose_f1_left/right_* and
    # pose_macro_f1_* above are always the original label-aware Pose-F1,
    # regardless of --detection-mode.
    if "n_matched_la" in valid.columns:
        if valid.empty:
            tp_la = fp_la = fn_la = 0
        else:
            tp_la = int(valid["n_matched_la"].sum())
            fp_la = int(valid["n_phantom_la"].sum())
            fn_la = int(valid["gt_hands"].sum()) - tp_la
        p_la, r_la, f1_la = precision_recall_f1(tp_la, fp_la, fn_la)
        out["tp_la"] = tp_la
        out["fp_la"] = fp_la
        out["fn_la"] = fn_la
        out["precision_la"] = p_la
        out["recall_la"] = r_la
        out["f1_la"] = f1_la

    pck_base_sfx = _alpha_suffix(GT_BASE_ALPHA)
    for label in ("left", "right"):
        pck_base_col = f"pck_{label}_{pck_base_sfx}"
        base_col = f"GT_base_{label}"
        if (
            valid.empty
            or pck_base_col not in valid.columns
            or base_col not in valid.columns
        ):
            out[f"GT_base_{label}_rate"] = float("nan")
            out[f"GT_base_{label}_count"] = 0
            out[f"GT_base_{label}_n_tp"] = 0
            continue
        tp_mask = valid[pck_base_col].notna()
        n_tp = int(tp_mask.sum())
        n_base = int(valid.loc[tp_mask, base_col].sum())
        out[f"GT_base_{label}_rate"] = (
            n_base / n_tp if n_tp > 0 else float("nan")
        )
        out[f"GT_base_{label}_count"] = n_base
        out[f"GT_base_{label}_n_tp"] = n_tp

    return out
