#!/usr/bin/env python3
"""vera_supervised.py — Release evaluation script for VERA-Hand (WACV 2027).

Runs the supervised evaluation protocol over one or more prediction JSONs
and produces three output CSVs:

    <name>_general.csv   — Det-F1, PCK, Pose-F1, Swap, cross-hand overlap,
                           duplicate hands, MPJPE  (one row per estimator)
    <name>_blur.csv      — Det-F1 and Pose-F1 × {sharp, medium, blurry}
                           (emitted only when --blur-tags is provided)
    <name>_region.csv    — PCK × {wrist, palm, fingertips, whole}

Where ``<name>`` is the JSON stem (single-file mode) or the folder name
(batch mode). Raw and clean predictions are treated identically — this
script is agnostic to the estimator source.

Usage (single file)
-------------------
    python vera_supervised.py \\
        --preds-path examples/RTMW-wb_raw.json \\
        --gt-path annotations/vera_hand_gt.json \\
        --out-dir outputs

Usage (batch — folder of prediction JSONs, one per estimator)
-------------------------------------------------------------
    python vera_supervised.py \\
        --preds-path predictions/bench_frames/ \\
        --gt-path annotations/vera_hand_gt.json \\
        --blur-tags annotations/blur_labels.csv \\
        --out-dir outputs

Optional flags
--------------
    --alpha         PCK / Pose-F1 alpha threshold (default: 0.10)
    --beta          Pose-F1 beta threshold (default: 0.50)
    --hand          Hand side to aggregate: left | right | both  (default: both)
    --blur-tags     Path to the CSV with blur_class per (frame, hand side),
                    i.e. annotations/blur_labels.csv.
                    If omitted, the blur CSV is not produced.
    --breakdown     Add precision & recall for Det-F1 and Pose-F1,
                    and both forms of MPJPE (normalized + pixels).
                    Default: OFF (only F1 and PCK).
    --detection-mode  label_agnostic (default) | label_aware.
                    label_agnostic: det_f1/det_precision/det_recall come from
                    a pooled match that ignores the predicted side (wrong-side
                    errors are left to Identification). label_aware: they come
                    from the handedness-aware matching, respecting --hand.
                    pose_f1 (and precision/recall), PCK, region CSV, MPJPE,
                    swap_rate and cross_hand_overlap are never affected.
    --only-gt-frames  Opt-in, default OFF. Excludes frames present in
                    --preds-path but absent from --gt-path entirely
                    (not scored as FP).
    --out-dir       Output directory (default: outputs)
    --log-level     DEBUG | INFO | WARNING | ERROR (default: INFO)

Cross-hand overlap note
-----------------------
cross_hand_overlap = total_cross / n_gt_hands_total
where total_cross comes from hand_frame_stats.compute_stats-style geometry
applied to the prediction JSON, and n_gt_hands_total comes from the
dataset-level aggregation of the GT DataFrame.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — self-contained within this repository
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from metrics.pipeline import (  # noqa: E402
    POSE_F1_PAIRS,
    _aggregate_group,
    _alpha_suffix,
    _pose_suffix,
    process_all_frames,
    valid_points,
    bbox_from_points,
    bbox_iou,
    normalized_center_distance,
    TH_IOU,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    normalize_hand_pts,
    is_right_to_label,
    padded_bbox,
    compute_per_keypoint_errors,
    match_slot,
)

logger = logging.getLogger("eval_supervised")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GT_PATH = str(_THIS_DIR / "annotations" / "vera_hand_gt.json")
DEFAULT_ALPHA = 0.10
DEFAULT_BETA = 0.50

# Cross-hand thresholds (same defaults as hand_frame_stats.py CLI)
_CROSS_TH_IOU = 0.80
_CROSS_TH_CD = 0.15

# Anatomical regions for the region CSV
# (subset of hand_region_analysis._REGIONS)
_REGIONS: Dict[str, tuple] = {
    "wrist":      (0,),
    "palm":       (0, 5, 9, 13, 17),
    "fingertips": (4, 8, 12, 16, 20),
    "whole":      tuple(range(21)),
}

_BLUR_CLASSES = ("sharp", "medium", "blurry")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--preds-path", required=True,
        help="Path to a prediction JSON file or a directory of JSON files.",
    )
    p.add_argument(
        "--gt-path", required=True,
        help="Path to the GT JSON.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="PCK / Pose-F1 alpha threshold (default: 0.10).",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Pose-F1 beta threshold (default: 0.50).",
    )
    p.add_argument(
        "--hand",
        choices=["left", "right", "both"],
        default="both",
        help="Hand side to aggregate (default: both = macro L/R).",
    )
    p.add_argument(
        "--blur-tags",
        default=None,
        help="CSV with blur_class per (frame_name, slot). "
             "Use annotations/blur_labels.csv. "
             "If omitted, the blur CSV is not produced.",
    )
    p.add_argument(
        "--breakdown",
        action="store_true",
        default=False,
        help="Include precision & recall alongside F1s, and both MPJPE "
             "forms (normalized + pixels).",
    )
    p.add_argument(
        "--detection-mode",
        choices=["label_aware", "label_agnostic"],
        default="label_agnostic",
        help="label_agnostic (default):"
             "DET columns are instead populated from the label-agnostic "
             "match (a hand is counted found regardless of predicted "
             "side; wrong-side errors are left entirely to "
             "Identification, not double counted here). The agnostic "
             "match is pooled by design (no left/right split), so --hand "
             "is ignored for det_f1/det_precision/det_recall specifically "
             "when this mode is active. pose_f1/pose_precision/"
             "pose_recall, PCK/region CSV/MPJPE, and Identification "
             "(swap_rate/cross_hand_overlap) are NEVER affected by this "
             "label_aware (R1 default): unchanged behavior — det_f1/"
             "det_precision/det_recall come from the current handedness-"
             "aware matching, respecting --hand."
    )
    p.add_argument(
        "--only-gt-frames",
        action="store_true",
        default=False,
        help="Opt-in, default OFF (unchanged behavior when omitted). When "
             "set, restrict frame coverage to frames present in --gt-path "
             "only: frames that exist in --preds-path but NOT in --gt-path "
             "are excluded entirely (not scored as FP, not scored at all). "
             "Use this when --preds-path was built against a newer/larger "
             "GT revision than the --gt-path you are intentionally "
             "evaluating against.",
    )
    p.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory (default: outputs).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_pred_paths(pred_arg: str) -> List[Path]:
    """Return sorted list of JSON paths from a file or directory."""
    p = Path(pred_arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        paths = sorted(p.glob("*.json"))
        if not paths:
            raise SystemExit(f"No JSON files found in directory: {p}")
        return paths
    raise FileNotFoundError(f"pred argument is not a file or directory: {p}")


def _output_stem(pred_arg: str, pred_paths: List[Path]) -> str:
    """Derive the CSV name stem from the input."""
    p = Path(pred_arg)
    if p.is_file():
        return p.stem
    # Directory: use folder name
    return p.resolve().name


# ---------------------------------------------------------------------------
# Cross-hand overlap (thin wrapper — no frames_dir required)
# ---------------------------------------------------------------------------


def _compute_cross_hand(
    pred_raw: Dict[str, Any],
    gt_frame_names: frozenset,
    th_iou: float = _CROSS_TH_IOU,
    th_cd: float = _CROSS_TH_CD,
) -> int:
    """Count cross-hand prediction pairs in the GT-scoped pred JSON.

    Iterates over frames that exist in the GT (gt_frame_names) and counts
    (right, left) prediction pairs where IoU >= th_iou AND normalized
    center distance <= th_cd.  Uses the geometry helpers from
    hand_frame_stats (bbox_iou, normalized_center_distance, etc.) without
    requiring access to image files.

    Returns:
        total_cross: number of qualifying (right, left) pairs found.
    """
    import math as _math

    total_cross = 0

    for frame_name in gt_frame_names:
        hands = pred_raw.get(frame_name)
        if not isinstance(hands, list):
            continue

        right_boxes = []
        left_boxes = []

        for hand in hands:
            if not isinstance(hand, dict):
                continue
            is_right = hand.get("is_right")
            hand_pts = hand.get("hand_pts", [])
            pts = valid_points(hand_pts)
            box = bbox_from_points(pts)
            if box is None:
                continue
            if is_right == 1:
                right_boxes.append(box)
            elif is_right == 0:
                left_boxes.append(box)

        for rbox in right_boxes:
            for lbox in left_boxes:
                iou_val = bbox_iou(rbox, lbox)
                cd_val = normalized_center_distance(rbox, lbox)
                if (
                    iou_val >= th_iou
                    and _math.isfinite(cd_val)
                    and cd_val <= th_cd
                ):
                    total_cross += 1

    return total_cross


# ---------------------------------------------------------------------------
# Region PCK (per-keypoint, region-aggregated)
# ---------------------------------------------------------------------------


def _compute_region_pck(
    df: pd.DataFrame,
    gt_raw: Dict[str, Any],
    pred_raw: Dict[str, Any],
    frame_w: int,
    frame_h: int,
    alpha: float,
    hand: str,
) -> Dict[str, float]:
    """Compute micro PCK@alpha for each anatomical region.

    Mirrors the logic of ``analysis/hand_region_analysis.py``:
      - For each TP (frame, slot), take the FIRST prediction of that slot
        (same as ``_split_first_by_handedness``).
      - Call ``compute_per_keypoint_errors`` on the (gt, pred) pair.
      - Only accumulate keypoints where ``errors_px[i] is not None``
        (i.e. BOTH GT and pred visible).  Keypoints where pred is None
        are excluded — they do NOT count as failures here (unlike the
        main PCK in ``compute_metrics.py`` which treats them as
        failures).  This matches the canonical region-analysis convention.

    Args:
        df:       Per-frame DataFrame from process_all_frames().
        gt_raw:   GT JSON dict.
        pred_raw: Prediction JSON dict.
        frame_w, frame_h: Frame dimensions in pixels.
        alpha:    PCK threshold.
        hand:     "left", "right", or "both".

    Returns:
        Dict mapping region name → PCK float (NaN if no eligible kps).
    """
    slots = (
        ["left", "right"] if hand == "both"
        else [hand]
    )

    # Accumulators: region → (n_correct, n_eligible)
    acc: Dict[str, List[int]] = {r: [0, 0] for r in _REGIONS}

    valid = df[df["frame_processable"] == True]  # noqa: E712

    for slot in slots:
        det_col = f"det_{slot}"
        hs_col = f"hand_size_{slot}"
        tp_frames = valid[valid[det_col] == "TP"]

        for _, row in tp_frames.iterrows():
            frame_name = row["frame_name"]
            hand_size = row[hs_col]
            if pd.isna(hand_size) or hand_size <= 0:
                continue

            gt_frame = gt_raw.get(frame_name, [])
            pred_frame = pred_raw.get(frame_name, [])

            # Use first pred of this slot — matches hand_region_analysis.py.
            gt_by_slot = _first_hand_per_slot(gt_frame, frame_w, frame_h, False)
            pred_by_slot = _first_hand_per_slot(pred_frame, frame_w, frame_h, True)

            gt_kps = gt_by_slot.get(slot)
            pred_kps = pred_by_slot.get(slot)
            if gt_kps is None or pred_kps is None:
                continue

            errors_px, _pred_missing = compute_per_keypoint_errors(
                gt_kps, pred_kps
            )

            # Accumulate per region.
            # Policy: only include keypoints where errors_px[i] is not None
            # (both GT and pred visible).  Pred-missing kps are excluded
            # rather than counted as failures — matching hand_region_analysis.
            thr = alpha * hand_size
            for region, kp_indices in _REGIONS.items():
                for i in kp_indices:
                    if i >= len(errors_px):
                        continue
                    e = errors_px[i]
                    if e is None:
                        continue  # GT absent OR pred missing — skip
                    acc[region][1] += 1  # n_eligible
                    if e <= thr:
                        acc[region][0] += 1  # n_correct

    result = {}
    for region, (nc, ne) in acc.items():
        result[region] = nc / ne if ne > 0 else float("nan")
    return result


def _first_hand_per_slot(
    raw_frame: list,
    frame_w: int,
    frame_h: int,
    apply_out_of_frame: bool,
) -> Dict[str, Optional[list]]:
    """Return the first hand of each slot, normalized."""
    out: Dict[str, Optional[list]] = {"left": None, "right": None}
    if not isinstance(raw_frame, list):
        return out
    for hand in raw_frame:
        if not isinstance(hand, dict):
            continue
        ir = hand.get("is_right")
        if ir is None:
            continue
        label = is_right_to_label(ir).lower()
        if label not in out or out[label] is not None:
            continue
        kps = normalize_hand_pts(
            hand.get("hand_pts", []), frame_w, frame_h,
            apply_out_of_frame_filter=apply_out_of_frame,
        )
        out[label] = kps
    return out


def _pred_list_for_slot(
    raw_frame: list,
    slot: str,
    frame_w: int,
    frame_h: int,
) -> List[list]:
    """All predictions of the given slot (is_right matching), normalized."""
    if not isinstance(raw_frame, list):
        return []
    out = []
    for hand in raw_frame:
        if not isinstance(hand, dict):
            continue
        ir = hand.get("is_right")
        if ir is None:
            continue
        if is_right_to_label(ir).lower() != slot:
            continue
        kps = normalize_hand_pts(
            hand.get("hand_pts", []), frame_w, frame_h,
            apply_out_of_frame_filter=True,
        )
        out.append(kps)
    return out


# ---------------------------------------------------------------------------
# Blur stratification
# ---------------------------------------------------------------------------


def _load_blur_tags(blur_csv: str) -> Dict[Tuple[str, str], str]:
    """Load blur CSV into a (frame_name, slot) -> blur_class dict."""
    df = pd.read_csv(blur_csv)
    required = ["frame_name", "slot", "blur_class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Blur CSV is missing columns {missing}. "
            f"Pass annotations/blur_labels.csv."
        )
    # Normalise slot to lowercase
    df["slot"] = df["slot"].str.lower().str.strip()
    df["blur_class"] = df["blur_class"].str.strip()
    lookup = {}
    for _, row in df.iterrows():
        key = (str(row["frame_name"]).strip(), row["slot"])
        lookup[key] = row["blur_class"]
    return lookup


def _assign_frame_blur_class(
    frame_name: str,
    blur_lookup: Dict[Tuple[str, str], str],
    gt_has_left: bool,
    gt_has_right: bool,
) -> Optional[str]:
    """Assign blur class to a frame using per-slot classes.

    Policy (from blur_conditional_analysis.py):
    - A frame's class is the MOST DEGRADED of its GT hands
      (severity: blurry > medium > sharp).
    - Frames without GT in either hand are excluded (return None).
    """
    severity = {"blurry": 2, "medium": 1, "sharp": 0}
    classes = []
    if gt_has_left:
        c = blur_lookup.get((frame_name, "left"), "")
        if c in severity:
            classes.append(c)
    if gt_has_right:
        c = blur_lookup.get((frame_name, "right"), "")
        if c in severity:
            classes.append(c)
    if not classes:
        return None
    return max(classes, key=lambda x: severity[x])


# ---------------------------------------------------------------------------
# Row building helpers
# ---------------------------------------------------------------------------


def _get_val(agg: Dict[str, Any], key: str) -> Any:
    return agg.get(key, float("nan"))


def _build_general_row(
    estimator: str,
    agg: Dict[str, Any],
    alpha: float,
    beta: float,
    hand: str,
    cross_hand_overlap: float,
    desglose: bool,
    detection_mode: str = "label_aware",
) -> Dict[str, Any]:
    """Build one row for the general CSV.

    ``detection_mode``: "label_aware" (default) computes det_f1/det_precision/
    det_recall from the handedness-aware match, respecting ``hand``.
    "label_agnostic" populates those SAME columns from the pooled
    label-agnostic match instead (``hand`` is ignored for THESE columns —
    the agnostic match has no left/right split by design). Pose-F1
    (pose_f1/pose_precision/pose_recall), PCK, and MPJPE are ALWAYS
    label-aware and always respect ``hand``, regardless of this flag — a
    label-agnostic Pose-F1 variant was tried and reverted after review.
    """
    sfx = _alpha_suffix(alpha)
    sfx_ab = _pose_suffix(alpha, beta)
    row: Dict[str, Any] = {"estimator": estimator}

    if detection_mode == "label_agnostic":
        # Pooled label-agnostic Detection: single precision/recall/f1,
        # no left/right split (see metrics/detection_agnostic.py).
        row["det_f1"] = _get_val(agg, "f1_la")
        if desglose:
            row["det_precision"] = _get_val(agg, "precision_la")
            row["det_recall"] = _get_val(agg, "recall_la")
    elif hand == "both":
        row["det_f1"] = _get_val(agg, "macro_f1")
        if desglose:
            # Macro precision and recall: average of L/R
            p_l = _get_val(agg, "precision_left")
            p_r = _get_val(agg, "precision_right")
            r_l = _get_val(agg, "recall_left")
            r_r = _get_val(agg, "recall_right")
            row["det_precision"] = (
                (p_l + p_r) / 2
                if not (math.isnan(p_l) or math.isnan(p_r))
                else (p_r if math.isnan(p_l) else p_l)
            )
            row["det_recall"] = (
                (r_l + r_r) / 2
                if not (math.isnan(r_l) or math.isnan(r_r))
                else (r_r if math.isnan(r_l) else r_l)
            )
    else:
        row["det_f1"] = _get_val(agg, f"f1_{hand}")
        if desglose:
            row["det_precision"] = _get_val(agg, f"precision_{hand}")
            row["det_recall"] = _get_val(agg, f"recall_{hand}")

    # PCK — always label-aware (Localization is untouched by detection_mode).
    if hand == "both":
        row["pck"] = _get_val(agg, f"pck_both_{sfx}_macro")
    else:
        row["pck"] = _get_val(agg, f"pck_{hand}_{sfx}")

    # Pose-F1 — ALWAYS label-aware, regardless of detection_mode. A
    # label-agnostic Pose-F1 variant was tried and reverted after review
    # (didn't hold up as a metric); only Detection gets a label-agnostic
    # variant. --hand IS respected here even under label_agnostic mode
    # (unlike det_f1/det_precision/det_recall above, which are always
    # pooled under label_agnostic).
    if hand == "both":
        row["pose_f1"] = _get_val(agg, f"pose_macro_f1_{sfx_ab}")
        if desglose:
            p_l = _get_val(agg, f"pose_precision_left_{sfx_ab}")
            p_r = _get_val(agg, f"pose_precision_right_{sfx_ab}")
            r_l = _get_val(agg, f"pose_recall_left_{sfx_ab}")
            r_r = _get_val(agg, f"pose_recall_right_{sfx_ab}")
            row["pose_precision"] = (
                (p_l + p_r) / 2
                if not (math.isnan(p_l) or math.isnan(p_r))
                else (p_r if math.isnan(p_l) else p_l)
            )
            row["pose_recall"] = (
                (r_l + r_r) / 2
                if not (math.isnan(r_l) or math.isnan(r_r))
                else (r_r if math.isnan(r_l) else r_l)
            )
    else:
        row["pose_f1"] = _get_val(agg, f"pose_f1_{hand}_{sfx_ab}")
        if desglose:
            row["pose_precision"] = _get_val(agg, f"pose_precision_{hand}_{sfx_ab}")
            row["pose_recall"] = _get_val(agg, f"pose_recall_{hand}_{sfx_ab}")

    # Swap rate
    row["swap_rate"] = _get_val(agg, "label_swap_rate")

    # Cross-hand overlap
    row["cross_hand_overlap"] = cross_hand_overlap

    # Duplicate hands rate
    row["duplicate_hands_rate"] = _get_val(agg, "duplicate_handedness_rate")

    # MPJPE (macro: mean of per-frame means, not pooled keypoints)
    if hand == "both":
        row["mpjpe"] = _get_val(agg, "mpjpe_both_normalized_mean_macro")
        if desglose:
            row["mpjpe_normalized"] = row.pop("mpjpe")
            row["mpjpe_pixels"] = _get_val(agg, "mpjpe_both_pixels_mean_macro")
    else:
        row["mpjpe"] = _get_val(agg, f"mpjpe_{hand}_normalized_mean")
        if desglose:
            row["mpjpe_normalized"] = row.pop("mpjpe")
            row["mpjpe_pixels"] = _get_val(agg, f"mpjpe_{hand}_pixels_mean")

    return row


def _build_blur_row(
    estimator: str,
    agg_by_class: Dict[str, Dict[str, Any]],
    alpha: float,
    beta: float,
    hand: str,
    detection_mode: str = "label_aware",
) -> Dict[str, Any]:
    """Build one row for the blur CSV. See ``_build_general_row`` for the
    ``detection_mode`` column-reuse contract."""
    sfx_ab = _pose_suffix(alpha, beta)
    row: Dict[str, Any] = {"estimator": estimator}

    # Column order: sharp_det_f1, medium_det_f1, blurry_det_f1,
    #               sharp_pose_f1, medium_pose_f1, blurry_pose_f1
    for cls in _BLUR_CLASSES:
        agg = agg_by_class.get(cls, {})
        if detection_mode == "label_agnostic":
            row[f"{cls}_det_f1"] = _get_val(agg, "f1_la")
        elif hand == "both":
            row[f"{cls}_det_f1"] = _get_val(agg, "macro_f1")
        else:
            row[f"{cls}_det_f1"] = _get_val(agg, f"f1_{hand}")
    # Pose-F1 is ALWAYS label-aware here too (see _build_general_row note).
    for cls in _BLUR_CLASSES:
        agg = agg_by_class.get(cls, {})
        if hand == "both":
            row[f"{cls}_pose_f1"] = _get_val(agg, f"pose_macro_f1_{sfx_ab}")
        else:
            row[f"{cls}_pose_f1"] = _get_val(agg, f"pose_f1_{hand}_{sfx_ab}")

    return row


def _build_region_row(
    estimator: str,
    region_pck: Dict[str, float],
) -> Dict[str, Any]:
    """Build one row for the region CSV."""
    row: Dict[str, Any] = {"estimator": estimator}
    for region in _REGIONS:
        row[f"pck_{region}"] = region_pck.get(region, float("nan"))
    return row


# ---------------------------------------------------------------------------
# Per-estimator evaluation
# ---------------------------------------------------------------------------


def _evaluate_one(
    pred_path: Path,
    gt_raw: Dict[str, Any],
    *,
    frame_h: int,
    frame_w: int,
    alpha: float,
    beta: float,
    hand: str,
    blur_lookup: Optional[Dict[Tuple[str, str], str]],
    desglose: bool,
    frames_dir: Optional[str],
    detection_mode: str = "label_aware",
    only_gt_frames: bool = False,
) -> Tuple[
    Dict[str, Any],              # general row
    Optional[Dict[str, Any]],   # blur row (None if no blur_lookup)
    Dict[str, Any],              # region row
]:
    estimator = pred_path.stem
    logger.info("Evaluating: %s", estimator)

    pred_raw = _load_json(str(pred_path))

    # Ensure the alpha/beta pair we'll report is computed by process_all_frames.
    # process_all_frames uses the POSE_F1_PAIRS constant from compute_metrics.
    # We need our alpha to be in the alphas list too.
    base_alphas = sorted({a for a, _ in POSE_F1_PAIRS} | {alpha})

    # Translate our 2-choice CLI flag into pipeline.process_all_frames's
    # internal "label_aware"/"both" convention: label_agnostic still needs
    # the label-aware match computed internally (Identification/PCK depend
    # on it), it's only the REPORTED det/pose-f1 columns that switch.
    pipeline_mode = "both" if detection_mode == "label_agnostic" else "label_aware"

    df, _vis, _summary = process_all_frames(
        gt_raw,
        pred_raw,
        frame_w=frame_w,
        frame_h=frame_h,
        alphas=base_alphas,
        frames_dir=frames_dir,
        progress_every=999_999,  # suppress progress log noise
        detection_mode=pipeline_mode,
        only_gt_frames=only_gt_frames,
    )

    agg = _aggregate_group(df, base_alphas)

    # Cross-hand overlap
    gt_frame_names = frozenset(gt_raw.keys())
    total_cross = _compute_cross_hand(pred_raw, gt_frame_names)
    n_gt_hands = agg.get("n_gt_hands_total", 0)
    cross_rate = total_cross / n_gt_hands if n_gt_hands > 0 else float("nan")

    # General row
    general_row = _build_general_row(
        estimator, agg, alpha, beta, hand, cross_rate, desglose,
        detection_mode=detection_mode,
    )

    # Blur row
    blur_row: Optional[Dict[str, Any]] = None
    if blur_lookup is not None:
        # Add frame-level blur class to the DataFrame.
        df["_blur_class"] = df.apply(
            lambda r: _assign_frame_blur_class(
                r["frame_name"],
                blur_lookup,
                bool(r.get("gt_has_left", False)),
                bool(r.get("gt_has_right", False)),
            ),
            axis=1,
        )
        agg_by_class: Dict[str, Dict[str, Any]] = {}
        for cls in _BLUR_CLASSES:
            subset = df[df["_blur_class"] == cls]
            agg_by_class[cls] = _aggregate_group(subset, base_alphas)
        blur_row = _build_blur_row(
            estimator, agg_by_class, alpha, beta, hand,
            detection_mode=detection_mode,
        )

    # Region row
    region_pck = _compute_region_pck(
        df, gt_raw, pred_raw, frame_w, frame_h, alpha, hand
    )
    region_row = _build_region_row(estimator, region_pck)

    return general_row, blur_row, region_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    args = build_arg_parser().parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Collect inputs
    pred_paths = _collect_pred_paths(args.preds_path)
    stem = _output_stem(args.preds_path, pred_paths)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH
    logger.info(
        "Running eval_supervised: %d estimator(s), alpha=%.2f, beta=%.2f, "
        "hand=%s, breakdown=%s, detection-mode=%s, only-gt-frames=%s",
        len(pred_paths), args.alpha, args.beta, args.hand, args.breakdown,
        args.detection_mode, args.only_gt_frames,
    )
    if args.detection_mode == "label_agnostic" and args.hand != "both":
        logger.warning(
            "--hand=%s is ignored for det_f1/det_precision/det_recall "
            "under --detection-mode label_agnostic: the agnostic match "
            "is pooled by design (no left/right split), so those three "
            "columns will report the single pooled value regardless of "
            "--hand. pose_f1/pose_precision/pose_recall (and PCK) are "
            "UNAFFECTED by --detection-mode and will still respect "
            "--hand=%s as usual.",
            args.hand, args.hand,
        )

    # Load GT once
    if not Path(args.gt_path).is_file():
        raise FileNotFoundError(f"GT JSON not found: {args.gt_path}")
    gt_raw = _load_json(args.gt_path)
    logger.info("GT loaded: %d frames.", len(gt_raw))

    # Load blur lookup once (optional)
    blur_lookup: Optional[Dict[Tuple[str, str], str]] = None
    if args.blur_tags:
        if not Path(args.blur_tags).is_file():
            raise FileNotFoundError(f"Blur CSV not found: {args.blur_tags}")
        blur_lookup = _load_blur_tags(args.blur_tags)
        logger.info("Blur tags loaded: %d (frame, slot) entries.", len(blur_lookup))

    general_rows: List[Dict[str, Any]] = []
    blur_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []

    for pred_path in pred_paths:
        gen, blr, reg = _evaluate_one(
            pred_path,
            gt_raw,
            frame_h=h,
            frame_w=w,
            alpha=args.alpha,
            beta=args.beta,
            hand=args.hand,
            blur_lookup=blur_lookup,
            desglose=args.breakdown,
            frames_dir=None,
            detection_mode=args.detection_mode,
            only_gt_frames=args.only_gt_frames,
        )
        general_rows.append(gen)
        if blr is not None:
            blur_rows.append(blr)
        region_rows.append(reg)

    # Write CSVs  (hand slot is embedded in every filename). Under
    # label_agnostic mode, --hand STILL matters for pose_f1/PCK (only
    # det_f1/det_precision/det_recall are pooled), so the tag keeps
    # --hand and appends "-agnostic" — this also avoids silently
    # overwriting a label_aware run's CSVs with a label_agnostic run's
    # CSVs for the same --hand in the same --out-dir.
    hand_tag = (
        f"{args.hand}-agnostic" if args.detection_mode == "label_agnostic"
        else args.hand
    )
    out_general = out_dir / f"{stem}_general-{hand_tag}.csv"
    out_region = out_dir / f"{stem}_region-{hand_tag}.csv"
    pd.DataFrame(general_rows).to_csv(out_general, index=False)
    pd.DataFrame(region_rows).to_csv(out_region, index=False)
    print(f"Written: {out_general}")
    print(f"Written: {out_region}")

    if blur_rows:
        out_blur = out_dir / f"{stem}_blur-{hand_tag}.csv"
        pd.DataFrame(blur_rows).to_csv(out_blur, index=False)
        print(f"Written: {out_blur}")
    elif args.blur_tags:
        logger.warning(
            "Blur tags provided but all blur rows were empty -- check that "
            "frame_name and slot values in the blur CSV match the GT JSON."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
