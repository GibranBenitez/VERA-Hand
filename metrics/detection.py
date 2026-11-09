"""Per-slot detection (Left/Right): TP / FP / FN / TN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .utils import KeypointList, centroid, euclidean, iou, padded_bbox

logger = logging.getLogger(__name__)

# IoU gate threshold. A prediction must have IoU strictly greater than
# this against the GT bbox to be eligible as a match.
TH_IOU = 0.2


@dataclass
class SlotMatchResult:
    """Result of slot matching for a single (Left or Right) slot in a frame.

    Attributes:
        outcome: ``"TP"`` | ``"FP"`` | ``"FN"`` | ``"TN"``.
        matched_pred: Keypoints of the matched prediction (only when TP).
        n_phantoms: Number of pooled predictions counted as FP (those not
            matched, including ones that failed the IoU gate).
        duplicate: ``True`` when pred had 2+ hands of the same handedness.
        matched_iou: IoU of the matched prediction against the GT bbox
            (NaN when no match / not applicable). Useful for auditing.
        gate_failed_lone: ``True`` when GT was present and the whole pool
            (>=1 prediction) failed the gate, turning an otherwise-TP slot
            into FN + FP. Used for the "ex-TP to FN" impact log.
    """

    outcome: str
    matched_pred: Optional[KeypointList]
    n_phantoms: int
    duplicate: bool
    matched_iou: float = float("nan")
    gate_failed_lone: bool = False


def match_slot(
    gt_kps: Optional[KeypointList],
    pred_kps_list: List[KeypointList],
    frame_w: int,
    frame_h: int,
    *,
    frame_name: str = "",
    slot_label: str = "",
) -> SlotMatchResult:
    """Resolve the slot assignment between GT and predictions with an IoU gate.

    Args:
        gt_kps: GT keypoints in this slot. ``None`` if GT is absent.
        pred_kps_list: List (>=0 entries) of predictions that share the
            same handedness as this slot.
        frame_w, frame_h: Frame dimensions (for padded bbox / clamping).
        frame_name, slot_label: For log messages only.

    Returns:
        A ``SlotMatchResult``.
    """
    n_pred = len(pred_kps_list)
    has_gt = gt_kps is not None
    duplicate = n_pred >= 2

    # GT absent.
    if not has_gt:
        if n_pred == 0:
            return SlotMatchResult(
                outcome="TN", matched_pred=None, n_phantoms=0, duplicate=False
            )
        return SlotMatchResult(
            outcome="FP", matched_pred=None, n_phantoms=n_pred, duplicate=duplicate
        )

    # GT present, no predictions → FN.
    if n_pred == 0:
        return SlotMatchResult(
            outcome="FN", matched_pred=None, n_phantoms=0, duplicate=False
        )

    # GT present with at least one prediction: apply the IoU gate.
    gt_box = padded_bbox(gt_kps, frame_w, frame_h)
    if gt_box is None:
        # GT present but bbox unstable (<3 visible keypoints). This is a
        # serious data problem: we cannot build the reference bbox, so we
        # cannot evaluate the gate. Warn and treat the slot as FN with all
        # predictions as FP (no unverified TP).
        logger.warning(
            "Frame %s slot=%s: GT present but has <3 visible keypoints; "
            "cannot build reference bbox. Slot treated as FN, %d "
            "prediction(s) as FP.",
            frame_name, slot_label, n_pred,
        )
        return SlotMatchResult(
            outcome="FN", matched_pred=None, n_phantoms=n_pred,
            duplicate=duplicate, gate_failed_lone=True,
        )

    # IoU of each pooled prediction against the GT bbox.
    ious: List[float] = []
    for pred_kps in pred_kps_list:
        pred_box = padded_bbox(pred_kps, frame_w, frame_h)
        if pred_box is None:
            # Prediction with <3 visible keypoints: unstable bbox, cannot
            # pass the gate. Debug-only (common for poor estimators).
            logger.debug(
                "Frame %s slot=%s: a prediction has <3 visible keypoints; "
                "its bbox is unstable and cannot pass the IoU gate (FP).",
                frame_name, slot_label,
            )
            ious.append(float("nan"))
        else:
            ious.append(iou(gt_box, pred_box))

    # Apply the gate: indices with IoU strictly above the threshold.
    eligible = [
        i for i, v in enumerate(ious)
        if (v == v) and v > TH_IOU  # v==v filters NaN
    ]

    if not eligible:
        # No prediction passes the gate → FN, every prediction is FP.
        return SlotMatchResult(
            outcome="FN", matched_pred=None, n_phantoms=n_pred,
            duplicate=duplicate, gate_failed_lone=True,
        )

    # Pick the eligible prediction with the highest IoU.
    max_iou = max(ious[i] for i in eligible)
    top = [i for i in eligible if ious[i] == max_iou]

    if len(top) > 1:
        # Exact IoU tie (rare): break by closest centroid, deterministic.
        logger.debug(
            "Frame %s slot=%s: %d predictions tie at IoU=%.4f; breaking "
            "the tie by closest centroid.",
            frame_name, slot_label, len(top), max_iou,
        )
        gt_c = centroid(gt_kps)
        if gt_c is not None:
            best_idx = top[0]
            best_dist = float("inf")
            for i in top:
                c = centroid(pred_kps_list[i])
                if c is None:
                    continue
                d = euclidean(gt_c, c)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        else:
            best_idx = top[0]
    else:
        best_idx = top[0]

    return SlotMatchResult(
        outcome="TP",
        matched_pred=pred_kps_list[best_idx],
        n_phantoms=n_pred - 1,
        duplicate=duplicate,
        matched_iou=ious[best_idx],
    )


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute (precision, recall, F1) from TP/FP/FN counts.

    Returns ``float('nan')`` for precision/recall/F1 when the corresponding
    denominator is zero (no data to define the metric). The caller decides
    how to render that.
    """
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if (p != p) or (r != r) or (p + r) == 0:  # NaN or both zero
        f1 = float("nan")
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1
