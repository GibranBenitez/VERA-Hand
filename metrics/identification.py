"""Handedness identification: per-slot label swap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .utils import (
    KeypointList,
    centroid,
    euclidean,
    iou,
    padded_bbox,
)
from .detection import TH_IOU

logger = logging.getLogger(__name__)


@dataclass
class LabelSwapFlags:
    """Per-slot label-swap and cross-hand booleans.

    Only meaningful where the slot's GT is present.

    A slot is ``cross_hand`` when the IoU-gated pool against that GT slot
    contains predictions of BOTH handedness labels (>=1 Right and >=1
    Left). That is an ambiguous overlap, distinct from a clean label
    swap, so a cross-hand slot does NOT also count as a label swap.
    """

    label_swap_left: bool
    label_swap_right: bool
    cross_hand_left: bool
    cross_hand_right: bool


def _geometric_match_label(
    gt_kps: Optional[KeypointList],
    pred_pool: List[Tuple[int, KeypointList]],
    frame_w: int,
    frame_h: int,
    *,
    frame_name: str,
    slot_label: str,
) -> Tuple[Optional[int], bool]:
    """Geometric match for ONE GT slot against a handedness-agnostic pool.

    ``pred_pool`` is a list of ``(is_right, kps)`` for every prediction in
    the frame. Returns ``(matched_label, cross_hand)`` where:
      - ``matched_label`` is the ``is_right`` label (0/1) of the
        highest-IoU gated prediction, or ``None`` if GT is absent /
        unstable / no prediction passes the IoU gate.
      - ``cross_hand`` is ``True`` when the gated pool contains BOTH
        handedness labels (>=1 Right and >=1 Left), i.e. an ambiguous
        overlap rather than a clean swap.

    Mirrors detection rules 3-4 (gate + highest IoU, centroid tie-break),
    but is completely independent of the detection matching.
    """
    if gt_kps is None:
        return None, False
    gt_box = padded_bbox(gt_kps, frame_w, frame_h)
    if gt_box is None:
        # GT <3 visible keypoints already warned about in detection; here
        # we just can't compute the swap.
        return None, False
    if not pred_pool:
        return None, False

    ious: List[float] = []
    for _is_right, kps in pred_pool:
        pred_box = padded_bbox(kps, frame_w, frame_h)
        ious.append(iou(gt_box, pred_box) if pred_box is not None else float("nan"))

    eligible = [i for i, v in enumerate(ious) if (v == v) and v > TH_IOU]
    if not eligible:
        return None, False

    # cross_hand: the gated pool spans both handedness labels.
    gated_labels = {int(pred_pool[i][0]) for i in eligible}
    cross_hand = (0 in gated_labels) and (1 in gated_labels)

    max_iou = max(ious[i] for i in eligible)
    top = [i for i in eligible if ious[i] == max_iou]
    if len(top) > 1:
        logger.debug(
            "Frame %s slot=%s (label-swap matching): %d predictions tie at "
            "IoU=%.4f; breaking by closest centroid.",
            frame_name, slot_label, len(top), max_iou,
        )
        gt_c = centroid(gt_kps)
        if gt_c is not None:
            best_idx = top[0]
            best_dist = float("inf")
            for i in top:
                c = centroid(pred_pool[i][1])
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

    return int(pred_pool[best_idx][0]), cross_hand


def detect_label_swaps(
    gt_left: Optional[KeypointList],
    gt_right: Optional[KeypointList],
    pred_pool: List[Tuple[int, KeypointList]],
    frame_w: int,
    frame_h: int,
    *,
    frame_name: str = "",
) -> LabelSwapFlags:
    """Per-slot label swap and cross-hand via independent geometric matching.

    Args:
        gt_left, gt_right: GT per slot (``None`` if absent).
        pred_pool: ``(is_right, kps)`` for EVERY prediction in the frame
            (both handedness labels), so a crossed label can be detected.
        frame_w, frame_h: Frame dimensions.

    Returns:
        ``LabelSwapFlags``. For each slot whose GT is present:
          - ``cross_hand_<slot>`` is True iff the IoU-gated pool against
            that GT slot spans BOTH handedness labels.
          - ``label_swap_<slot>`` is True iff (NOT cross_hand) AND the
            highest-IoU gated match has the opposite handedness label.
        A cross-hand slot never also counts as a label swap; the two
        slots are evaluated independently.
    """
    swap_left = False
    swap_right = False
    cross_left = False
    cross_right = False

    # GT-Left slot: a geometric match labeled Right (is_right==1) is a swap,
    # unless the gated pool was cross-hand (both labels present).
    if gt_left is not None:
        matched_label, cross_left = _geometric_match_label(
            gt_left, pred_pool, frame_w, frame_h,
            frame_name=frame_name, slot_label="Left",
        )
        if not cross_left and matched_label == 1:
            swap_left = True

    # GT-Right slot: a geometric match labeled Left (is_right==0) is a swap,
    # unless the gated pool was cross-hand.
    if gt_right is not None:
        matched_label, cross_right = _geometric_match_label(
            gt_right, pred_pool, frame_w, frame_h,
            frame_name=frame_name, slot_label="Right",
        )
        if not cross_right and matched_label == 0:
            swap_right = True

    return LabelSwapFlags(
        label_swap_left=swap_left,
        label_swap_right=swap_right,
        cross_hand_left=cross_left,
        cross_hand_right=cross_right,
    )
