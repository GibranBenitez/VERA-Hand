"""Frame categorization: motion blur
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .utils import (
    BBOX_PADDING_FRAC,
    KeypointList,
    MIN_KPS_FOR_BBOX,
    WRIST_IDX,
    bbox_from_kps,
    centroid,
    padded_bbox,
    visible_keypoints,
)

logger = logging.getLogger(__name__)

# cv2 is optional; if it's missing, blur columns are NaN.
try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False
    logger.debug(
        "OpenCV (cv2) is not installed; blur scores will be reported as NaN."
    )


# Bbox padding for per-hand blur computation. These now alias the shared
# constants in utils so the blur crop and the IoU detection gate derive
# hand bboxes identically (single source of truth).
_BBOX_PADDING_FRAC = BBOX_PADDING_FRAC

# Minimum number of visible GT keypoints required for a stable bbox.
_MIN_KPS_FOR_BBOX = MIN_KPS_FOR_BBOX


@dataclass
class FrameCategorization:
    """Full categorization output for a frame."""

    blur_score_image: float = float("nan")
    blur_score_left_hand: float = float("nan")
    blur_score_right_hand: float = float("nan")
    bbox_area_left_hand: float = float("nan")  # pre-padding, px²
    bbox_area_right_hand: float = float("nan")
    hands_close: float = float("nan")
    hand_near_edge_left: float = float("nan")
    hand_near_edge_right: float = float("nan")
    # TODO(hand_near_face): requires --auxiliary-json with pose landmarks
    # (nose = pose idx 0). Will be populated once that JSON is available.
    hand_near_face_left: float = float("nan")
    hand_near_face_right: float = float("nan")
    hand_face_distance_left: float = float("nan")
    hand_face_distance_right: float = float("nan")

    # Audit flags for warnings.
    _bbox_left_unstable: bool = field(default=False, repr=False)
    _bbox_right_unstable: bool = field(default=False, repr=False)
    _frame_load_failed: bool = field(default=False, repr=False)


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------


def _laplacian_variance(gray: "np.ndarray") -> float:
    """Laplacian variance of a grayscale image."""
    if not _HAS_CV2:
        return float("nan")
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _blur_for_hand(
    gray: "np.ndarray",
    gt_kps: KeypointList,
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, bool]:
    """Blur score and pre-padding bbox area for one hand.

    Returns:
        ``(blur_score, bbox_area_pre_padding, unstable_flag)``.
        If <3 visible keypoints, ``blur_score=NaN`` and ``unstable_flag=True``.
    """
    visible = visible_keypoints(gt_kps)
    if len(visible) < _MIN_KPS_FOR_BBOX:
        return float("nan"), float("nan"), True
    bbox = bbox_from_kps(gt_kps)
    if bbox is None:
        return float("nan"), float("nan"), True
    xmin, ymin, xmax, ymax = bbox
    pre_area = max(0.0, (xmax - xmin) * (ymax - ymin))

    if not _HAS_CV2:
        return float("nan"), pre_area, False
    padded = padded_bbox(gt_kps, frame_w, frame_h)
    if padded is None:
        return float("nan"), pre_area, True
    x0, y0, x1, y1 = (int(round(v)) for v in padded)
    if x1 <= x0 or y1 <= y0:
        return float("nan"), pre_area, True
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return float("nan"), pre_area, True
    return _laplacian_variance(crop), pre_area, False


def categorize_frame(
    frame_name: str,
    gt_left: Optional[KeypointList],
    gt_right: Optional[KeypointList],
    hand_size_left: Optional[float],
    hand_size_right: Optional[float],
    frame_w: int,
    frame_h: int,
    frames_dir: Optional[str],
) -> FrameCategorization:
    """Compute every categorization metric for a frame.

    Args:
        frame_name: Filename (used to locate the RGB image).
        gt_left, gt_right: GT per slot.
        hand_size_left, hand_size_right: Per-hand hand_size (with fallback
            already resolved). In pixels.
        frame_w, frame_h: Frame dimensions.
        frames_dir: Path to the RGB frames directory. ``None`` disables
            blur scores. If the directory exists but a specific frame is
            missing, blur scores stay NaN and a warning is logged.

    Returns:
        ``FrameCategorization`` with every column we can compute.
    """
    cat = FrameCategorization()

    # --- hands_close ----------------------------------------------------
    if gt_left is not None and gt_right is not None:
        wl = gt_left[WRIST_IDX]
        wr = gt_right[WRIST_IDX]
        if wl is not None and wr is not None:
            d = math.hypot(wl[0] - wr[0], wl[1] - wr[1])
            hs_avg = None
            if hand_size_left and hand_size_right:
                hs_avg = (hand_size_left + hand_size_right) / 2.0
            elif hand_size_left:
                hs_avg = hand_size_left
            elif hand_size_right:
                hs_avg = hand_size_right
            if hs_avg and hs_avg > 0:
                cat.hands_close = d / hs_avg

    # --- hand_near_edge -------------------------------------------------
    half_min_side = min(frame_w, frame_h) / 2.0
    for label, kps in (("left", gt_left), ("right", gt_right)):
        c = centroid(kps)
        if c is None:
            continue
        cx, cy = c
        d_edge = min(cx, cy, frame_w - cx, frame_h - cy)
        if half_min_side > 0:
            # Larger value = closer to an edge.
            value = max(0.0, 1.0 - d_edge / half_min_side)
            setattr(cat, f"hand_near_edge_{label}", value)

    # --- blur -----------------------------------------------------------
    if frames_dir is None or not _HAS_CV2:
        # No frames_dir or no cv2: leave all blur metrics as NaN.
        return cat

    img_path = os.path.join(frames_dir, frame_name)
    if not os.path.isfile(img_path):
        cat._frame_load_failed = True
        logger.warning("RGB frame not found: %s", img_path)
        return cat

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        cat._frame_load_failed = True
        logger.warning("Could not read frame: %s", img_path)
        return cat

    h_real, w_real = img.shape[:2]
    if (w_real, h_real) != (frame_w, frame_h):
        logger.debug(
            "Frame %s has dims (%d,%d); using actual dimensions.",
            frame_name,
            w_real,
            h_real,
        )
        frame_w, frame_h = w_real, h_real

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cat.blur_score_image = _laplacian_variance(gray)

    if gt_left is not None:
        b, area, unstable = _blur_for_hand(gray, gt_left, frame_w, frame_h)
        cat.blur_score_left_hand = b
        cat.bbox_area_left_hand = area
        cat._bbox_left_unstable = unstable
        if unstable:
            logger.warning(
                "Frame %s: Left bbox unstable (<%d visible keypoints).",
                frame_name,
                _MIN_KPS_FOR_BBOX,
            )
    if gt_right is not None:
        b, area, unstable = _blur_for_hand(gray, gt_right, frame_w, frame_h)
        cat.blur_score_right_hand = b
        cat.bbox_area_right_hand = area
        cat._bbox_right_unstable = unstable
        if unstable:
            logger.warning(
                "Frame %s: Right bbox unstable (<%d visible keypoints).",
                frame_name,
                _MIN_KPS_FOR_BBOX,
            )

    return cat
