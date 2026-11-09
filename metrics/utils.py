"""Shared utilities: frame name parsing, keypoint normalization, etcs
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Dataset / MediaPipe Hands topology constants.
N_HAND_KEYPOINTS = 21
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

# IPN Hand default resolution. Override via CLI when needed.
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480

# Regex to parse names like "1CM1_1_R_#217_000024.jpg" into
#   subject_id="1CM1_1_R", video_id="#217", frame_idx=24
_FRAME_NAME_RE = re.compile(r"^(?P<subject>.+)_(?P<video>#\d+)_(?P<frame>\d+)\.jpg$")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A hand: a plain list (not array) of 21 keypoints. Each keypoint is either
# ``(x, y)`` in pixels or ``None``. We keep ``None`` explicit instead of NaN
# because GT uses it that way and it's less error-prone than mixing both.
KeypointList = List[Optional[Tuple[float, float]]]


@dataclass
class FrameId:
    """Identifiers extracted from a frame's filename."""

    frame_name: str
    subject_id: str
    video_id: str
    frame_idx: int

    @property
    def video_uid(self) -> str:
        """Unique video identifier combining subject and video.

        Required because ``video_id`` alone (e.g. ``#217``) can repeat
        across subjects. We use this combined key for per-video grouping.
        """
        return f"{self.subject_id}|{self.video_id}"


# ---------------------------------------------------------------------------
# Frame name parsing
# ---------------------------------------------------------------------------


def parse_frame_name(frame_name: str) -> Optional[FrameId]:
    """Parse ``<subject>_<#video>_<frame_idx>.jpg`` into its components.

    Args:
        frame_name: Filename, e.g. ``"1CM1_1_R_#217_000024.jpg"``.

    Returns:
        A ``FrameId`` with ``subject_id``, ``video_id`` (including the
        leading ``#``) and ``frame_idx`` as int. Returns ``None`` and logs
        a warning with the raw name if the pattern does not match.
    """
    m = _FRAME_NAME_RE.match(frame_name)
    if m is None:
        logger.warning("Could not parse frame name: %s", frame_name)
        return None
    return FrameId(
        frame_name=frame_name,
        subject_id=m.group("subject"),
        video_id=m.group("video"),
        frame_idx=int(m.group("frame")),
    )


# ---------------------------------------------------------------------------
# Keypoint normalization
# ---------------------------------------------------------------------------


def is_inside_frame(
    point: Tuple[float, float],
    frame_w: int,
    frame_h: int,
) -> bool:
    """Return ``True`` if ``(x, y)`` falls inside the frame rectangle."""
    x, y = point
    return 0 <= x < frame_w and 0 <= y < frame_h


def normalize_hand_pts(
    raw_pts: List,
    frame_w: int,
    frame_h: int,
    *,
    apply_out_of_frame_filter: bool,
) -> KeypointList:
    """Convert raw ``hand_pts`` to the canonical 21-keypoint list.

    Steps:
        1. Ensure length is 21 (log and pad with None if not).
        2. Normalize each entry to ``(x, y)`` or ``None``. Tolerates direct
           ``None`` entries and lists with ``None`` components.
        3. If ``apply_out_of_frame_filter`` is on, convert to ``None`` any
           keypoint with ``x < 0``, ``x >= frame_w``, ``y < 0`` or
           ``y >= frame_h``. Only applied to predictions; in GT, off-frame
           keypoints already come as explicit ``None``.

    Args:
        raw_pts: The raw list as parsed from JSON.
        frame_w, frame_h: Frame dimensions in pixels.
        apply_out_of_frame_filter: Toggle the off-frame filter.

    Returns:
        A 21-entry list with ``(x, y)`` or ``None``.
    """
    out: KeypointList = [None] * N_HAND_KEYPOINTS
    if not isinstance(raw_pts, list):
        logger.warning("hand_pts is not a list: %r", type(raw_pts))
        return out

    n = min(len(raw_pts), N_HAND_KEYPOINTS)
    if len(raw_pts) != N_HAND_KEYPOINTS:
        logger.warning(
            "hand_pts has %d entries, expected %d; truncating/padding.",
            len(raw_pts),
            N_HAND_KEYPOINTS,
        )

    for i in range(n):
        kp = raw_pts[i]
        if kp is None:
            continue
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        x, y = kp[0], kp[1]
        if x is None or y is None:
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if apply_out_of_frame_filter and not is_inside_frame((xf, yf), frame_w, frame_h):
            continue
        out[i] = (xf, yf)
    return out


def is_right_to_label(is_right: int) -> str:
    """Map ``is_right`` to ``"Right"`` / ``"Left"``.

    Confirmed by the dataset author: ``is_right=1`` → ``"Right"``,
    ``is_right=0`` → ``"Left"``.
    """
    return "Right" if int(is_right) == 1 else "Left"


# ---------------------------------------------------------------------------
# Hand size: dist(idx 0, idx 9), with per-video fallback
# ---------------------------------------------------------------------------


def hand_size_from_kps(kps: KeypointList) -> Optional[float]:
    """Compute ``||kp[0] - kp[9]||`` in pixels.

    Returns:
        The distance, or ``None`` if either keypoint is missing.
    """
    a = kps[WRIST_IDX]
    b = kps[MIDDLE_MCP_IDX]
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_video_hand_size_means(
    gt_hands_per_frame: Dict[str, Dict[str, KeypointList]],
    frame_to_video: Dict[str, str],
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """Pre-compute hand_size means per (video, handedness) and global.

    Iterates over all GT frames and, for every hand with both idx 0 and
    idx 9 visible, accumulates ``dist(0, 9)``. Returns two dicts: one
    indexed by ``(video_uid, handedness)`` and a global per-handedness
    fallback for videos that have no valid hand_size.

    Args:
        gt_hands_per_frame: ``{frame_name: {"Left": kps, "Right": kps}}``.
        frame_to_video: ``{frame_name: video_uid}``.

    Returns:
        ``(per_video_means, global_means)`` where:
        - ``per_video_means[(video_uid, "Left"|"Right")] = float`` (mean px).
        - ``global_means["Left"|"Right"] = float`` (mean px). Empty if there
          is no data.
    """
    per_video_acc: Dict[Tuple[str, str], List[float]] = {}
    global_acc: Dict[str, List[float]] = {"Left": [], "Right": []}

    for frame_name, hands in gt_hands_per_frame.items():
        video = frame_to_video.get(frame_name)
        if video is None:
            continue
        for hand_label in ("Left", "Right"):
            kps = hands.get(hand_label)
            if kps is None:
                continue
            hs = hand_size_from_kps(kps)
            if hs is None or hs <= 0:
                continue
            per_video_acc.setdefault((video, hand_label), []).append(hs)
            global_acc[hand_label].append(hs)

    per_video_means: Dict[Tuple[str, str], float] = {
        k: float(np.mean(v)) for k, v in per_video_acc.items() if len(v) > 0
    }
    global_means: Dict[str, float] = {
        k: float(np.mean(v)) for k, v in global_acc.items() if len(v) > 0
    }
    return per_video_means, global_means


def resolve_hand_size(
    gt_kps: Optional[KeypointList],
    video_uid: str,
    hand_label: str,
    per_video_means: Dict[Tuple[str, str], float],
    global_means: Dict[str, float],
    frame_name_for_log: str,
) -> Tuple[Optional[float], bool, bool]:
    """Resolve hand_size for a specific hand of a frame.

    Policy (see spec):
        1. If GT exists for this hand and both idx 0 and idx 9 are valid
           → ``dist(0, 9)``, fallback=False.
        2. If idx 0 or idx 9 is missing → mean for that
           (video, handedness). fallback=True.
        3. If that video has no hand_size of the same handedness →
           global mean of the same handedness.
           fallback=True, fallback_global=True.
        4. If there is no global either (degenerate case) → ``None``.

    This function does NOT emit warnings — the caller should inspect the
    ``used_global_fallback`` flag and decide whether to log (typically with
    dedup by video+handedness).

    Args:
        gt_kps: GT keypoints for this hand (or ``None`` if no GT in slot).
        video_uid: Video identifier.
        hand_label: ``"Left"`` or ``"Right"``.
        per_video_means: Pre-computed mean dict.
        global_means: Pre-computed global mean dict.
        frame_name_for_log: Reserved (unused; see note above).

    Returns:
        ``(hand_size, used_fallback, used_global_fallback)``.
    """
    if gt_kps is not None:
        hs = hand_size_from_kps(gt_kps)
        if hs is not None and hs > 0:
            return hs, False, False

    # Fallback 1: video mean.
    video_mean = per_video_means.get((video_uid, hand_label))
    if video_mean is not None and video_mean > 0:
        return video_mean, True, False

    # Fallback 2: global mean.
    global_mean = global_means.get(hand_label)
    if global_mean is not None and global_mean > 0:
        return global_mean, True, True

    return None, True, True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def visible_keypoints(kps: Optional[KeypointList]) -> List[Tuple[float, float]]:
    """Return only the non-None keypoints of a hand. Empty list if the
    hand is ``None``.
    """
    if kps is None:
        return []
    return [kp for kp in kps if kp is not None]


def centroid(kps: Optional[KeypointList]) -> Optional[Tuple[float, float]]:
    """Centroid (mean) of the visible keypoints of a hand.

    Returns ``None`` if the hand is ``None`` or has no visible keypoints.
    """
    pts = visible_keypoints(kps)
    if len(pts) == 0:
        return None
    arr = np.array(pts, dtype=float)
    cx, cy = arr.mean(axis=0)
    return float(cx), float(cy)


def euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """2D Euclidean distance between two points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def bbox_from_kps(
    kps: Optional[KeypointList],
) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box (xmin, ymin, xmax, ymax) of the visible keypoints.

    Returns ``None`` if there are no visible keypoints or if ``kps`` itself
    is ``None``.
    """
    pts = visible_keypoints(kps)
    if len(pts) == 0:
        return None
    arr = np.array(pts, dtype=float)
    xmin, ymin = arr.min(axis=0)
    xmax, ymax = arr.max(axis=0)
    return float(xmin), float(ymin), float(xmax), float(ymax)


# ---------------------------------------------------------------------------
# Bounding boxes and IoU (shared by detection gate and frame categorization)
# ---------------------------------------------------------------------------

# Minimum visible keypoints for a stable bbox (shared constant).
MIN_KPS_FOR_BBOX = 3

# Padding fraction applied per axis of the bbox (15% of each side).
BBOX_PADDING_FRAC = 0.15


def padded_bbox(
    kps: Optional[KeypointList],
    frame_w: int,
    frame_h: int,
    *,
    frac: float = BBOX_PADDING_FRAC,
    min_kps: int = MIN_KPS_FOR_BBOX,
) -> Optional[Tuple[float, float, float, float]]:
    """Padded, frame-clamped bbox of a hand's visible keypoints.

    Single source of truth for hand bounding boxes used both by the IoU
    detection gate and by per-hand frame categorization, so GT and
    prediction bboxes are derived identically (symmetric IoU).

    Steps:
        1. Require at least ``min_kps`` visible keypoints; else ``None``.
        2. Tight bbox via ``bbox_from_kps``.
        3. Pad by ``frac`` of each axis (width/height) per side.
        4. Clamp to [0, frame_w] x [0, frame_h].

    Returns ``(xmin, ymin, xmax, ymax)`` as floats, or ``None`` if the
    hand is missing / has fewer than ``min_kps`` visible keypoints / the
    padded box collapses.
    """
    pts = visible_keypoints(kps)
    if len(pts) < min_kps:
        return None
    tight = bbox_from_kps(kps)
    if tight is None:
        return None
    xmin, ymin, xmax, ymax = tight
    w = xmax - xmin
    h = ymax - ymin
    pad_x = frac * w
    pad_y = frac * h
    x0 = max(0.0, xmin - pad_x)
    y0 = max(0.0, ymin - pad_y)
    x1 = min(float(frame_w), xmax + pad_x)
    y1 = min(float(frame_h), ymax + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def iou(
    box_a: Optional[Tuple[float, float, float, float]],
    box_b: Optional[Tuple[float, float, float, float]],
) -> float:
    """Intersection-over-union of two ``(xmin, ymin, xmax, ymax)`` boxes.

    Returns 0.0 if either box is ``None`` or if there is no overlap.
    """
    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def bbox_center(
    box: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Center ``(cx, cy)`` of a ``(xmin, ymin, xmax, ymax)`` box."""
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def bbox_diag(box: Tuple[float, float, float, float]) -> float:
    """Diagonal length of a ``(xmin, ymin, xmax, ymax)`` box."""
    x0, y0, x1, y1 = box
    return math.hypot(x1 - x0, y1 - y0)


def normalized_center_distance(
    box_a: Optional[Tuple[float, float, float, float]],
    box_b: Optional[Tuple[float, float, float, float]],
) -> float:
    """Center distance normalized by the largest bbox diagonal.

    Smaller means the centers are closer. Returns ``inf`` if either box
    is ``None`` or the scale is non-positive (so it never satisfies a
    "<= threshold" test by accident).
    """
    if box_a is None or box_b is None:
        return math.inf
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    dist = math.hypot(ax - bx, ay - by)
    scale = max(bbox_diag(box_a), bbox_diag(box_b))
    if scale <= 0:
        return math.inf
    return dist / scale
