"""Per-keypoint visibility recall and precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .utils import N_HAND_KEYPOINTS, KeypointList


@dataclass
class VisibilityCounters:
    """Per-keypoint accumulators for visibility.

    For each keypoint i in [0, 20]:
      - tp_vis: both visible (does not feed precision/recall directly but
        is useful for diagnostics).
      - both_none: both ``None`` (numerator of both metrics).
      - gt_none_pred_visible: GT=None, pred visible (recall denominator).
      - gt_visible_pred_none: GT visible, pred=None (precision denominator).
    """

    tp_vis: List[int] = field(default_factory=lambda: [0] * N_HAND_KEYPOINTS)
    both_none: List[int] = field(default_factory=lambda: [0] * N_HAND_KEYPOINTS)
    gt_none_pred_visible: List[int] = field(
        default_factory=lambda: [0] * N_HAND_KEYPOINTS
    )
    gt_visible_pred_none: List[int] = field(
        default_factory=lambda: [0] * N_HAND_KEYPOINTS
    )

    def update(self, gt_kps: KeypointList, pred_kps: KeypointList) -> None:
        """Update accumulators with a TP pair of hands."""
        for i in range(N_HAND_KEYPOINTS):
            g = gt_kps[i] if i < len(gt_kps) else None
            p = pred_kps[i] if i < len(pred_kps) else None
            if g is None and p is None:
                self.both_none[i] += 1
            elif g is None and p is not None:
                self.gt_none_pred_visible[i] += 1
            elif g is not None and p is None:
                self.gt_visible_pred_none[i] += 1
            else:
                self.tp_vis[i] += 1

    def recall_per_kp(self) -> List[float]:
        """``both_none / (both_none + gt_none_pred_visible)`` per keypoint.
        ``NaN`` if denominator is 0.
        """
        out = []
        for i in range(N_HAND_KEYPOINTS):
            denom = self.both_none[i] + self.gt_none_pred_visible[i]
            out.append(self.both_none[i] / denom if denom > 0 else float("nan"))
        return out

    def precision_per_kp(self) -> List[float]:
        """``both_none / (both_none + gt_visible_pred_none)`` per keypoint.
        ``NaN`` if denominator is 0.
        """
        out = []
        for i in range(N_HAND_KEYPOINTS):
            denom = self.both_none[i] + self.gt_visible_pred_none[i]
            out.append(self.both_none[i] / denom if denom > 0 else float("nan"))
        return out


def per_frame_visibility(
    gt_kps: KeypointList, pred_kps: KeypointList
) -> Dict[str, float]:
    """Per-hand recall and precision aggregated across the 21 keypoints,
    intended for the per-frame CSV.

    Returns:
        Dict with two keys: ``"recall"`` and ``"precision"``. Each in
        [0, 1] or ``NaN`` if not applicable for this frame.
    """
    both_none = 0
    gt_none_pred_visible = 0
    gt_visible_pred_none = 0
    for i in range(N_HAND_KEYPOINTS):
        g = gt_kps[i] if i < len(gt_kps) else None
        p = pred_kps[i] if i < len(pred_kps) else None
        if g is None and p is None:
            both_none += 1
        elif g is None and p is not None:
            gt_none_pred_visible += 1
        elif g is not None and p is None:
            gt_visible_pred_none += 1

    rec_denom = both_none + gt_none_pred_visible
    prec_denom = both_none + gt_visible_pred_none
    return {
        "recall": both_none / rec_denom if rec_denom > 0 else float("nan"),
        "precision": both_none / prec_denom if prec_denom > 0 else float("nan"),
    }
