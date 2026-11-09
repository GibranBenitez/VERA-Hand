"""Localization: PCK and MPJPE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils import KeypointList


@dataclass
class HandLocResult:
    """Localization result for one TP hand in a frame.

    Attributes:
        per_kp_errors: 21-entry list. ``float`` for the Euclidean distance
            in pixels when both GT and pred are visible for that keypoint;
            ``None`` when GT is None OR when pred is None. The latter case
            is distinguished via ``per_kp_pred_missing``, so the PCK/MPJPE
            policies can differ.
        per_kp_pred_missing: 21-entry list of bool. ``True`` when GT is
            present but pred is None (counts as PCK failure but is excluded
            from MPJPE).
        hand_size_used: hand_size used to normalize PCK (pixels).
        pck_per_alpha: ``{alpha: pck}``. Only counts keypoints with GT
            present. ``NaN`` if no keypoints have GT.
        mpjpe_pixels: MPJPE in pixels, averaged over keypoints where both
            GT and pred exist. ``NaN`` if there are none.
        mpjpe_normalized: ``mpjpe_pixels / hand_size_used``.
    """

    per_kp_errors: List[Optional[float]] = field(default_factory=list)
    per_kp_pred_missing: List[bool] = field(default_factory=list)
    hand_size_used: float = float("nan")
    pck_per_alpha: dict = field(default_factory=dict)
    # Per-alpha keypoint counts for true micro aggregation downstream:
    # n_correct = #{kp eligible AND error <= alpha * hand_size}
    # n_eligible = #{kp with GT present (incl. those where pred is None)}
    n_correct_per_alpha: dict = field(default_factory=dict)
    n_eligible_per_alpha: dict = field(default_factory=dict)
    mpjpe_pixels: float = float("nan")
    mpjpe_normalized: float = float("nan")


def compute_per_keypoint_errors(
    gt_kps: KeypointList,
    pred_kps: KeypointList,
) -> Tuple[List[Optional[float]], List[bool]]:
    """Per-keypoint distances and a pred-missing flag.

    Returns:
        Tuple ``(errors, pred_missing)`` both of length 21:
        - ``errors[i] = None`` if GT[i] is None.
        - ``errors[i] = None`` if GT[i] is present but pred[i] is None
          (in which case ``pred_missing[i] = True``).
        - ``errors[i] = float`` Euclidean distance when both exist.
    """
    errors: List[Optional[float]] = [None] * len(gt_kps)
    pred_missing: List[bool] = [False] * len(gt_kps)
    for i in range(len(gt_kps)):
        g = gt_kps[i]
        if g is None:
            continue
        p = pred_kps[i] if i < len(pred_kps) else None
        if p is None:
            pred_missing[i] = True
            continue
        errors[i] = math.hypot(g[0] - p[0], g[1] - p[1])
    return errors, pred_missing


def pck_with_counts_from_errors(
    errors: Sequence[Optional[float]],
    pred_missing: Sequence[bool],
    hand_size: float,
    alphas: Sequence[float],
) -> Tuple[dict, dict, dict]:
    """Like ``pck_from_errors`` but also returns per-alpha keypoint counts.

    Returns:
        Tuple ``(pck_dict, n_correct_dict, n_eligible_dict)`` keyed by alpha.
        ``n_eligible`` is the count of keypoints where GT is present (which
        includes the cases where pred is None and they count as failure).
        ``n_correct`` is the count of those that satisfy
        ``error <= alpha * hand_size``.
    """
    pck = {a: float("nan") for a in alphas}
    n_correct = {a: 0 for a in alphas}
    n_eligible = {a: 0 for a in alphas}

    if not (hand_size and hand_size > 0):
        return pck, n_correct, n_eligible

    eligible_idx = [
        i for i, e in enumerate(errors)
        if (e is not None) or pred_missing[i]
    ]
    n_elig = len(eligible_idx)
    if n_elig == 0:
        return pck, n_correct, n_eligible

    for a in alphas:
        thr = a * hand_size
        nc = 0
        for i in eligible_idx:
            if pred_missing[i]:
                continue
            if errors[i] is not None and errors[i] <= thr:
                nc += 1
        n_correct[a] = nc
        n_eligible[a] = n_elig
        pck[a] = nc / n_elig
    return pck, n_correct, n_eligible


def pck_from_errors(
    errors: Sequence[Optional[float]],
    pred_missing: Sequence[bool],
    hand_size: float,
    alphas: Sequence[float],
) -> dict:
    """PCK per threshold over keypoints with GT present.

    Policy:
        - Keypoints with GT=None are excluded from the computation.
        - Keypoints with GT present and pred=None count as failure (1 to
          the denominator, 0 to the numerator).
        - Keypoints with both present count as a hit when
          ``error <= alpha * hand_size``.

    Args:
        errors: Output of ``compute_per_keypoint_errors``.
        pred_missing: Same output.
        hand_size: In pixels. Must be > 0.
        alphas: Threshold list (e.g. ``[0.02, 0.05, 0.1, 0.2]``).

    Returns:
        ``{alpha: pck}`` (each PCK in [0, 1] or ``NaN`` if no keypoints).
    """
    pck, _, _ = pck_with_counts_from_errors(errors, pred_missing, hand_size, alphas)
    return pck


def mpjpe_from_errors(errors: Sequence[Optional[float]]) -> float:
    """MPJPE in pixels over keypoints where both GT and pred exist.

    Returns ``NaN`` if there is no valid error.
    """
    valid = [e for e in errors if e is not None]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


def localize_hand(
    gt_kps: KeypointList,
    pred_kps: KeypointList,
    hand_size: float,
    alphas: Sequence[float],
) -> HandLocResult:
    """Full localization pipeline for one TP hand."""
    errors, pred_missing = compute_per_keypoint_errors(gt_kps, pred_kps)
    pck, n_correct, n_eligible = pck_with_counts_from_errors(
        errors, pred_missing, hand_size, alphas
    )
    mpjpe_px = mpjpe_from_errors(errors)
    mpjpe_norm = (
        mpjpe_px / hand_size if (hand_size and hand_size > 0 and not math.isnan(mpjpe_px))
        else float("nan")
    )
    return HandLocResult(
        per_kp_errors=list(errors),
        per_kp_pred_missing=list(pred_missing),
        hand_size_used=hand_size,
        pck_per_alpha=pck,
        n_correct_per_alpha=n_correct,
        n_eligible_per_alpha=n_eligible,
        mpjpe_pixels=mpjpe_px,
        mpjpe_normalized=mpjpe_norm,
    )


def pck_pool_both_hands_with_counts(
    left_result: Optional[HandLocResult],
    right_result: Optional[HandLocResult],
    alphas: Sequence[float],
) -> Tuple[dict, dict, dict]:
    """Like ``pck_pool_both_hands`` but also returns keypoint counts.

    Returns:
        Tuple ``(pck_dict, n_correct_dict, n_eligible_dict)``. All zeros if
        either hand is missing or has no eligible keypoints.
    """
    pck = {a: float("nan") for a in alphas}
    n_correct = {a: 0 for a in alphas}
    n_eligible = {a: 0 for a in alphas}

    if left_result is None or right_result is None:
        return pck, n_correct, n_eligible

    items = []
    for h in (left_result, right_result):
        hs = h.hand_size_used
        for e, pm in zip(h.per_kp_errors, h.per_kp_pred_missing):
            items.append((e, pm, hs))

    eligible = [(e, pm, hs) for e, pm, hs in items if (e is not None) or pm]
    n_elig = len(eligible)
    if n_elig == 0:
        return pck, n_correct, n_eligible

    for a in alphas:
        nc = 0
        for e, pm, hs in eligible:
            if pm or (hs is None) or (hs <= 0):
                continue
            if e is not None and e <= a * hs:
                nc += 1
        n_correct[a] = nc
        n_eligible[a] = n_elig
        pck[a] = nc / n_elig
    return pck, n_correct, n_eligible


def pck_pool_both_hands(
    left_result: Optional[HandLocResult],
    right_result: Optional[HandLocResult],
    alphas: Sequence[float],
) -> dict:
    """"Both" PCK by pooling keypoints across both hands.

    Only defined when BOTH hands have a ``HandLocResult`` (both TP). If any
    is missing, returns ``{a: NaN for a in alphas}``.

    For the pool, each keypoint is normalized by its own hand_size (left vs
    right may differ). Mixing absolute pixel distances under different
    hand_sizes would be incorrect. The effective rule is:
    ``error_i <= alpha * hand_size_i``.
    """
    pck, _, _ = pck_pool_both_hands_with_counts(left_result, right_result, alphas)
    return pck


def mpjpe_pool_both_hands(
    left_result: Optional[HandLocResult],
    right_result: Optional[HandLocResult],
) -> Tuple[float, float]:
    """"Both" MPJPE: pool of pixel and normalized errors.

    For the normalized variant, each error is divided by its own hand_size
    before averaging (consistent with ``pck_pool_both_hands``).

    Returns:
        ``(mpjpe_pixels, mpjpe_normalized)``. ``NaN`` if not computable.
    """
    if left_result is None or right_result is None:
        return float("nan"), float("nan")

    px_errors: List[float] = []
    norm_errors: List[float] = []
    for h in (left_result, right_result):
        hs = h.hand_size_used
        for e in h.per_kp_errors:
            if e is None:
                continue
            px_errors.append(e)
            if hs and hs > 0:
                norm_errors.append(e / hs)

    if len(px_errors) == 0:
        return float("nan"), float("nan")
    mpjpe_px = float(np.mean(px_errors))
    mpjpe_norm = float(np.mean(norm_errors)) if norm_errors else float("nan")
    return mpjpe_px, mpjpe_norm
