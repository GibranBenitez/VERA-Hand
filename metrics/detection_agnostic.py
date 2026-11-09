"""Label-agnostic detection matching (rebuttal Round 2, WACV 2027).

Companion to ``detection.py``'s ``match_slot`` (label-aware). This module
implements a SECOND, independent matching policy that never looks at a
prediction's ``is_right`` label: it matches the (up to 2) GT hands of a
frame against a SINGLE combined pool of every prediction in the frame,
one-to-one, purely by geometry.

Purpose (rebuttal context): today, a hand that is well-localized but
labeled with the wrong side is penalized twice — once in Detection
(label-aware matching never even considers it a candidate) and again in
Identification (label_swap). Label-agnostic detection decouples "did we
find the hand" from "did we get the side right": the latter stays fully
the job of ``identification.py``, untouched by this module.

Matching policy:
1. Bbox / IoU gate: identical to ``detection.py`` — ``padded_bbox`` (15%
   padding, >=3 visible keypoints) and ``TH_IOU = 0.2`` (strict >).
2. GLOBAL one-to-one assignment per frame: at most 2 GT slots (Left,
   Right), each may claim at most one prediction from the combined pool;
   each prediction may be claimed by at most one GT slot. Since there
   are never more than 2 GT slots, every candidate assignment (none, one
   slot matched, both slots matched with two DIFFERENT predictions) is
   enumerated explicitly rather than run through a general-purpose
   solver — the problem is too small to need one, and explicit
   enumeration is easy to audit for a reviewer-facing metric.
3. Selection priority among candidate assignments, in order:
     a. Maximum number of matched GT slots.
     b. Maximum total IoU (sum over matched slots).
     c. Minimum total centroid distance (sum over matched slots).
     d. Deterministic final tie-break (enumeration order — stable given
        the same input, no randomness).
4. ``is_right`` is never read by this module. The caller is responsible
   for building the combined pool WITHOUT its per-hand label attached
   (a plain ``List[KeypointList]``), which structurally guarantees the
   label cannot influence the assignment.

What this module does NOT do:
- It does not compute PCK / MPJPE. Localization stays 100% on the
  label-aware match (``detection.py`` + ``localization.py``), per the
  explicit Paso 1 instruction. Callers that need agnostic PCK (e.g. for
  the label-agnostic Pose-F1 variant) call ``metrics.localization``
  themselves using the ``matched_pred`` this module returns, and must
  NOT write that PCK into the standard ``pck_*`` / ``mpjpe_*`` columns.
- It does not touch identification (label_swap / cross_hand). That
  logic in ``identification.py`` is completely independent and stays as
  is.
- Frame-level FP ("phantoms"): predictions left unmatched after the
  best assignment. Because the pool is shared across both GT slots (not
  pre-partitioned by label the way ``detection.py``'s caller does), an
  unmatched prediction cannot be meaningfully attributed to "the Left
  slot" or "the Right slot" without re-introducing a dependency on its
  own claimed label — which would violate the label-agnostic contract.
  Phantom counting is therefore reported at the FRAME level, not
  per-slot. This is also why the aggregate label-agnostic Detection
  metric is a single pooled (micro) Precision/Recall/F1, not split by
  handedness — the whole point of this mode is that handedness should
  not gate a match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .detection import TH_IOU
from .utils import KeypointList, centroid, euclidean, iou, padded_bbox

_SLOTS = ("left", "right")


@dataclass
class AgnosticSlotMatch:
    """Outcome of the label-agnostic match for ONE GT slot.

    Attributes:
        matched: True iff this GT slot was assigned a prediction.
        matched_pred: Keypoints of the matched prediction (only when
            ``matched`` is True).
        matched_iou: IoU of the matched prediction against this GT slot's
            bbox (NaN when not applicable).
        status: ``"TP"`` (matched), ``"FN"`` (GT present, no match found),
            or ``"NONE"`` (GT absent for this slot — nothing to score).
    """

    matched: bool
    matched_pred: Optional[KeypointList]
    matched_iou: float
    status: str


@dataclass
class AgnosticFrameMatchResult:
    """Label-agnostic matching result for one frame.

    Attributes:
        left, right: Per-GT-slot outcome (see ``AgnosticSlotMatch``).
        n_matched: Total GT slots matched this frame (0, 1, or 2) — the
            frame's label-agnostic Detection TP count.
        n_phantom: Predictions in the combined pool left unmatched — the
            frame's label-agnostic Detection FP count (frame-level, see
            module docstring for why this cannot be split per slot).
    """

    left: AgnosticSlotMatch
    right: AgnosticSlotMatch
    n_matched: int
    n_phantom: int


def match_frame_agnostic(
    gt_left: Optional[KeypointList],
    gt_right: Optional[KeypointList],
    pred_pool: List[KeypointList],
    frame_w: int,
    frame_h: int,
    *,
    frame_name: str = "",
) -> AgnosticFrameMatchResult:
    """Match up to 2 GT hands against a combined, label-blind prediction pool.

    Args:
        gt_left, gt_right: GT keypoints per slot (``None`` if absent).
            This is GT's OWN label — always authoritative, never in
            question. What changes vs. label-aware is only which
            predictions are ALLOWED to match against it.
        pred_pool: Every prediction in the frame, both handedness labels
            combined, as plain keypoint lists (no label attached).
        frame_w, frame_h: Frame dimensions (for padded bbox / clamping).
        frame_name: For potential future logging; unused today (no
            per-frame warnings emitted here — mirrors detection.py's
            silence on the common/benign cases).

    Returns:
        An ``AgnosticFrameMatchResult``.
    """
    n_pred = len(pred_pool)

    gt_by_slot: Dict[str, Optional[KeypointList]] = {"left": gt_left, "right": gt_right}
    gt_boxes: Dict[str, Optional[Tuple[float, float, float, float]]] = {}
    gt_centroids: Dict[str, Optional[Tuple[float, float]]] = {}
    for slot in _SLOTS:
        gt_kps = gt_by_slot[slot]
        if gt_kps is None:
            gt_boxes[slot] = None
            gt_centroids[slot] = None
            continue
        gt_boxes[slot] = padded_bbox(gt_kps, frame_w, frame_h)
        gt_centroids[slot] = centroid(gt_kps)

    pred_boxes = [padded_bbox(p, frame_w, frame_h) for p in pred_pool]
    pred_centroids = [centroid(p) for p in pred_pool]

    # Eligible (pred_idx, iou) candidates per slot: IoU strictly > TH_IOU,
    # identical gate to the label-aware path.
    candidates: Dict[str, List[Tuple[int, float]]] = {"left": [], "right": []}
    for slot in _SLOTS:
        gt_box = gt_boxes[slot]
        if gt_box is None:
            continue  # GT absent, or <3 visible keypoints (unstable bbox)
        for j, pbox in enumerate(pred_boxes):
            if pbox is None:
                continue  # prediction with <3 visible keypoints
            v = iou(gt_box, pbox)
            if v > TH_IOU:
                candidates[slot].append((j, v))

    def _dist(slot: str, j: int) -> float:
        gc = gt_centroids[slot]
        pc = pred_centroids[j]
        if gc is None or pc is None:
            return 0.0
        return euclidean(gc, pc)

    # Enumerate every candidate GLOBAL assignment: {}, {left: j}, {right: k},
    # {left: j, right: k} (j != k). With at most 2 GT slots this is always
    # small (bounded by the number of eligible predictions per slot).
    Assignment = Dict[str, int]
    best_assignment: Assignment = {}
    best_key: Optional[Tuple[int, float, float, int]] = None
    enum_counter = 0  # deterministic final tie-break: first-seen wins

    def _consider(assignment: Assignment) -> None:
        nonlocal best_assignment, best_key, enum_counter
        n_matched = len(assignment)
        total_iou = sum(v for slot, j in assignment.items()
                         for (jj, v) in candidates[slot] if jj == j)
        total_dist = sum(_dist(slot, j) for slot, j in assignment.items())
        # Sortable key: maximize n_matched, maximize total_iou, minimize
        # total_dist, then first-enumerated wins (stable/deterministic).
        key = (n_matched, total_iou, -total_dist, -enum_counter)
        enum_counter += 1
        if best_key is None or key > best_key:
            best_key = key
            best_assignment = assignment

    _consider({})  # no match at all is always a valid (worst) candidate

    for j, _v in candidates["left"]:
        _consider({"left": j})
    for k, _v in candidates["right"]:
        _consider({"right": k})
    for j, _vj in candidates["left"]:
        for k, _vk in candidates["right"]:
            if j != k:
                _consider({"left": j, "right": k})

    def _slot_result(slot: str) -> AgnosticSlotMatch:
        gt_kps = gt_by_slot[slot]
        if slot in best_assignment:
            j = best_assignment[slot]
            matched_iou = next(v for (jj, v) in candidates[slot] if jj == j)
            return AgnosticSlotMatch(
                matched=True, matched_pred=pred_pool[j],
                matched_iou=matched_iou, status="TP",
            )
        if gt_kps is None:
            return AgnosticSlotMatch(
                matched=False, matched_pred=None,
                matched_iou=float("nan"), status="NONE",
            )
        return AgnosticSlotMatch(
            matched=False, matched_pred=None,
            matched_iou=float("nan"), status="FN",
        )

    left_result = _slot_result("left")
    right_result = _slot_result("right")
    n_matched = len(best_assignment)
    n_phantom = n_pred - n_matched

    return AgnosticFrameMatchResult(
        left=left_result, right=right_result,
        n_matched=n_matched, n_phantom=n_phantom,
    )
