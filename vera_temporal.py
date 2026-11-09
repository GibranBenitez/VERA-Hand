#!/usr/bin/env python3
"""vera_temporal.py — Release temporal evaluation script for VERA-Hand (WACV 2027).

Computes prediction-intrinsic temporal jitter for ONE hand side over a
directory of per-video prediction JSONs and writes a single summary CSV.

Aggregation across videos (mirrors temporal_summary.py):
    Jacc, Jjerk, Ev  — MEDIAN across videos, then scaled ×100.
    Gaps             — SUM of gap events across all videos.

Output
------
One CSV per run, columns: estimator, Gaps, Jacc, Jjerk, Ev
Filename: {folder_name}_temporal-{hand}_{region}.csv
The estimator name and variant are inferred from the directory name.

Modes
-----
Single estimator
    --preds-path points to a folder that contains .json files directly.
    Example:  predictions/full_videos/Sapiens2-1B_raw/

Batch (multiple estimators)
    --preds-path points to a folder whose subdirectories contain .json files.
    Each subdirectory is treated as one estimator; all estimators are written
    as rows of one CSV in --out-dir.
    Example:  predictions/full_videos/

Usage
-----
    # Single
    python vera_temporal.py \\
        --preds-path predictions/full_videos/Sapiens2-1B_raw/ \\
        --hand right --region all --out-dir outputs

    # Batch
    python vera_temporal.py \\
        --preds-path predictions/full_videos/ \\
        --hand right --region all --out-dir outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from metrics.temporal import load_video_json, process_video  # noqa: E402

logger = logging.getLogger("eval_temporal")

_VALID_HANDS = ("left", "right")
_VALID_REGIONS = ("all", "base", "tips")


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
        help="Directory with per-video JSONs (single mode) or directory "
             "containing estimator subdirectories, each with per-video JSONs "
             "(batch mode). The directory name is used as the estimator label.",
    )
    p.add_argument(
        "--hand", default="right", choices=list(_VALID_HANDS),
        help="Hand side to evaluate: left or right (default: right). "
             "'both' is not supported — temporal metrics are per-hand.",
    )
    p.add_argument(
        "--region", default="all", choices=list(_VALID_REGIONS),
        help="Keypoint subset: all, base, or tips (default: all).",
    )
    p.add_argument(
        "--out-dir", default="outputs",
        help="Output directory (default: outputs).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _json_files_in(directory: Path) -> List[Path]:
    """Return sorted .json files in directory (excluding *.scores.json)."""
    return sorted(
        p for p in directory.glob("*.json")
        if not p.name.endswith(".scores.json")
    )


def _discover_estimator_dirs(root: Path) -> List[Path]:
    """Find subdirectories of root that contain at least one .json file."""
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and _json_files_in(d)
    )


# ---------------------------------------------------------------------------
# Aggregation helpers (mirror temporal_summary.py)
# ---------------------------------------------------------------------------

def _median_finite(values: np.ndarray) -> float:
    """Median ignoring NaN/Inf. Returns NaN if no finite values."""
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    return float(np.median(values[mask]))


def _summarize(wide_rows: List[Dict], hand: str, region: str) -> Dict:
    """One-row summary for a single hand slot over all videos.

    Jacc/Jjerk/Ev: median across videos × 100.
    Gaps: total sum across videos.
    """
    slot_rows = [r for r in wide_rows if str(r.get("slot", "")).lower() == hand]
    if not slot_rows:
        logger.warning("No rows found for slot=%s.", hand)
        return {"Gaps": 0, "Jacc": float("nan"), "Jjerk": float("nan"), "Ev": float("nan")}

    jacc_vals  = np.array([float(r.get(f"Jacc_{region}",  float("nan"))) for r in slot_rows])
    jjerk_vals = np.array([float(r.get(f"Jjerk_{region}", float("nan"))) for r in slot_rows])
    ev_vals    = np.array([float(r.get(f"Ev_{region}",    float("nan"))) for r in slot_rows])
    gaps_total = int(sum(r.get("n_gaps", 0) or 0 for r in slot_rows))

    return {
        "Gaps":  gaps_total,
        "Jacc":  _median_finite(jacc_vals)  * 100,
        "Jjerk": _median_finite(jjerk_vals) * 100,
        "Ev":    _median_finite(ev_vals)    * 100,
    }


# ---------------------------------------------------------------------------
# Core processing for one estimator directory
# ---------------------------------------------------------------------------

def process_estimator_dir(
    estimator_dir: Path,
    hand: str,
    region: str,
    frame_h: int,
    frame_w: int,
    detection_min_kp: int,
    jitter_min_windows: int,
) -> Optional[Dict]:
    """Process all videos in estimator_dir and return a summary dict.

    Returns None if no valid videos were found.
    """
    stem = estimator_dir.name
    json_files = _json_files_in(estimator_dir)
    if not json_files:
        logger.warning("No JSON files in %s — skipping.", estimator_dir)
        return None

    logger.info(
        "%s: %d video JSONs — hand=%s region=%s",
        stem, len(json_files), hand, region,
    )

    all_wide: List[Dict] = []
    for json_path in json_files:
        video_label = json_path.stem
        try:
            frames_json = load_video_json(str(json_path))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("  Skipping %s: %s", json_path.name, exc)
            continue
        if not isinstance(frames_json, dict) or not frames_json:
            logger.warning("  Skipping %s: empty or non-dict JSON.", json_path.name)
            continue

        wide_rows, _ = process_video(
            video_label, frames_json, frame_w, frame_h,
            detection_min_kp, jitter_min_windows,
        )
        all_wide.extend(wide_rows)
        logger.debug("  %s: %d slot rows", video_label, len(wide_rows))

    if not all_wide:
        logger.warning("%s: no valid video rows - skipping.", stem)
        return None

    summary = _summarize(all_wide, hand, region)
    logger.info(
        "  -> Gaps=%d Jacc=%.3f Jjerk=%.3f Ev=%.3f",
        summary["Gaps"], summary["Jacc"], summary["Jjerk"], summary["Ev"],
    )
    return {"estimator": stem, **summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(levelname)s] %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    preds_dir = Path(args.preds_path)
    if not preds_dir.is_dir():
        raise SystemExit(f"--preds-path is not a directory: {preds_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = 480, 640

    # Detect mode: single if preds_dir contains JSONs directly, batch otherwise.
    direct_jsons = _json_files_in(preds_dir)
    if direct_jsons:
        estimator_dirs = [preds_dir]
        logger.info("Single mode: %s (%d JSONs)", preds_dir.name, len(direct_jsons))
    else:
        estimator_dirs = _discover_estimator_dirs(preds_dir)
        if not estimator_dirs:
            raise SystemExit(
                f"No JSON files found directly in {preds_dir} and no "
                "subdirectories with JSONs found either."
            )
        logger.info("Batch mode: %d estimator directories found.", len(estimator_dirs))

    fieldnames = ["estimator", "Gaps", "Jacc", "Jjerk", "Ev"]
    batch_mode = len(estimator_dirs) > 1

    rows: List[Dict] = []
    for est_dir in estimator_dirs:
        result = process_estimator_dir(
            est_dir, args.hand, args.region, h, w,
            detection_min_kp=5, jitter_min_windows=10,
        )
        if result is not None:
            rows.append(result)

    if not rows:
        raise SystemExit("No valid estimator data produced.")

    if batch_mode:
        # One CSV named after the parent folder, one row per estimator.
        stem = preds_dir.name
    else:
        # Single mode: CSV named after the estimator folder itself.
        stem = estimator_dirs[0].name

    out_csv = out_dir / f"{stem}_temporal-{args.hand}_{args.region}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
