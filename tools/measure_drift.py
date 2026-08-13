"""Measure how far a region drifts across a video. CPU-only, no checkpoints.

Written for the failure this project targets: Wan2.2 generates "robot arm
closes the drawer" but the whole cabinet slides across the frame instead of
staying bolted to the scene. Before changing the sampler it is worth knowing,
numerically, what is actually wrong -- and afterwards, whether it improved.

This does three jobs:

  1. **Confirms the diagnosis.** Template matching recovers the best rigid
     translation of a region *and* how well it still matches there. A high
     correlation at a large offset means the region translated (fixable by
     pinning it to an anchor); a low correlation at every offset means it
     deformed instead, which pinning will not fix.
  2. **Picks the frame range.** `--threshold` reports the first frame whose
     displacement exceeds a tolerance, which is the `A` to pass to
     `static_range_edit.py`.
  3. **Scores the fix.** Re-run it on the edited video and compare. This is
     the local stand-in for the ObjMC metric SG-I2V reports.

Pass several `--box` arguments to tell camera motion apart from object
motion: if every region drifts by the same vector, the camera moved, and
holding one region still is the wrong remedy.

Usage:
    python tools/measure_drift.py --video bad.mp4 --box 620,180,900,540
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# From draw_box, not frame_range_edit: the latter imports `wan` at module
# scope, which drags in the whole Wan2.2 dependency stack, and this tool is
# meant to run on a laptop with neither checkpoints nor a GPU (see README).
from draw_box import parse_box, read_video_frames  # noqa: E402


def track_box(
    frames: np.ndarray,
    box: tuple[int, int, int, int],
    ref_frame: int = 0,
    search_radius: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Best rigid (dx, dy) of `box` in every frame, plus the match score there.

    Returns `(displacement, score)` with shapes `(T, 2)` and `(T,)`. Uses
    normalised cross-correlation, so `score` is in [-1, 1] and comparable
    across frames regardless of brightness changes -- which is what lets the
    caller distinguish "moved" (high score, large offset) from "deformed"
    (low score everywhere).
    """
    x1, y1, x2, y2 = box
    h, w = frames.shape[1:3]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"box {box} is outside the {w}x{h} frame")

    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    template = gray[ref_frame][y1:y2, x1:x2]

    sx1, sy1 = max(0, x1 - search_radius), max(0, y1 - search_radius)
    sx2, sy2 = min(w, x2 + search_radius), min(h, y2 + search_radius)
    if sx2 - sx1 < template.shape[1] or sy2 - sy1 < template.shape[0]:
        raise ValueError("search region is smaller than the template -- reduce --search_radius or the box")

    disp = np.zeros((len(frames), 2), dtype=np.float32)
    score = np.zeros(len(frames), dtype=np.float32)
    for i, g in enumerate(gray):
        result = cv2.matchTemplate(g[sy1:sy2, sx1:sx2], template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        # max_loc is the template's top-left within the search region
        disp[i] = (sx1 + max_loc[0] - x1, sy1 + max_loc[1] - y1)
        score[i] = max_val
    return disp, score


def summarize(disp: np.ndarray, score: np.ndarray) -> dict:
    magnitude = np.linalg.norm(disp, axis=1)
    return {
        "max_px": float(magnitude.max()),
        "mean_px": float(magnitude.mean()),
        "final_px": float(magnitude[-1]),
        "min_score": float(score.min()),
        "mean_score": float(score.mean()),
        "argmax": int(magnitude.argmax()),
    }


def first_frame_over(disp: np.ndarray, threshold: float) -> int | None:
    magnitude = np.linalg.norm(disp, axis=1)
    over = np.flatnonzero(magnitude > threshold)
    return int(over[0]) if len(over) else None


def looks_like_camera_motion(displacements: list[np.ndarray], tol: float = 4.0) -> bool:
    """True when every tracked region moves by nearly the same vector.

    Holding one region still cannot fix a camera move -- the boxes would have
    to follow the scene instead -- so this is worth knowing before running an
    edit that cannot work.
    """
    if len(displacements) < 2:
        return False
    stacked = np.stack(displacements)  # (n_boxes, T, 2)
    spread = np.linalg.norm(stacked - stacked.mean(axis=0, keepdims=True), axis=2)
    return bool(spread.max() < tol and np.linalg.norm(stacked.mean(axis=0), axis=1).max() > tol)


def write_overlay(
    path: Path,
    frames: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    displacements: list[np.ndarray],
    fps: float,
) -> None:
    """Original box in red, tracked position in green -- the gap is the drift."""
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, frame in enumerate(frames):
        img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        for box, disp in zip(boxes, displacements):
            x1, y1, x2, y2 = box
            dx, dy = int(round(disp[i][0])), int(round(disp[i][1]))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1 + dx, y1 + dy), (x2 + dx, y2 + dy), (0, 255, 0), 2)
            cv2.putText(img, f"({dx:+d},{dy:+d})", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(img)
    writer.release()
    print(f"wrote {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--box", type=str, action="append", required=True,
                    help="'x1,y1,x2,y2' region to track, in video pixels. Repeatable -- pass a second "
                         "box on distant background to tell camera motion from object motion")
    ap.add_argument("--ref_frame", type=int, default=0,
                    help="frame the region is measured against; use the last known-good frame")
    ap.add_argument("--search_radius", type=int, default=96,
                    help="how far around the original position to search, in pixels")
    ap.add_argument("--threshold", type=float, default=4.0,
                    help="displacement in pixels that counts as drift, for the suggested --start_frame")
    ap.add_argument("--per_frame", action="store_true", help="print every frame, not just the summary")
    ap.add_argument("--overlay", type=Path, default=None, help="write a video with the tracked boxes drawn")
    ap.add_argument("--csv", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    boxes = [parse_box(b) for b in args.box]
    frames, fps = read_video_frames(args.video)
    print(f"{args.video}: {len(frames)} frames, {frames.shape[2]}x{frames.shape[1]}, {fps:.2f} fps")
    if not (0 <= args.ref_frame < len(frames)):
        raise ValueError(f"--ref_frame must be in [0, {len(frames) - 1}]")

    displacements, scores = [], []
    for box in boxes:
        disp, score = track_box(frames, box, ref_frame=args.ref_frame, search_radius=args.search_radius)
        displacements.append(disp)
        scores.append(score)

        stats = summarize(disp, score)
        print(f"\nbox {box}")
        print(f"  drift:  max {stats['max_px']:.1f}px (frame {stats['argmax']}), "
              f"mean {stats['mean_px']:.1f}px, final {stats['final_px']:.1f}px")
        print(f"  match:  min {stats['min_score']:.3f}, mean {stats['mean_score']:.3f}")
        if stats["min_score"] < 0.5:
            print("  NOTE: the region stops matching well even at its best offset -- it is deforming, "
                  "not just translating. Pinning it to an anchor will not fix that.")
        first = first_frame_over(disp, args.threshold)
        if first is None:
            print(f"  never exceeds {args.threshold}px -- this region looks stable")
        else:
            print(f"  first frame over {args.threshold}px: {first}  "
                  f"(suggests --start_frame {max(0, first - 1)})")

        if args.per_frame:
            for i, (d, s) in enumerate(zip(disp, score)):
                print(f"    {i:4d}  dx={d[0]:+7.1f}  dy={d[1]:+7.1f}  |d|={np.linalg.norm(d):6.1f}  score={s:.3f}")

    if looks_like_camera_motion(displacements):
        print("\nWARNING: every tracked region drifts by nearly the same vector, which means the "
              "camera moved rather than the objects. Holding one region still is the wrong fix -- "
              "the boxes would need to track the scene instead.")

    if args.csv:
        rows = ["frame," + ",".join(f"box{i}_dx,box{i}_dy,box{i}_score" for i in range(len(boxes)))]
        for f in range(len(frames)):
            cells = []
            for d, s in zip(displacements, scores):
                cells += [f"{d[f][0]:.2f}", f"{d[f][1]:.2f}", f"{s[f]:.4f}"]
            rows.append(f"{f}," + ",".join(cells))
        args.csv.write_text("\n".join(rows) + "\n")
        print(f"\nwrote {args.csv}")

    if args.overlay:
        write_overlay(args.overlay, frames, boxes, displacements, fps)


if __name__ == "__main__":
    main()
