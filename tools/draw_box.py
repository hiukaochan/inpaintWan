"""Draw boxes on one frame of a video, to check where they land.

`static_range_edit.py --static_box x1,y1,x2,y2` takes coordinates in
source-video pixels, and getting them wrong is expensive: a box overlapping
the drawer front fights the very motion the video is supposed to show, and you
find out only after a GPU run. This renders the box onto a frame so you can
look first.

**The thing worth looking at is the second rectangle.** A box does not take
effect at the coordinates you type. `static_range_edit.py` maps it through
`frame_mapping.pixel_box_to_latent_box`, which rounds *outward* to whole
16-pixel latent cells -- so the region actually pinned is always at least as
large as what you asked for, and up to `stride - 1` pixels larger on each
side. A box that looks clear of the drawer front can overlap it once snapped.
So the preview draws both: the box as typed (red) and the region that will
really be affected (green).

Deliberately depends on nothing but OpenCV, numpy, and `frame_mapping` (which
is itself torch-free) -- no checkpoints, no GPU, no GUI window. It therefore
runs on a laptop and over SSH alike. In particular it does *not* import
`frame_range_edit.py`, which pulls in the whole Wan2.2 dependency stack at
module scope.

Usage:
    python tools/draw_box.py --video initial.mp4 --frame 52 \
        --box 620,180,900,540 --output boxes.png --grid
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frame_mapping import (  # noqa: E402
    linear_trajectory,
    load_trajectory_npy,
    pixel_box_to_latent_box,
)

Box = tuple[int, int, int, int]

TYPED_COLOR = (255, 64, 64)      # RGB red    -- the box as given
EFFECTIVE_COLOR = (64, 255, 96)  # RGB green  -- what actually gets pinned
GRID_COLOR = (255, 255, 0)
TRAJ_COLOR = (80, 170, 255)      # RGB blue   -- an object's path
TRAJ_END_COLOR = (255, 200, 60)  # RGB amber  -- where the path ends


def parse_box(spec: str) -> Box:
    """Parse an 'x1,y1,x2,y2' box spec. Same format as `--static_box` and
    `--mask_box` elsewhere in the project."""
    parts = spec.split(",")
    if len(parts) != 4:
        raise ValueError(f"box must be 'x1,y1,x2,y2', got {spec!r}")
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"box coordinates must be integers, got {spec!r}") from None
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"box must have x1<x2 and y1<y2, got {spec!r}")
    return x1, y1, x2, y2


def read_video_frames(path: Path) -> tuple[np.ndarray, float]:
    """Read every frame as a (T, H, W, 3) uint8 RGB array, plus fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no frames read from {path}")
    return np.stack(frames), fps


def read_frame(path: Path, index: int) -> tuple[np.ndarray, int, float]:
    """Read a single frame as (H, W, 3) uint8 RGB, plus the frame count and fps.

    `index` may be negative, counting back from the end (-1 is the last frame).

    Seeking with `CAP_PROP_POS_FRAMES` is not reliable on every mp4 codec --
    it can land on a nearby keyframe instead -- and a preview that silently
    showed a neighbouring frame would be worse than useless. So the seek is
    verified against `CAP_PROP_POS_FRAMES` after the read, and falls back to a
    sequential scan when it did not land where asked.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    resolved = index + total if index < 0 else index
    if total > 0 and not (0 <= resolved < total):
        cap.release()
        raise ValueError(f"--frame {index} is out of range for {path} ({total} frames)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, resolved)
    landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ok, frame = cap.read()
    if not ok or landed != resolved:
        # Seek was unreliable (or the container has no usable index): rewind
        # and walk forward, which always works.
        cap.release()
        frames, fps = read_video_frames(path)
        total = len(frames)
        resolved = index + total if index < 0 else index
        if not (0 <= resolved < total):
            raise ValueError(f"--frame {index} is out of range for {path} ({total} frames)")
        return frames[resolved], total, fps

    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), total, fps


def snap_box_to_latent(box: Box, frame_h: int, frame_w: int, stride: int = 16) -> Box:
    """The pixel region a box really covers once mapped to the latent grid.

    Delegates to `frame_mapping.pixel_box_to_latent_box` -- the same function
    `static_range_edit.py` uses -- and converts back to pixels, so the preview
    cannot drift out of agreement with what the sampler will actually do.
    """
    latent_h, latent_w = frame_h // stride, frame_w // stride
    lx1, ly1, lx2, ly2 = pixel_box_to_latent_box(box, latent_h, latent_w, vae_spatial_stride=stride)
    return lx1 * stride, ly1 * stride, lx2 * stride, ly2 * stride


def _draw_rect(img: np.ndarray, box: Box, color, thickness: int) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _draw_grid(img: np.ndarray, stride: int, color) -> None:
    """Faint latent-cell lattice, so the outward snap is legible."""
    h, w = img.shape[:2]
    overlay = img.copy()
    for x in range(0, w, stride):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, stride):
        cv2.line(overlay, (0, y), (w, y), color, 1)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)


def draw_boxes(
    frame: np.ndarray,
    boxes: list[Box],
    *,
    show_effective: bool = True,
    grid: bool = False,
    labels: bool = True,
    stride: int = 16,
    thickness: int = 2,
) -> np.ndarray:
    """Return `frame` (H, W, 3 RGB) with `boxes` drawn on a copy.

    Red is the box as given; green is the region that will actually be pinned
    after the outward snap to latent cells (see the module docstring). When the
    two coincide only one rectangle is visible, which is itself the signal that
    the box is already latent-aligned.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H, W, 3) RGB, got {frame.shape}")
    img = np.ascontiguousarray(frame.copy())
    h, w = img.shape[:2]

    if grid:
        _draw_grid(img, stride, GRID_COLOR)

    for box in boxes:
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            print(f"  WARNING: box {box} extends outside the {w}x{h} frame -- it will be clipped")

        if show_effective:
            _draw_rect(img, snap_box_to_latent(box, h, w, stride), EFFECTIVE_COLOR, thickness)
        _draw_rect(img, box, TYPED_COLOR, thickness)

        if labels:
            text = f"{x1},{y1},{x2},{y2}"
            ty = y1 - 6 if y1 > 20 else y2 + 18
            # dark backing so the label stays readable over any footage
            cv2.putText(img, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TYPED_COLOR, 1)
    return img


def draw_trajectory(
    frame: np.ndarray,
    path: np.ndarray,
    *,
    intermediate: int = 3,
    thickness: int = 2,
) -> np.ndarray:
    """Draw an object's box path on a copy of `frame`.

    `path` is `(F, 4)` boxes as produced by `frame_mapping.linear_trajectory`
    or `load_trajectory_npy`. The centre track is drawn as a polyline, with the
    box outlined at the start (blue), the end (amber), and a few intermediate
    positions so the sweep is legible.

    This is the check worth doing before a GPU run: a path that clips off-frame
    or crosses something that should not move is obvious here and expensive
    to discover afterwards.
    """
    if path.ndim != 2 or path.shape[1] != 4:
        raise ValueError(f"path must be (F, 4), got {path.shape}")
    img = np.ascontiguousarray(frame.copy())

    centres = np.stack([(path[:, 0] + path[:, 2]) / 2, (path[:, 1] + path[:, 3]) / 2], axis=1)
    pts = centres.round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=TRAJ_COLOR, thickness=thickness)

    picks = np.unique(np.linspace(0, len(path) - 1, intermediate + 2).round().astype(int))
    for k, i in enumerate(picks):
        x1, y1, x2, y2 = (int(round(v)) for v in path[i])
        last = k == len(picks) - 1
        color = TRAJ_END_COLOR if last else TRAJ_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness if (k == 0 or last) else 1)

    cx, cy = (int(round(v)) for v in centres[-1])
    cv2.circle(img, (cx, cy), max(4, thickness * 3), TRAJ_END_COLOR, -1)
    return img


def save_image(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
        raise ValueError(f"failed to write {path}")
    print(f"wrote {path}")


def describe(box: Box, frame_h: int, frame_w: int, stride: int = 16) -> str:
    latent_h, latent_w = frame_h // stride, frame_w // stride
    lx1, ly1, lx2, ly2 = pixel_box_to_latent_box(box, latent_h, latent_w, vae_spatial_stride=stride)
    eff = snap_box_to_latent(box, frame_h, frame_w, stride)
    grew = (eff[2] - eff[0]) * (eff[3] - eff[1]) - (box[2] - box[0]) * (box[3] - box[1])
    return (f"  {box}  ->  latent cells x[{lx1}:{lx2}] y[{ly1}:{ly2}]  "
            f"->  effective pixels {eff}  (+{grew} px^2)")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=0,
                    help="frame index to draw on; negative counts from the end (-1 is the last frame)")
    ap.add_argument("--box", type=str, action="append", default=None,
                    help="'x1,y1,x2,y2' in video pixels: a static box. Repeatable")
    ap.add_argument("--object_traj", type=str, action="append", default=None,
                    help="trajectory .npy in SG-I2V's [N, 2+F, 2] format, drawn as a path. Repeatable. "
                         "Same flag name as static_range_edit.py takes")
    ap.add_argument("--object_box", type=str, default=None,
                    help="'x1,y1,x2,y2' start box for a straight-line path preview; needs --move_to")
    ap.add_argument("--move_to", type=str, default=None,
                    help="'dx,dy' pixel displacement for --object_box")
    ap.add_argument("--output", type=Path, default=Path("boxes.png"))
    ap.add_argument("--grid", action="store_true",
                    help="overlay the latent-cell lattice, so the outward snap is visible")
    ap.add_argument("--stride", type=int, default=16,
                    help="VAE spatial stride (16 for ti2v-5B); sets the latent cell size")
    ap.add_argument("--no_effective", action="store_true",
                    help="draw only the box as typed, not the snapped region it really covers")
    ap.add_argument("--no_labels", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if bool(args.object_box) != bool(args.move_to):
        raise ValueError("--object_box and --move_to must be given together")
    if not args.box and not args.object_traj and not args.object_box:
        raise ValueError("pass at least one of --box, --object_traj or --object_box")
    boxes = [parse_box(b) for b in (args.box or [])]

    frame, total, fps = read_frame(args.video, args.frame)
    h, w = frame.shape[:2]
    print(f"{args.video}: frame {args.frame} of {total}, {w}x{h}, {fps:.2f} fps")

    paths = []
    for traj in (args.object_traj or []):
        paths.extend(load_trajectory_npy(Path(traj), (w, h), (w, h)))
    if args.object_box:
        parts = args.move_to.split(",")
        if len(parts) != 2:
            raise ValueError(f"--move_to must be 'dx,dy', got {args.move_to!r}")
        paths.append(linear_trajectory(parse_box(args.object_box),
                                       (float(parts[0]), float(parts[1])), 25))

    for box in boxes:
        print(describe(box, h, w, args.stride))
    for i, path in enumerate(paths):
        start = tuple(int(round(v)) for v in path[0])
        end = tuple(int(round(v)) for v in path[-1])
        dx = (end[0] + end[2] - start[0] - start[2]) / 2
        dy = (end[1] + end[3] - start[1] - start[3]) / 2
        print(f"  object {i}: {start} -> {end}   moves ({dx:+.0f}, {dy:+.0f}) px "
              f"= ({dx / args.stride:+.1f}, {dy / args.stride:+.1f}) latent cells")
        if max(abs(dx), abs(dy)) < args.stride:
            print(f"    WARNING: the whole path is under one {args.stride}px latent cell -- "
                  f"this motion cannot be expressed by a latent-space edit")

    annotated = draw_boxes(
        frame, boxes,
        show_effective=not args.no_effective, grid=args.grid,
        labels=not args.no_labels, stride=args.stride)
    for path in paths:
        annotated = draw_trajectory(annotated, path)
    save_image(args.output, annotated)


if __name__ == "__main__":
    main()
