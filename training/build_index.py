"""Build the DROID training index.

Walks the DROID tree at exactly depth 2 (`<session>/<camera_serial>/`) and emits
`index.json`: one entry per usable (session, camera_serial) pair, with the fps
needed to resample to Wan's 24 fps.

Not every episode is usable. Of 38656 `rgb.mp4` files, ~1271 have no
`caption.pickle` and ~275 no `episode_meta.npz`, so the index is built by
intersecting all three rather than by globbing videos.

Sessions listed in `benchmark_manifest.json["kept"]` are assigned to the `val`
split so the curated benchmark stays genuinely held out.
"""

import argparse
import csv
import json
import os
from pathlib import Path

REQUIRED = ("rgb.mp4", "caption.pickle", "episode_meta.npz")

# DROID's nominal rate; used when download_log.csv has no true_fps for a row
# (the ~2400 'skipped-exists' rows). 38283 of 38298 logged episodes are exactly
# this value, so the fallback is almost never wrong.
DEFAULT_FPS = 14.9254


def load_fps_table(droid_parent: Path) -> dict[tuple[str, str], float]:
    """Map (session, camera_serial) -> true fps from download_log.csv.

    The mp4 containers report a bogus `r_frame_rate` of 179/12, so the log is
    the only trustworthy source.
    """
    path = droid_parent / "download_log.csv"
    if not path.exists():
        return {}
    table = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            fps = row.get("true_fps", "").strip()
            if fps:
                table[(row["episode_id"], row["camera_serial"])] = float(fps)
    return table


def load_val_sessions(droid_parent: Path) -> set[str]:
    path = droid_parent / "benchmark_manifest.json"
    if not path.exists():
        return set()
    with open(path) as fh:
        return set(json.load(fh).get("kept", []))


def scan(root: Path, success_only: bool) -> list[tuple[str, str]]:
    """Yield (session, serial) pairs whose directory has all of REQUIRED.

    `follow_symlinks=False` matters: the DROID root contains a self-symlink
    `droid -> <root>`, which sends any naive recursive walk into a loop.
    """
    pairs = []
    with os.scandir(root) as sessions:
        for session in sessions:
            if not session.is_dir(follow_symlinks=False):
                continue
            if success_only and "__success__" not in session.name:
                continue
            with os.scandir(session.path) as serials:
                for serial in serials:
                    if not serial.is_dir(follow_symlinks=False):
                        continue
                    if all(os.path.exists(os.path.join(serial.path, f)) for f in REQUIRED):
                        pairs.append((session.name, serial.name))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--droid_root",
        type=Path,
        default=Path("/media/NVME_8TB/genvideorobot/OSCAR_robot_droid/droid/droid"),
        help="directory containing the <session>/<serial>/ episode dirs",
    )
    ap.add_argument("--output", type=Path, default=Path("training/index.json"))
    ap.add_argument(
        "--all_outcomes",
        action="store_true",
        help="include __failure__ sessions (default: __success__ only)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="keep only the first N train entries, for a smoke-test cache",
    )
    args = ap.parse_args()

    root = args.droid_root.resolve()
    parent = root.parent

    pairs = scan(root, success_only=not args.all_outcomes)
    fps_table = load_fps_table(parent)
    val_sessions = load_val_sessions(parent)

    entries = []
    for session, serial in sorted(pairs):
        entries.append(
            {
                "session": session,
                "serial": serial,
                "path": str(root / session / serial),
                "fps": fps_table.get((session, serial), DEFAULT_FPS),
                "split": "val" if session in val_sessions else "train",
            }
        )

    if args.limit:
        train = [e for e in entries if e["split"] == "train"][: args.limit]
        val = [e for e in entries if e["split"] == "val"][: max(8, args.limit // 20)]
        entries = train + val

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(
            {
                "droid_root": str(root),
                "success_only": not args.all_outcomes,
                "entries": entries,
            },
            fh,
            indent=1,
        )

    n_train = sum(e["split"] == "train" for e in entries)
    n_val = len(entries) - n_train
    n_sessions = len({e["session"] for e in entries})
    n_logged = sum((e["session"], e["serial"]) in fps_table for e in entries)
    print(f"wrote {args.output}: {len(entries)} episodes ({n_train} train / {n_val} val)")
    print(f"  {n_sessions} sessions, {len(entries) - n_logged} using fallback fps")


if __name__ == "__main__":
    main()
