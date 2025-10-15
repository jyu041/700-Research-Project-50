#!/usr/bin/env python3
"""
keep_two_per_person.py

Scan a directory of videos named as CLASS_PERSON_INDEX.mp4 (e.g., 001_002_003.mp4),
and for each (class, person) pair, keep only the first two videos (lowest INDEX).
Copies (default) or moves them into an output directory, organized by class.

Usage:
  python keep_two_per_person.py --src /path/to/input --dst /path/to/output
  # move instead of copy:
  python keep_two_per_person.py --src /path/to/input --dst /path/to/output --move
  # dry run (show what would happen):
  python keep_two_per_person.py --src /path/to/input --dst /path/to/output --dry-run
  # allow overwriting existing files at the destination:
  python keep_two_per_person.py --src /path/to/input --dst /path/to/output --overwrite
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
import shutil
import sys

FNAME_RE = re.compile(r'^(\d{3})_(\d{3})_(\d{3})\.mp4$', re.IGNORECASE)

def parse_filename(p: Path):
    """
    Return (class_id, person_id, index) as strings '001', '002', '003' if matches, else None.
    """
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    cls, person, idx = m.groups()
    return cls, person, idx

def plan_selection(src_dir: Path):
    """
    Build a mapping (class_id, person_id) -> [Path, ...] sorted by index, then filename.
    Return a dict with only the first two per group selected.
    """
    groups = defaultdict(list)

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        parsed = parse_filename(p)
        if not parsed:
            continue
        cls, person, idx = parsed
        # use numeric sort on index, but keep original zero-padded strings
        groups[(cls, person)].append((int(idx), p.name, p))

    selected = {}
    for key, items in groups.items():
        items.sort(key=lambda x: (x[0], x[1]))  # sort by numeric index, then name for stability
        chosen = [t[2] for t in items[:2]]      # pick first two Paths
        selected[key] = chosen

    return selected

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def do_transfer(src: Path, dst: Path, move: bool, overwrite: bool, dry_run: bool):
    if dst.exists():
        if dst.is_dir():
            raise IsADirectoryError(f"Destination is a directory: {dst}")
        if not overwrite:
            print(f"[skip] Exists: {dst}")
            return
    else:
        ensure_dir(dst.parent)

    action = "MOVE" if move else "COPY"
    print(f"[{action}] {src} -> {dst}")
    if not dry_run:
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

def main():
    ap = argparse.ArgumentParser(description="Keep only first two videos per (class, person) pair.")
    ap.add_argument("--src", required=True, type=Path, help="Source directory containing videos")
    ap.add_argument("--dst", required=True, type=Path, help="Destination directory for selected videos")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without changing files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    ap.add_argument("--flat", action="store_true",
                    help="Do not organize by class subfolders; place files directly in --dst")
    args = ap.parse_args()

    if not args.src.exists() or not args.src.is_dir():
        print(f"Source directory not found or not a directory: {args.src}", file=sys.stderr)
        sys.exit(1)

    selection = plan_selection(args.src)

    if not selection:
        print("No matching files found (expected pattern: 000_000_000.mp4).")
        sys.exit(0)

    total_selected = 0
    for (cls, person), files in sorted(selection.items()):
        if not files:
            continue
        # Output path: either flat or grouped by class
        out_dir = args.dst if args.flat else (args.dst / cls)

        if len(files) < 2:
            print(f"[warn] Found only {len(files)} file(s) for class {cls}, person {person}.")

        for src_path in files:
            dst_path = out_dir / src_path.name
            try:
                do_transfer(src_path, dst_path, move=args.move, overwrite=args.overwrite, dry_run=args.dry_run)
                total_selected += 1
            except Exception as e:
                print(f"[error] {src_path} -> {dst_path}: {e}", file=sys.stderr)

    print(f"\nDone. Selected {total_selected} file(s).")
    if args.dry_run:
        print("This was a dry run. Re-run without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
