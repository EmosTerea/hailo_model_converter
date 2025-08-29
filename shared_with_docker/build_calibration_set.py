#!/usr/bin/env python3
"""
Build a representative calibration image set from a YOLO dataset.

Inputs
 - Archive layout assumed:
   Archive/
     train/{images,labels}
     val/{images,labels}
     test/{images,labels}  (optional)
 - Labels are YOLO TXT files: cls cx cy w h (normalized).

Outputs
 - A folder of symlinks (or copies if symlink fails) to images selected
   for calibration, plus a manifest.json with selection details.

Selection strategy
 - Prefer a split (default: val) but may backfill from train (and test if allowed).
 - Ensure a minimum number of images per class (based on labels present).
 - Encourage diversity of object sizes (small/medium/large by bbox area).
 - Cap the number of images taken from the same sequence/prefix to avoid near-duplicates.
 - Unlabeled images are allowed as fill but are not used for class coverage.

Example
  python build_calibration_set.py \
    --archive-root Archive --outdir calib_yolo --n 400 \
    --prefer val --min-per-class 12 --max-per-seq 6

Then convert with:
  python converter.py --onnx best.onnx --model-name best \
    --calib-folder calib_yolo --img-size 640 640 \
    --preproc letterbox --letterbox-pad 114 \
    --normalize-mean 0 --normalize-std 255 --max-calib 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

random.seed(0)


def read_labels(lbl_path: Path):
    """Read YOLO labels from a .txt file. Returns list of (cls, area_norm)."""
    out = []
    try:
        with open(lbl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = line.replace("\t", " ")
                parts = [p for p in line.split() if p]
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    w = float(parts[3])
                    h = float(parts[4])
                    out.append((cls, max(0.0, w * h)))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return out


def seq_name(p: Path) -> str:
    """Get a coarse sequence/group prefix from filename stem."""
    s = p.stem
    if "." in s:
        s = s.split(".", 1)[0]
    if "_d_" in s:
        s = s.split("_d_", 1)[0]
    return s


def size_bucket(area: float) -> str:
    if area < 0.02:
        return "small"
    if area < 0.10:
        return "medium"
    return "large"


def collect_split(root: Path, split: str) -> List[Dict]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    items: List[Dict] = []
    if not img_dir.exists():
        return items
    for img in img_dir.rglob("*"):
        if not img.is_file():
            continue
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        anns = read_labels(lbl)
        rec = {
            "img": img,
            "labels": anns,
            "split": split,
            "seq": seq_name(img),
        }
        cls_set = {c for c, _ in anns}
        rec["cls_set"] = cls_set
        rec["buckets"] = {size_bucket(a) for _, a in anns} if anns else set()
        items.append(rec)
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", type=Path, default=Path("Archive"))
    ap.add_argument("--outdir", type=Path, default=Path("calib_yolo"))
    ap.add_argument("--n", type=int, default=300, help="Target number of calibration images")
    ap.add_argument("--prefer", choices=["val", "train"], default="val", help="Preferred split to draw from")
    ap.add_argument("--min-per-class", type=int, default=10, help="Minimum images covering each class")
    ap.add_argument("--max-per-seq", type=int, default=6, help="Limit images taken from the same sequence/prefix")
    ap.add_argument("--include-test", action="store_true", help="Allow using test split for backfill if needed")
    args = ap.parse_args()

    root = args.archive_root
    if not (root / "train").exists():
        print(f"[error] {root} missing expected split folders (train/val).", file=sys.stderr)
        return 2

    order = [args.prefer, "train" if args.prefer == "val" else "val"]
    if args.include_test:
        order.append("test")

    # Gather candidates
    items: List[Dict] = []
    for s in order:
        items += collect_split(root, s)

    if not items:
        print("[error] No candidate images found.", file=sys.stderr)
        return 2

    # Global class frequency across candidates
    global_cls = Counter()
    for it in items:
        for c, _ in it.get("labels", []):
            global_cls[c] += 1

    # Compute a score per image that favors: rarer classes, preferred split, size diversity
    def score_item(it: Dict) -> float:
        rarity = sum(1.0 / (global_cls[c] or 1) for c in it["cls_set"]) if it["cls_set"] else 0.0
        split_bonus = 0.2 if it["split"] == args.prefer else 0.0
        bucket_bonus = 0.15 * len(it["buckets"])
        # unlabeled images get only split bonus
        return rarity + split_bonus + bucket_bonus

    items.sort(key=score_item, reverse=True)

    # Coverage first: ensure each class meets min-per-class by counting images (not instances)
    per_class_images = Counter()
    chosen: List[Dict] = []
    chosen_set = set()
    per_seq = Counter()

    classes = sorted(global_cls.keys())
    # Greedy pass: take images that help unmet classes while obeying per-seq cap
    for it in items:
        if len(chosen) >= args.n:
            break
        if per_seq[it["seq"]] >= args.max_per_seq:
            continue
        if not it["cls_set"]:
            continue  # unlabeled handled in fill
        unmet = any(per_class_images[c] < args.min_per_class for c in it["cls_set"])
        if unmet:
            chosen.append(it)
            chosen_set.add(it["img"]) 
            per_seq[it["seq"]] += 1
            for c in it["cls_set"]:
                if per_class_images[c] < args.min_per_class:
                    per_class_images[c] += 1

    # Fill to N with highest-scoring items, allow unlabeled too, still enforce per-seq cap
    for it in items:
        if len(chosen) >= args.n:
            break
        if it["img"] in chosen_set:
            continue
        if per_seq[it["seq"]] >= args.max_per_seq:
            continue
        chosen.append(it)
        chosen_set.add(it["img"]) 
        per_seq[it["seq"]] += 1

    # Materialize output directory
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Create symlinks (or copy fallback) with split-prefixed filenames to avoid collisions
    manifest = []
    for it in chosen:
        src: Path = it["img"]
        dst = outdir / f"{it['split']}__{src.name}"
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)
        manifest.append({
            "src": str(src),
            "dst": str(dst),
            "split": it["split"],
            "classes": sorted(list(it["cls_set"])) if it["cls_set"] else [],
            "seq": it["seq"],
        })

    with open(outdir / "manifest.json", "w") as f:
        json.dump({
            "total": len(chosen),
            "min_per_class": args.min_per_class,
            "max_per_seq": args.max_per_seq,
            "preferred_split": args.prefer,
            "selected": manifest,
        }, f, indent=2)

    print(f"[ok] Selected {len(chosen)} images → {outdir}")
    print(
        f"[hint] Use with: converter.py --calib-folder {outdir} --img-size 640 640 "
        f"--preproc letterbox --letterbox-pad 114 --normalize-mean 0 --normalize-std 255"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

