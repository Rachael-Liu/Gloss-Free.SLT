"""
Prepare Kaggle Phoenix-2014T (compressed) into this repo's dataset layout.

Input (already downloaded/unzipped from Kaggle):
  data/kaggle_raw/
    phoenix14t.pami0.{train,dev,test}.annotations_only.gzip   # gzip(pickle(list[dict]))
    videos_phoenix/videos/{train,dev,test}/*.mp4

Output layout (matches src/data/dataset.py expectations):
  <out_root>/
    train|val|test/
      <sample_id>/
        rgb/000001.jpg ...
        flow/u/000001.npy ...
        flow/v/000001.npy ...
        tgt.txt

Notes:
  - This uses OpenCV Farneback optical flow for a cheap baseline.
  - This is CPU-heavy and can take a long time for full dataset.
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def _load_annotations(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Unexpected annotation type: {type(data)}")
    return data


def _iter_split_samples(
    kaggle_root: Path, split: str
) -> Iterable[Tuple[str, Path, str]]:
    """
    Yields (sample_id, mp4_path, text).
    sample_id corresponds to annotation name without leading '<split>/'.
    """
    ann_path = kaggle_root / f"phoenix14t.pami0.{split}.annotations_only.gzip"
    anns = _load_annotations(ann_path)
    vid_dir = kaggle_root / "videos_phoenix" / "videos" / split

    for item in anns:
        name = item["name"]  # e.g. "train/11August_2010_...-1"
        if not name.startswith(split + "/"):
            # Some dumps may include the split prefix; be defensive.
            sample_id = name.split("/", 1)[-1]
        else:
            sample_id = name.split("/", 1)[1]
        mp4 = vid_dir / f"{sample_id}.mp4"
        if not mp4.is_file():
            raise FileNotFoundError(f"Missing video for sample '{name}': {mp4}")
        text = str(item.get("text", "")).strip()
        yield sample_id, mp4, text


def _write_rgb_and_flow(
    mp4_path: Path,
    rgb_dir: Path,
    u_dir: Path,
    v_dir: Path,
    max_frames: int | None,
    resize_hw: Tuple[int, int] = (224, 224),
) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    prev_gray: np.ndarray | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        frame = cv2.resize(frame, resize_hw[::-1], interpolation=cv2.INTER_AREA)  # (W,H)
        rgb_path = rgb_dir / f"{frame_idx:06d}.jpg"
        cv2.imwrite(str(rgb_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            u = np.zeros_like(gray, dtype=np.float32)
            v = np.zeros_like(gray, dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            u = flow[..., 0].astype(np.float32)
            v = flow[..., 1].astype(np.float32)

        np.save(str(u_dir / f"{frame_idx:06d}.npy"), u)
        np.save(str(v_dir / f"{frame_idx:06d}.npy"), v)
        prev_gray = gray

    cap.release()
    if frame_idx == 0:
        raise RuntimeError(f"No frames decoded from: {mp4_path}")
    return frame_idx


def prepare(
    kaggle_root: Path,
    out_root: Path,
    limit_per_split: int | None,
    limit_train: int | None,
    limit_val: int | None,
    limit_test: int | None,
    max_frames: int | None,
    overwrite: bool,
) -> None:
    kaggle_root = kaggle_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "dev": "val", "test": "test"}

    for src_split, dst_split in split_map.items():
        split_limit: int | None
        if src_split == "train" and limit_train is not None:
            split_limit = limit_train
        elif src_split == "dev" and limit_val is not None:
            split_limit = limit_val
        elif src_split == "test" and limit_test is not None:
            split_limit = limit_test
        else:
            split_limit = limit_per_split

        dst_split_dir = out_root / dst_split
        dst_split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for sample_id, mp4_path, text in _iter_split_samples(kaggle_root, src_split):
            count += 1
            if split_limit is not None and count > split_limit:
                break

            sample_dir = dst_split_dir / sample_id
            rgb_dir = sample_dir / "rgb"
            u_dir = sample_dir / "flow" / "u"
            v_dir = sample_dir / "flow" / "v"

            if sample_dir.exists() and overwrite:
                # Remove only the subfolders we create to avoid surprises.
                for p in [rgb_dir, u_dir, v_dir]:
                    if p.exists():
                        for child in p.glob("*"):
                            child.unlink()

            rgb_dir.mkdir(parents=True, exist_ok=True)
            u_dir.mkdir(parents=True, exist_ok=True)
            v_dir.mkdir(parents=True, exist_ok=True)

            with open(sample_dir / "tgt.txt", "w", encoding="utf-8") as f:
                f.write(text + "\n")

            _write_rgb_and_flow(
                mp4_path=mp4_path,
                rgb_dir=rgb_dir,
                u_dir=u_dir,
                v_dir=v_dir,
                max_frames=max_frames,
            )

            if count % 50 == 0:
                print(f"[{src_split}->{dst_split}] processed {count} samples ...")

        print(f"[{src_split}->{dst_split}] done: {count} samples")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Kaggle Phoenix-2014T to OFTS dataset layout.")
    ap.add_argument(
        "--kaggle-root",
        type=Path,
        default=Path("data") / "kaggle_raw",
        help="Path containing phoenix14t annotations + videos_phoenix folder.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("data") / "phoenix_ofts",
        help="Output dataset root for training (--dataset-root).",
    )
    ap.add_argument("--limit-per-split", type=int, default=50, help="Process only N samples per split (debug).")
    ap.add_argument("--limit-train", type=int, default=None, help="Only process N samples for train (overrides --limit-per-split).")
    ap.add_argument("--limit-val", type=int, default=None, help="Only process N samples for dev/val (overrides --limit-per-split).")
    ap.add_argument("--limit-test", type=int, default=None, help="Only process N samples for test (overrides --limit-per-split).")
    ap.add_argument("--max-frames", type=int, default=120, help="Decode at most N frames per video.")
    ap.add_argument("--full", action="store_true", help="Process the full dataset (sets limit-per-split=None).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite already-prepared samples.")
    args = ap.parse_args()

    limit = None if args.full else args.limit_per_split
    prepare(
        kaggle_root=args.kaggle_root,
        out_root=args.out_root,
        limit_per_split=limit,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        max_frames=args.max_frames,
        overwrite=bool(args.overwrite),
    )
    print(f"[ok] dataset root ready: {args.out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

