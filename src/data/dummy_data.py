"""Tiny synthetic OFTS layout for smoke tests (not for real experiments)."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from PIL import Image


def _write_video(
    split_dir: str,
    vid: str,
    num_frames: int,
    tgt_line: str,
    rng: np.random.Generator,
) -> None:
    vid_dir = os.path.join(split_dir, vid)
    rgb_dir = os.path.join(vid_dir, "rgb")
    u_dir = os.path.join(vid_dir, "flow", "u")
    v_dir = os.path.join(vid_dir, "flow", "v")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    h, w = 64, 64
    for i in range(1, num_frames + 1):
        name = f"{i:06d}"
        arr = (rng.random((h, w, 3)) * 255).astype("uint8")
        Image.fromarray(arr, mode="RGB").save(os.path.join(rgb_dir, f"{name}.jpg"), quality=85)
        u = rng.standard_normal((h, w)).astype(np.float32) * 0.1
        v = rng.standard_normal((h, w)).astype(np.float32) * 0.1
        np.save(os.path.join(u_dir, f"{name}.npy"), u)
        np.save(os.path.join(v_dir, f"{name}.npy"), v)

    with open(os.path.join(vid_dir, "tgt.txt"), "w", encoding="utf-8") as f:
        f.write(tgt_line.strip() + "\n")


def ensure_dummy_ofts_dataset(root: str, seed: int = 42) -> str:
    """
    Create (or refresh) a minimal dataset at ``root`` matching ``SignTranslationDataset`` layout.

    Returns absolute path to ``root``.
    """
    root = os.path.abspath(root)
    rng = np.random.default_rng(seed)

    specs: List[Tuple[str, List[Tuple[str, int, str]]]] = [
        (
            "train",
            [
                ("vid_0001", 8, "hello world"),
                ("vid_0002", 6, "sign language test"),
                ("vid_0003", 10, "short"),
                ("vid_0004", 7, "another example sentence"),
            ],
        ),
        (
            "val",
            [
                ("vid_0001", 8, "hello there"),
                ("vid_0002", 6, "validation sample"),
            ],
        ),
    ]

    for split, videos in specs:
        split_dir = os.path.join(root, split)
        os.makedirs(split_dir, exist_ok=True)
        for vid, n_frames, text in videos:
            _write_video(split_dir, vid, n_frames, text, rng)

    marker = os.path.join(root, ".dummy_ofts_manifest")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("v1\n")

    return root
