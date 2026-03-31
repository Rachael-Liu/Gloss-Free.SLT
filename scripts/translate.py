"""
One-command wrapper around scripts/infer.py.

Goal: reduce arguments you need to type. It picks sensible defaults:
- ckpt: checkpoints/oftsnet_best.pt
- vocab: checkpoints/phoenix_vocab.json if exists else checkpoints/vocab.json, else use ckpt['vocab']
- max_frames: from ckpt config if present, else 120
- device: cuda if available else cpu

Examples:
  python scripts/translate.py --video path/to/video.mp4
  python scripts/translate.py --sample-dir data/phoenix_ofts_subset/test/<sample_id>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure repo root import works even if run from elsewhere
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.infer import main as infer_main


def _default_ckpt() -> str:
    return str(_ROOT / "checkpoints" / "oftsnet_best.pt")


def _pick_vocab_path() -> str | None:
    cand = [
        _ROOT / "checkpoints" / "phoenix_vocab.json",
        _ROOT / "checkpoints" / "vocab.json",
    ]
    for p in cand:
        if p.is_file():
            return str(p)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Translate wrapper (calls scripts/infer.py with defaults).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str, help="input .mp4")
    g.add_argument("--sample-dir", type=str, help="sample folder with rgb/flow")

    ap.add_argument("--ckpt", type=str, default=_default_ckpt(), help="checkpoint path (default: checkpoints/oftsnet_best.pt)")
    ap.add_argument("--vocab-path", type=str, default=None, help="vocab json path (optional)")
    ap.add_argument("--max-frames", type=int, default=None, help="override max_frames (optional)")
    ap.add_argument("--device", type=str, default=None, help="cpu/cuda (optional, auto by default)")
    ap.add_argument("--max-decode", type=int, default=None, help="override max decode steps (optional)")
    args = ap.parse_args()

    ckpt = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    # auto device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # auto vocab path (prefer phoenix_vocab.json if present)
    vocab_path = args.vocab_path
    if vocab_path is None:
        vocab_path = _pick_vocab_path()

    # auto max_frames from ckpt config if not provided
    max_frames = args.max_frames
    if max_frames is None:
        try:
            meta = torch.load(ckpt, map_location="cpu")
            cfg = meta.get("config") or {}
            if "max_frames" in cfg:
                max_frames = int(cfg["max_frames"])
        except Exception:
            max_frames = None
    if max_frames is None:
        max_frames = 120

    # Build argv for infer_main()
    argv = ["infer.py", "--ckpt", ckpt, "--device", device, "--max-frames", str(max_frames)]
    if args.max_decode is not None:
        argv += ["--max-decode", str(int(args.max_decode))]
    if vocab_path is not None:
        argv += ["--vocab-path", os.path.abspath(vocab_path)]

    if args.video:
        argv += ["--video", os.path.abspath(args.video)]
    else:
        argv += ["--sample-dir", os.path.abspath(args.sample_dir)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        return int(infer_main() or 0)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

