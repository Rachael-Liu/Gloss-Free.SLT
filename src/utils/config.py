from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class TrainConfig:
    # Data
    dataset_root: str
    dummy_data: bool = False
    tokenization: str = "whitespace"
    vocab_path: str = "checkpoints/vocab.json"

    # Training
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "cuda"  # fallback handled in train

    # Sequence lengths
    max_frames: int = 120
    max_tgt_length: int = 128  # includes sos/eos

    # Model
    d_model: int = 256
    nhead: int = 8
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # Optimization / misc
    log_interval: int = 20
    val_interval: int = 1
    seed: int = 42
    resume_from: Optional[str] = None


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train OFTS-Net (minimal version)")

    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="root folder containing train/val (and optional test); omit to use built-in synthetic data under data/dummy_ofts",
    )
    p.add_argument(
        "--dummy-data",
        action="store_true",
        help="always (re)generate synthetic samples; optional with custom path via --dataset-root",
    )
    p.add_argument("--tokenization", type=str, default="whitespace", choices=["whitespace", "char"])
    p.add_argument("--vocab-path", type=str, default="checkpoints/vocab.json")

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--max-tgt-length", type=int, default=128)

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-decoder-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", type=str, default=None, help="checkpoint path to resume training")

    args = p.parse_args()
    default_dummy = os.path.abspath(os.path.join("data", "dummy_ofts"))
    # No --dataset-root → synthetic smoke-test data; --dummy-data forces (re)generation even if root was set.
    if args.dummy_data or args.dataset_root is None:
        dataset_root = args.dataset_root or default_dummy
        dummy_data = True
    else:
        dataset_root = args.dataset_root
        dummy_data = False

    cfg = TrainConfig(
        dataset_root=dataset_root,
        dummy_data=dummy_data,
        tokenization=args.tokenization,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        device=args.device,
        max_frames=args.max_frames,
        max_tgt_length=args.max_tgt_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        seed=args.seed,
        resume_from=args.resume_from,
    )
    # Normalize dataset path for less surprise.
    cfg.dataset_root = os.path.abspath(cfg.dataset_root)
    cfg.vocab_path = os.path.abspath(cfg.vocab_path)
    if cfg.resume_from:
        cfg.resume_from = os.path.abspath(cfg.resume_from)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep deterministic-ish (may affect speed).
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    # If user passes "cuda:0" etc.
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    return "cpu"

