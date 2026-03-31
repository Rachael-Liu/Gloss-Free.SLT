from __future__ import annotations

import sys
from pathlib import Path
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as `python src/train.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import SignTranslationDataset, collate_fn
from src.data.dummy_data import ensure_dummy_ofts_dataset
from src.models.ofts_net import OFTSNet
from src.utils.vocab import Vocab
from src.utils.config import parse_args, resolve_device, set_seed


def _maybe_build_vocab(
    dataset_root: str,
    vocab_path: str,
    tokenization: str,
    max_tgt_length: int,
) -> Vocab:
    if os.path.isfile(vocab_path):
        print(f"[vocab] Loading existing vocab: {vocab_path}")
        return Vocab.load(vocab_path)

    print("[vocab] Building vocab from training split tgt.txt ...")
    # max_content_length excludes special tokens; keep it conservative.
    # Vocab building is only for token ids; the actual per-sample truncation is handled by dataset.
    max_content_length = max(0, max_tgt_length - 2)
    vocab = Vocab.from_dataset_root(
        dataset_root=dataset_root,
        split="train",
        tokenization=tokenization,
        min_freq=1,
        max_vocab_size=None,
        max_content_length=max_content_length,
    )
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    vocab.save(vocab_path)
    print(f"[vocab] Saved vocab: {vocab_path} (size={len(vocab)})")
    return vocab


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(dataloader, desc="val", leave=False):
        rgb = batch["rgb"].to(device)
        flow = batch["flow"].to(device)
        tgt = batch["tgt"].to(device)
        src_lengths = batch["lengths"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        logits = model(rgb=rgb, flow=flow, tgt_input=tgt_input, src_lengths=src_lengths)
        # logits: [B, L, vocab] -> [B*L, vocab]
        b, l, v = logits.shape
        loss = criterion(logits.reshape(b * l, v), tgt_target.reshape(b * l))

        # Track average loss (approx; ignore_index already drops pad positions in loss).
        total_loss += float(loss.item())
        total_count += 1

    return total_loss / max(1, total_count)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    grad_clip_norm: float,
    log_interval: int = 20,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(tqdm(dataloader, desc="train", leave=False), start=1):
        rgb = batch["rgb"].to(device)
        flow = batch["flow"].to(device)
        tgt = batch["tgt"].to(device)
        src_lengths = batch["lengths"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        logits = model(rgb=rgb, flow=flow, tgt_input=tgt_input, src_lengths=src_lengths)
        b, l, v = logits.shape

        loss = criterion(logits.reshape(b * l, v), tgt_target.reshape(b * l))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1
        if log_interval and (step % log_interval == 0):
            avg = total_loss / max(1, total_steps)
            print(f"[train] step={step} loss={loss.item():.4f} avg_loss={avg:.4f}")

    return total_loss / max(1, total_steps)


def main() -> None:
    cfg = parse_args()
    if cfg.dummy_data:
        ensure_dummy_ofts_dataset(cfg.dataset_root, seed=cfg.seed)
        print(f"[dummy-data] Synthetic dataset ready at: {cfg.dataset_root}")
    set_seed(cfg.seed)
    cfg.device = resolve_device(cfg.device)
    device = torch.device(cfg.device)

    if device.type == "cuda":
        print(f"[device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[device] Using CPU")

    # Vocab
    vocab = _maybe_build_vocab(
        dataset_root=cfg.dataset_root,
        vocab_path=cfg.vocab_path,
        tokenization=cfg.tokenization,
        max_tgt_length=cfg.max_tgt_length,
    )

    # Datasets
    train_root = os.path.join(cfg.dataset_root, "train")
    val_root = os.path.join(cfg.dataset_root, "val")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Training split not found: {train_root}")

    train_ds = SignTranslationDataset(
        root=cfg.dataset_root,
        vocab=vocab,
        split="train",
        max_frames=cfg.max_frames,
        max_tgt_length=cfg.max_tgt_length,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    val_loader: Optional[DataLoader] = None
    if os.path.isdir(val_root):
        val_ds = SignTranslationDataset(
            root=cfg.dataset_root,
            vocab=vocab,
            split="val",
            max_frames=cfg.max_frames,
            max_tgt_length=cfg.max_tgt_length,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )

    # Model
    model = OFTSNet(
        vocab_size=len(vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        pad_id=vocab.pad_id,
        max_len=cfg.max_tgt_length + 10,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Checkpoints
    ckpt_dir = os.path.dirname(cfg.vocab_path)
    best_val_loss = float("inf")
    best_path = os.path.join(ckpt_dir, "oftsnet_best.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 1

    if cfg.resume_from:
        if not os.path.isfile(cfg.resume_from):
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device)
        if "model_state" not in ckpt:
            raise KeyError(f"Invalid checkpoint format (missing 'model_state'): {cfg.resume_from}")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "best_val_loss" in ckpt:
            best_val_loss = float(ckpt["best_val_loss"])
        elif "val_loss" in ckpt:
            best_val_loss = float(ckpt["val_loss"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"[resume] Loaded checkpoint: {cfg.resume_from}")
        print(f"[resume] start_epoch={start_epoch} best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"\n[epoch] {epoch}/{cfg.epochs}")
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=vocab.pad_id,
            grad_clip_norm=cfg.grad_clip_norm,
            log_interval=cfg.log_interval,
        )
        print(f"[epoch] train_loss={train_loss:.4f}")

        if val_loader is not None and (epoch % cfg.val_interval == 0):
            val_loss = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                pad_id=vocab.pad_id,
            )
            print(f"[epoch] val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "vocab": vocab.to_json(),
                        "config": cfg.__dict__,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                    },
                    best_path,
                )
                print(f"[ckpt] Saved best model: {best_path}")


if __name__ == "__main__":
    main()

