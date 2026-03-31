"""
Greedy decoding inference: sign video (rgb + flow) -> text.

Usage (converted sample folder, same layout as training):
  python scripts/infer.py --ckpt checkpoints/oftsnet_best.pt --sample-dir data/phoenix_ofts/test/01November_2010_Monday_tagesschau-133

Usage (raw mp4, same OpenCV Farneback preprocessing as prepare_phoenix_to_ofts.py):
  python scripts/infer.py --ckpt checkpoints/oftsnet_best.pt --video path/to/video.mp4 --max-frames 90
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Project root (parent of scripts/)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ofts_net import OFTSNet
from src.utils.vocab import Vocab

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 opencv-python：pip install opencv-python") from e


def _rgb_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_tensors_from_sample_dir(sample_dir: str, max_frames: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return rgb [T,3,224,224], flow [T,2,224,224], t_len (unpadded length before temporal pad)."""
    tfm = _rgb_transform()
    rgb_dir = os.path.join(sample_dir, "rgb")
    frame_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")

    if len(frame_paths) > max_frames:
        indices = np.linspace(0, len(frame_paths) - 1, max_frames).astype(int)
        frame_paths = [frame_paths[i] for i in indices]

    imgs = [tfm(Image.open(p).convert("RGB")) for p in frame_paths]
    t_len = len(imgs)
    if t_len < max_frames:
        imgs.extend([imgs[-1]] * (max_frames - t_len))
    rgb = torch.stack(imgs, dim=0)

    u_dir = os.path.join(sample_dir, "flow", "u")
    v_dir = os.path.join(sample_dir, "flow", "v")
    u_paths = sorted(glob.glob(os.path.join(u_dir, "*.npy")))
    v_paths = sorted(glob.glob(os.path.join(v_dir, "*.npy")))
    if not u_paths or not v_paths:
        raise FileNotFoundError(f"Missing flow under {sample_dir}/flow")

    u_paths = u_paths[:t_len]
    v_paths = v_paths[:t_len]
    u_arr = np.stack([np.load(p) for p in u_paths], axis=0)
    v_arr = np.stack([np.load(p) for p in v_paths], axis=0)
    flow_np = np.stack([u_arr, v_arr], axis=1)
    flow_tensor = torch.from_numpy(flow_np).float()
    t, c, h, w = flow_tensor.shape
    flow_tensor = torch.nn.functional.interpolate(
        flow_tensor.view(t * c, 1, h, w),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).view(t, c, 224, 224)
    if t < max_frames:
        pad = flow_tensor[-1:].repeat(max_frames - t, 1, 1, 1)
        flow_tensor = torch.cat([flow_tensor, pad], dim=0)

    return rgb, flow_tensor, t_len


def _write_rgb_and_flow_from_mp4(
    mp4_path: Path,
    rgb_dir: Path,
    u_dir: Path,
    v_dir: Path,
    max_frames: int | None,
    resize_hw: Tuple[int, int] = (224, 224),
) -> int:
    """Same logic as scripts/prepare_phoenix_to_ofts.py."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    prev_gray: np.ndarray | None = None
    frame_idx = 0
    rgb_dir.mkdir(parents=True, exist_ok=True)
    u_dir.mkdir(parents=True, exist_ok=True)
    v_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        frame = cv2.resize(frame, resize_hw[::-1], interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(rgb_dir / f"{frame_idx:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

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


@torch.no_grad()
def encode_memory(
    model: OFTSNet,
    rgb: torch.Tensor,
    flow: torch.Tensor,
    src_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """rgb/flow: [B,T,...], src_lengths: [B]. Returns fused [B,T,d], memory_padding_mask [B,T]."""
    device = rgb.device
    b, t, _, _, _ = rgb.shape
    rgb_seq = model.rgb_stream(rgb)
    flow_seq = model.flow_stream(flow)
    ar = torch.arange(t, device=device)[None, :]
    memory_padding_mask = ar >= src_lengths[:, None]
    fused = model.fusion(rgb_seq, flow_seq, flow_padding_mask=memory_padding_mask)
    return fused, memory_padding_mask


@torch.no_grad()
def greedy_decode(
    model: OFTSNet,
    fused: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    vocab: Vocab,
    device: torch.device,
    max_decode_steps: int,
    min_tokens_before_eos: int = 2,
    forbid_early_eos: bool = True,
) -> List[int]:
    """
    Greedy decode. Many under-trained checkpoints put highest mass on <eos> at the first step;
    masking <eos> for the first few steps avoids an empty decoded string.
    """
    tokens: List[int] = [vocab.sos_id]
    pad_id = vocab.pad_id
    for _ in range(max_decode_steps):
        tgt_input = torch.tensor([tokens], dtype=torch.long, device=device)
        tgt_padding_mask = tgt_input.eq(pad_id)
        logits = model.decoder(
            tgt_input=tgt_input,
            memory=fused,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )
        logits_last = logits[0, -1].detach().clone()
        n_after_sos = len(tokens) - 1
        if forbid_early_eos and n_after_sos < min_tokens_before_eos:
            logits_last[vocab.eos_id] = float("-inf")
        next_id = int(logits_last.argmax(dim=-1).item())
        tokens.append(next_id)
        if next_id == vocab.eos_id:
            break
    return tokens


def _build_model(ckpt: Dict[str, Any], vocab: Vocab, device: torch.device) -> OFTSNet:
    cfg = ckpt.get("config") or {}
    max_len = int(cfg.get("max_tgt_length", 128)) + 10
    model = OFTSNet(
        vocab_size=len(vocab),
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        num_decoder_layers=int(cfg.get("num_decoder_layers", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_id=vocab.pad_id,
        max_len=max_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="OFTS-Net greedy inference (video -> text).")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt (e.g. checkpoints/oftsnet_best.pt)")
    ap.add_argument("--vocab-path", type=str, default=None, help="override vocab json; default: use ckpt['vocab']")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample-dir", type=str, help="folder with rgb/ and flow/u,v/ (training layout)")
    g.add_argument("--video", type=str, help="path to .mp4 (decoded with same OpenCV pipeline as Phoenix prep)")
    ap.add_argument("--max-frames", type=int, default=None, help="must match training; default from ckpt config or 120")
    ap.add_argument("--max-decode", type=int, default=None, help="max new tokens after <sos>; default max_tgt_length from ckpt")
    ap.add_argument(
        "--min-tokens-before-eos",
        type=int,
        default=2,
        help="first N predictions cannot be <eos> (avoids empty [pred] when step-1 argmax is <eos>)",
    )
    ap.add_argument(
        "--allow-immediate-eos",
        action="store_true",
        help="do not mask <eos>; pure greedy (may yield empty string)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device_s = args.device
    if device_s.startswith("cuda") and not torch.cuda.is_available():
        device_s = "cpu"
    device = torch.device(device_s)

    ckpt_path = os.path.abspath(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)

    if args.vocab_path:
        vocab = Vocab.load(os.path.abspath(args.vocab_path))
    else:
        if "vocab" not in ckpt:
            raise SystemExit("checkpoint 中没有 vocab 字段，请传 --vocab-path")
        vocab = Vocab.from_json(ckpt["vocab"])

    cfg = ckpt.get("config") or {}
    max_frames = args.max_frames if args.max_frames is not None else int(cfg.get("max_frames", 120))
    max_decode = args.max_decode if args.max_decode is not None else int(cfg.get("max_tgt_length", 128))

    if args.sample_dir:
        sample_dir = os.path.abspath(args.sample_dir)
        rgb, flow, t_len = load_tensors_from_sample_dir(sample_dir, max_frames)
    else:
        mp4 = Path(os.path.abspath(args.video))
        if not mp4.is_file():
            raise FileNotFoundError(str(mp4))
        tmp = tempfile.mkdtemp(prefix="ofts_infer_")
        try:
            _write_rgb_and_flow_from_mp4(
                mp4,
                Path(tmp) / "rgb",
                Path(tmp) / "flow" / "u",
                Path(tmp) / "flow" / "v",
                max_frames,
            )
            rgb, flow, t_len = load_tensors_from_sample_dir(tmp, max_frames)
        finally:
            # keep tmp for debug if needed; remove to save disk
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    rgb_b = rgb.unsqueeze(0).to(device)
    flow_b = flow.unsqueeze(0).to(device)
    src_lengths = torch.tensor([t_len], dtype=torch.long, device=device)

    model = _build_model(ckpt, vocab, device)
    fused, mem_pad = encode_memory(model, rgb_b, flow_b, src_lengths)
    ids = greedy_decode(
        model,
        fused,
        mem_pad,
        vocab,
        device,
        max_decode_steps=max_decode,
        min_tokens_before_eos=max(0, int(args.min_tokens_before_eos)),
        forbid_early_eos=not bool(args.allow_immediate_eos),
    )
    text = vocab.decode(ids, remove_special=True)

    ref: str | None = None
    if args.sample_dir:
        ref_path = os.path.join(os.path.abspath(args.sample_dir), "tgt.txt")
        if os.path.isfile(ref_path):
            with open(ref_path, "r", encoding="utf-8") as f:
                ref = f.read().strip()

    print("[pred]", text if text else "(empty)")
    if ref is not None:
        print("[ref ]", ref)

    cfg_ds = cfg.get("dataset_root", "")
    if "dummy_ofts" in str(cfg_ds) and args.sample_dir and "phoenix_ofts" in str(args.sample_dir):
        print(
            "[warn] checkpoint 是在 dummy_ofts + 小词表上训练的；"
            "Phoenix 德语参考句不在该词表中，[pred] 只能是少量英文词或 <unk>，与 [ref] 不可比。"
            "若要德语输出，请用 Phoenix 训练集重建词表并重新训练后再推理。"
        )
    if not text:
        toks = [vocab.id_to_token[i] for i in ids]
        print("[debug] id sequence:", ids[:40])
        print("[debug] tokens:", toks[:40])
        print(
            "[hint] 若 [pred] 仍为空：常见为第一步 argmax 即 <eos>（已默认屏蔽前几步 <eos>），"
            "或连续预测 <unk>/<pad>（decode 会去掉特殊符号）。可加 --allow-immediate-eos 对照原始贪心。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
