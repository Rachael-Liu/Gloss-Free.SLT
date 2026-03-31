import os
import glob
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SignTranslationDataset(Dataset):
    """
    Gloss-free sign language translation dataset.

    Expected directory structure per split:

    root/
      train|val|test/
        video_id_1/
          rgb/
            000001.jpg
            ...
          flow/
            u/
              000001.npy
              ...
            v/
              000001.npy
              ...
          tgt.txt        # target sentence
    """

    def __init__(
        self,
        root: str,
        vocab,
        split: str = "train",
        max_frames: int = 120,
        max_tgt_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = os.path.join(root, split)
        self.vocab = vocab
        self.max_frames = max_frames
        self.max_tgt_length = max_tgt_length

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Split directory not found: {self.root}")

        self.samples: List[str] = sorted(
            [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        )

        self.rgb_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, vid_dir: str) -> Tuple[torch.Tensor, int]:
        rgb_dir = os.path.join(vid_dir, "rgb")
        frame_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        if not frame_paths:
            raise RuntimeError(f"No RGB frames found in {rgb_dir}")

        if len(frame_paths) > self.max_frames:
            indices = np.linspace(0, len(frame_paths) - 1, self.max_frames).astype(int)
            frame_paths = [frame_paths[i] for i in indices]

        imgs = [self.rgb_transform(Image.open(p).convert("RGB")) for p in frame_paths]
        t_len = len(imgs)

        if t_len < self.max_frames:
            pad_frames = [imgs[-1]] * (self.max_frames - t_len)
            imgs.extend(pad_frames)

        rgb = torch.stack(imgs, dim=0)  # [T, 3, 224, 224]
        return rgb, t_len

    def _load_flow(self, vid_dir: str, t_len: int) -> torch.Tensor:
        u_dir = os.path.join(vid_dir, "flow", "u")
        v_dir = os.path.join(vid_dir, "flow", "v")

        u_paths = sorted(glob.glob(os.path.join(u_dir, "*.npy")))
        v_paths = sorted(glob.glob(os.path.join(v_dir, "*.npy")))

        if not u_paths or not v_paths:
            raise RuntimeError(f"No flow files found in {u_dir} or {v_dir}")

        u_paths = u_paths[:t_len]
        v_paths = v_paths[:t_len]

        u = [np.load(p) for p in u_paths]
        v = [np.load(p) for p in v_paths]

        u_arr = np.stack(u, axis=0)  # [T, H, W]
        v_arr = np.stack(v, axis=0)

        flow = np.stack([u_arr, v_arr], axis=1)  # [T, 2, H, W]
        flow_tensor = torch.from_numpy(flow).float()

        t, c, h, w = flow_tensor.shape
        flow_tensor = torch.nn.functional.interpolate(
            flow_tensor.view(t * c, 1, h, w),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).view(t, c, 224, 224)

        if t < self.max_frames:
            pad = flow_tensor[-1:].repeat(self.max_frames - t, 1, 1, 1)
            flow_tensor = torch.cat([flow_tensor, pad], dim=0)

        return flow_tensor  # [T, 2, 224, 224]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid_name = self.samples[idx]
        vid_dir = os.path.join(self.root, vid_name)

        rgb, t_len = self._load_rgb(vid_dir)
        flow = self._load_flow(vid_dir, t_len)

        tgt_path = os.path.join(vid_dir, "tgt.txt")
        if not os.path.isfile(tgt_path):
            raise FileNotFoundError(f"Target text not found: {tgt_path}")

        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_sentence = f.read().strip()

        tgt_ids = self.vocab.encode(tgt_sentence, max_length=self.max_tgt_length)
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

        return {
            "rgb": rgb,  # [T, 3, 224, 224]
            "flow": flow,  # [T, 2, 224, 224]
            "tgt": tgt_tensor,
            "length": t_len,
            "id": vid_name,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_tgt_len = max(item["tgt"].size(0) for item in batch)

    rgbs = torch.stack([item["rgb"] for item in batch], dim=0)
    flows = torch.stack([item["flow"] for item in batch], dim=0)

    tgt_pad = []
    tgt_lengths = []
    for item in batch:
        tgt = item["tgt"]
        length = tgt.size(0)
        tgt_lengths.append(length)
        if length < max_tgt_len:
            pad = tgt.new_full((max_tgt_len - length,), fill_value=0)
            tgt = torch.cat([tgt, pad], dim=0)
        tgt_pad.append(tgt)

    tgt_pad_tensor = torch.stack(tgt_pad, dim=0)

    return {
        "rgb": rgbs,
        "flow": flows,
        "tgt": tgt_pad_tensor,
        "tgt_lengths": torch.tensor(tgt_lengths, dtype=torch.long),
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "ids": [item["id"] for item in batch],
    }

