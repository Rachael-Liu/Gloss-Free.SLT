from __future__ import annotations

import json
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


def _tokenize(sentence: str, mode: str) -> List[str]:
    sentence = (sentence or "").strip()
    if not sentence:
        return []

    if mode == "whitespace":
        return sentence.split()

    if mode == "char":
        # Remove whitespace but keep other characters.
        return [ch for ch in sentence if not ch.isspace()]

    raise ValueError(f"Unknown tokenization mode: {mode}")


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    tokenization: str = "whitespace"

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def build_from_tgt_files(
        cls,
        tgt_files: Iterable[str],
        tokenization: str = "whitespace",
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        max_content_length: Optional[int] = None,
    ) -> "Vocab":
        # Collect token frequencies from all tgt.txt.
        freq: Dict[str, int] = {}
        for p in tgt_files:
            with open(p, "r", encoding="utf-8") as f:
                sentence = f.read().strip()

            tokens = _tokenize(sentence, tokenization)
            if max_content_length is not None:
                tokens = tokens[:max_content_length]

            for t in tokens:
                freq[t] = freq.get(t, 0) + 1

        # Sort tokens by frequency then lexicographically for reproducibility.
        kept = [t for t, c in freq.items() if c >= min_freq]
        kept.sort(key=lambda x: (-freq[x], x))
        if max_vocab_size is not None:
            kept = kept[:max_vocab_size]

        # Create id mappings with fixed special token ids.
        id_to_token: List[str] = [
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["sos"],
            SPECIAL_TOKENS["eos"],
            SPECIAL_TOKENS["unk"],
        ]
        token_to_id: Dict[str, int] = {t: i for i, t in enumerate(id_to_token)}
        for t in kept:
            token_to_id[t] = len(id_to_token)
            id_to_token.append(t)

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad_id=token_to_id[SPECIAL_TOKENS["pad"]],
            sos_id=token_to_id[SPECIAL_TOKENS["sos"]],
            eos_id=token_to_id[SPECIAL_TOKENS["eos"]],
            unk_id=token_to_id[SPECIAL_TOKENS["unk"]],
            tokenization=tokenization,
        )

    @classmethod
    def from_dataset_root(
        cls,
        dataset_root: str,
        split: str = "train",
        tokenization: str = "whitespace",
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        max_content_length: Optional[int] = None,
    ) -> "Vocab":
        split_root = os.path.join(dataset_root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Dataset split not found: {split_root}")

        tgt_files = sorted(glob.glob(os.path.join(split_root, "*", "tgt.txt")))
        if not tgt_files:
            raise FileNotFoundError(f"No tgt.txt found under: {split_root}")

        return cls.build_from_tgt_files(
            tgt_files=tgt_files,
            tokenization=tokenization,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
            max_content_length=max_content_length,
        )

    def encode(
        self,
        sentence: str,
        add_sos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        tokens = _tokenize(sentence, self.tokenization)
        if max_length is not None:
            # max_length counts special tokens too.
            special = int(add_sos) + int(add_eos)
            content_max = max(0, max_length - special)
            tokens = tokens[:content_max]

        ids: List[int] = []
        if add_sos:
            ids.append(self.sos_id)
        for t in tokens:
            ids.append(self.token_to_id.get(t, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], remove_special: bool = True) -> str:
        tokens: List[str] = []
        for i in ids:
            token = self.id_to_token[int(i)]
            if remove_special and token in SPECIAL_TOKENS.values():
                continue
            tokens.append(token)
        if self.tokenization == "whitespace":
            return " ".join(tokens).strip()
        return "".join(tokens).strip()

    def to_json(self) -> Dict:
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "tokenization": self.tokenization,
        }

    @classmethod
    def from_json(cls, obj: Dict) -> "Vocab":
        token_to_id = {str(k): int(v) for k, v in obj["token_to_id"].items()}
        id_to_token = [str(x) for x in obj["id_to_token"]]
        tokenization = obj.get("tokenization", "whitespace")
        pad_id = token_to_id[SPECIAL_TOKENS["pad"]]
        sos_id = token_to_id[SPECIAL_TOKENS["sos"]]
        eos_id = token_to_id[SPECIAL_TOKENS["eos"]]
        unk_id = token_to_id[SPECIAL_TOKENS["unk"]]
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            unk_id=unk_id,
            tokenization=tokenization,
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_json(obj)

