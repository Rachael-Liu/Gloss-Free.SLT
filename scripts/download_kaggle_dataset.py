"""
Download a Kaggle dataset using the official CLI (needs API credentials).

Setup:
  pip install kaggle
  Place kaggle.json in ~/.kaggle/ (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)

Usage:
  python scripts/download_kaggle_dataset.py --slug username/dataset-name --out data/kaggle_raw

Most Kaggle archives will NOT match this repo's rgb/flow/tgt.txt layout; you still need
offline preprocessing to reorganize files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Download & unzip a Kaggle dataset via the kaggle CLI.")
    ap.add_argument("--slug", required=True, help='Dataset slug, e.g. "author/my-dataset"')
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "kaggle_download",
        help="Directory to write the zip / extracted files (created if missing).",
    )
    ap.add_argument("--no-unzip", action="store_true", help="Only download the zip, do not unzip.")
    args = ap.parse_args()

    kaggle_exe = shutil.which("kaggle")
    if not kaggle_exe:
        print(
            "未找到 kaggle 命令。请先: pip install kaggle，并在用户目录配置 API 密钥 (kaggle.json)。",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle_exe, "datasets", "download", "-d", args.slug, "-p", str(args.out)]
    if not args.no_unzip:
        cmd.append("--unzip")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return 1
    print(f"完成: 输出目录 {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
