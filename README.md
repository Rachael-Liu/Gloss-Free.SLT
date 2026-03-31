# OFTS-Net：无词汇手语翻译（Gloss-Free SLT）参考实现

基于论文 *Gloss-Free Sign Language Translation With Optical-Flow Guided Two-Stream Network (OFTS-Net)* 的 PyTorch 骨架：双流（RGB + 光流）、跨流注意力融合、Transformer 文本解码器，用于端到端手语翻译实验。

---

## 仓库目录说明

| 路径 | 作用 |
|------|------|
| **`src/`** | 核心源码：数据集、模型、训练入口、配置与词表。 |
| **`scripts/`** | 独立工具脚本：Kaggle 下载、Phoenix 数据转换、推理（`infer.py`）。 |
| **`data/`** | **数据根目录（默认不入库，体积大）**，常见子目录见下表。 |
| **`checkpoints/`** | **训练产出**：`oftsnet_best.pt`（最优验证损失时的权重）、`*.json` 词表等；可按实验重命名备份。 |
| **`.vscode/`** | 编辑器工作区配置（如 Python 解释器路径）。 |
| **`.idea/`** | JetBrains IDE 工程文件（可选）。 |
| **`requirements.txt`** | 训练/推理主依赖（含 `torch`、`opencv-python` 等）。 |
| **`requirements-kaggle.txt`** | 可选：仅用于 `scripts/download_kaggle_dataset.py` 的 Kaggle CLI。 |

### `data/` 下常见子目录

| 路径 | 作用 |
|------|------|
| **`data/kaggle_raw/`** | 从 Kaggle 下载并解压后的 **Phoenix 原始数据**（如 `videos_phoenix/`、`*.annotations_only.gzip`）。 |
| **`data/phoenix_ofts/`** | 由 `scripts/prepare_phoenix_to_ofts.py` 转换得到的 **全量或默认输出**（`train|val|test/<id>/rgb|flow|tgt.txt`）。 |
| **`data/phoenix_ofts_subset/`** | 同上脚本、但使用 `--limit-train` 等参数时的 **子集输出**，便于在有限磁盘上实验。 |
| **`data/dummy_ofts/`** | **合成冒烟数据**（不传 `--dataset-root` 或 `--dummy-data` 时自动生成）；仅用于验证管线，不用于真实论文指标。 |

> **说明**：`data/`、`checkpoints/` 体积可能很大；若使用 Git，建议用 `.gitignore` 忽略二者，仅保留代码与说明。

---

## `src/` 源码结构

| 文件/目录 | 作用 |
|-----------|------|
| **`train.py`** | 训练与验证主循环；支持 `--resume-from` 续训。 |
| **`utils/config.py`** | 命令行参数与 `TrainConfig`（`--dataset-root`、`--max-frames`、`--vocab-path` 等）。 |
| **`utils/vocab.py`** | 词表构建、编码/解码、保存为 JSON。 |
| **`data/dataset.py`** | `SignTranslationDataset`：读取 `rgb/*.jpg`、`flow/u|v/*.npy`、`tgt.txt`，`collate_fn` 组 batch。 |
| **`data/dummy_data.py`** | 生成极小合成样本目录（冒烟测试）。 |
| **`models/ofts_net.py`** | 整体网络：RGB 流 + 光流流 + 融合 + 解码器。 |
| **`models/rgb_stream.py`** | RGB 分支（MobileNetV3 + BiLSTM）。 |
| **`models/flow_stream.py`** | 光流分支（MobileNetV3 + 1D CNN）。 |
| **`models/cross_attention.py`** | 跨流注意力融合。 |
| **`models/decoder.py`** | Transformer 解码器（自回归训练时 teacher forcing）。 |

---

## `scripts/` 脚本

| 脚本 | 作用 |
|------|------|
| **`prepare_phoenix_to_ofts.py`** | 将 Kaggle Phoenix（`kaggle_raw`）转为训练所需目录结构；支持 `--out-root`、`--limit-*`、`--max-frames`、`--overwrite`。 |
| **`infer.py`** | 加载 checkpoint，对 **样本目录** 或 **`.mp4`** 做贪心解码，输出 `[pred]`（及可选 `[ref]`）。 |
| **`translate.py`** | **一键封装推理入口**：自动选择默认 `ckpt/vocab/device/max_frames`，只需传 `--video` 或 `--sample-dir`。 |
| **`download_kaggle_dataset.py`** | 调用本机 `kaggle` CLI 下载数据集；下载后通常仍需 `prepare_*` 整理为 `rgb/flow/tgt.txt`。 |

---

## 数据格式约定（训练与推理一致）

每个样本为一个目录：

```text
<dataset_root>/
  train|val|test/
    <sample_id>/
      rgb/000001.jpg ...
      flow/u/000001.npy ...
      flow/v/000001.npy ...
      tgt.txt
```

- **`--max-frames`**：转换脚本与 `train.py` / `infer.py` 应保持一致（或理解 padding 行为）。  
- **词表**：真实 Phoenix 实验请使用由 **该次训练数据 `train/*/tgt.txt`** 生成的词表（例如 `checkpoints/phoenix_vocab.json`），勿与 `dummy_ofts` 的小词表混用。

---

## 常用命令

### 环境

```powershell
py -3 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### Phoenix：转换（示例：子集 + 30 帧）

```powershell
.\.venv\Scripts\python.exe scripts\prepare_phoenix_to_ofts.py `
  --kaggle-root data\kaggle_raw `
  --out-root data\phoenix_ofts_subset `
  --limit-train 2000 --limit-val 200 --limit-test 200 `
  --max-frames 30 --overwrite
```

### 训练（示例：Phoenix 子集 + 独立词表）

```powershell
.\.venv\Scripts\python.exe -m src.train `
  --dataset-root data\phoenix_ofts_subset `
  --vocab-path checkpoints\phoenix_vocab.json `
  --max-frames 30 `
  --batch-size 1 --num-workers 0 `
  --epochs 5 --device cpu
```

### 推理

```powershell
.\.venv\Scripts\python.exe scripts\infer.py `
  --ckpt checkpoints\oftsnet_best.pt `
  --vocab-path checkpoints\phoenix_vocab.json `
  --sample-dir data\phoenix_ofts_subset\test\<样本目录名> `
  --max-frames 30 --device cpu
```

### 推理（一键封装：推荐）

只需要传入一个输入源（样本目录或 mp4），其余参数会自动选择：

- 默认 checkpoint：`checkpoints/oftsnet_best.pt`
- 默认词表：优先 `checkpoints/phoenix_vocab.json`，否则 `checkpoints/vocab.json`，若都不存在则从 checkpoint 内取
- `max_frames`：优先读取 checkpoint 的 `config.max_frames`，否则 120
- `device`：优先 CUDA（不可用则 CPU）

```powershell
# 1) 翻译 mp4
.\.venv\Scripts\python.exe scripts\translate.py --video path\to\video.mp4

# 2) 翻译已转换好的样本目录
.\.venv\Scripts\python.exe scripts\translate.py --sample-dir data\phoenix_ofts_subset\test\<样本目录名>
```

### 冒烟测试（无真实数据）

不传 `--dataset-root` 时会在 `data/dummy_ofts` 生成合成数据并训练（仅验证代码能跑通）：

```powershell
.\.venv\Scripts\python.exe -m src.train --device cpu --epochs 1
```

### （可选）Kaggle 下载

配置 [Kaggle API](https://www.kaggle.com/docs/api) 后：

```powershell
pip install -r requirements-kaggle.txt
python scripts\download_kaggle_dataset.py --slug 作者/数据集名 --out data\kaggle_download
```

---

## 注意

- 本仓库为**实验骨架**；Phoenix 等原始数据需自行按协议获取。  
- 光流在 `prepare_phoenix_to_ofts.py` 中默认使用 **OpenCV Farneback**；也可按论文使用 RAFT 等离线生成后对齐目录结构。  
- 推理为**贪心解码**；若多条样本输出相似套话，多与数据规模、训练轮数及 Phoenix 语料高频模板有关，需从训练与解码策略上改进。
