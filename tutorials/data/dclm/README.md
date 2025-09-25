# DCLM Data Preprocessing Tutorial

This guide explains how to download, decompress, merge, and preprocess the **DCLM-baseline** dataset for language model pretraining.  

The **DCLM-baseline** dataset contains **4T tokens** across **3B documents**, achieving strong performance on language model benchmarks.

**Dataset source:** [Hugging Face DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/tree/main/global-shard_01_of_10)

---

## Dataset Overview

- The dataset is organized into **10 global shards**: `global-shard_01_of_10` … `global-shard_10_of_10`.  
- Each global shard contains **10 local shards**: `local-shard_0_of_10` … `local-shard_9_of_10`.  
- Each local shard contains ~280 compressed JSONL files: `*.jsonl.zst`.  
- Total dataset size: **~722 GB compressed**, **~2.1T decompressed**.


## Downloading Dataset

> **NOTE:**
This tutorial demonstrates preprocessing for a **single local shard**: global-shard_01_of_10/local-shard_0_of_10.

```bash

python3 download.py \
  --token HF_TOKEN \
  --num_workers 16 \
  --path_to_save /home/data/dclm \
  --patterns global-shard_01_of_10/local-shard_0_of_10/**

```

**Parameters:**
- `--token` — Hugging Face token for authentication.
- `--num_workers` — Number of parallel downloads; higher is faster.
- `--path_to_save` — Target directory for saving the dataset.
- `--patterns` — Subset of dataset to download. Optional; omit to download the full dataset.


## Decompressing Dataset

After downloading, decompress `.zst` files to `.jsonl`:

```bash

python3 decompress.py \
  --path_to_save /home/data/dclm/decompressed \
  --num_workers 32 \
  --source_dir /home/data/dclm/global-shard_01_of_10/local-shard_0_of_10

```

> **NOTE:**
Dependencies: parallel and zstd may need to be installed:

```bash
apt update
apt install parallel
apt install zstd
```

