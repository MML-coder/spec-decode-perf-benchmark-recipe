# SPEED-Bench Dataset Reference

Source: [nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) on HuggingFace

## Overview

SPEED-Bench is a benchmark dataset designed to evaluate LLM inference performance across varying input sequence lengths (ISL). It contains synthetic prompts at multiple ISL tiers, plus a qualitative split with real-world tasks.

## Splits

### Throughput Splits (Synthetic)

Each throughput split contains prompts padded to a target ISL range. Prompts are categorized as `high_entropy` (hard-to-predict tokens) or `low_entropy` (predictable tokens).

| Split | Samples | Word min | Word median | Word mean | Word max |
|-------|---------|----------|-------------|-----------|----------|
| throughput_1k | 897 | 61 | 508 | 525 | 998 |
| throughput_2k | 905 | 74 | 1,115 | 1,102 | 1,677 |
| throughput_8k | 808 | 277 | 4,418 | 4,444 | 6,353 |
| throughput_16k | 787 | 541 | 9,570 | 9,049 | 12,631 |
| throughput_32k | 824 | 1,130 | 18,924 | 17,910 | 25,512 |

Word counts are whitespace-split approximations (~1.3 tokens per word for English text). For accurate token counts, use:
```bash
python3 analyze_speedbench_isl.py --data-dir /tmp/speed-bench-processed --tokenizer <model_id>
```

### Qualitative Split (Real-world tasks)

| Split | Samples |
|-------|---------|
| qualitative | 386 |

Contains real-world prompts: summarization, creative writing, reasoning, code generation. Varied ISL distribution (mostly <256 tokens with long tail).

### Combined Splits

| Split | Samples | Description |
|-------|---------|-------------|
| combined | 4,607 | All splits merged |
| throughput_1k_8k | 1,705 | throughput_1k + throughput_8k merged |

## Preprocessing

Raw SPEED-Bench has a `turns` column (list of strings). Preprocessing flattens this into a single `prompt` string.

### Steps

```bash
# Install dependencies
pip install datasets

# Preprocess all splits
python3 preprocess_speedbench.py --config all --output-dir /tmp/speed-bench-processed

# Or a single split
python3 preprocess_speedbench.py --config throughput_1k --output-dir /tmp/speed-bench-processed
```

Script: [`preprocess_speedbench.py`](https://github.com/MML-coder/ai-tools/blob/main/preprocess_speedbench.py)

### Output Structure

```
/tmp/speed-bench-processed/
  qualitative/
    all.jsonl              # All categories combined
    high_entropy.jsonl     # Per-category files
    low_entropy.jsonl
    stats.json             # Processing statistics
  throughput_1k/
    all.jsonl
    high_entropy.jsonl
    low_entropy.jsonl
    stats.json
  throughput_2k/
    ...
```

### JSONL Format

Each line is a JSON object:
```json
{"prompt": "I want you to act as a book author...", "category": "high_entropy"}
```

## ISL Analysis

Use [`analyze_speedbench_isl.py`](https://github.com/MML-coder/ai-tools/blob/main/analyze_speedbench_isl.py) to get per-split token distribution with ISL buckets:

```bash
python3 analyze_speedbench_isl.py --data-dir /tmp/speed-bench-processed

# With accurate tokenizer (requires transformers library)
python3 analyze_speedbench_isl.py --data-dir /tmp/speed-bench-processed --tokenizer openai/gpt-oss-120b
```

ISL buckets: `<256`, `256-512`, `512-1K`, `1K-2K`, `2K-4K`, `4K-8K`, `8K-16K`, `16K-32K`, `32K+`

## Uploading to Cluster

Copy preprocessed datasets to a PVC accessible by the GuideLLM pod:

```bash
# Create a tar of the processed datasets
tar czf speed-bench-processed.tar.gz -C /tmp speed-bench-processed

# Copy to pod
oc cp speed-bench-processed.tar.gz <namespace>/<pod>:/data/datasets/

# Extract on pod
oc exec -n <namespace> <pod> -- tar xzf /data/datasets/speed-bench-processed.tar.gz -C /data/datasets/
```

Expected pod path: `/data/datasets/speed-bench-processed/<split>/all.jsonl`
