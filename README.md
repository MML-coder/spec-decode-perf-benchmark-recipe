# Speculative Decoding Performance Benchmark Recipe

End-to-end recipe for benchmarking vLLM inference with speculative decoding (Eagle3) using GuideLLM and SPEED-Bench datasets.

## What This Covers

1. **Dataset preparation** - Preprocessing SPEED-Bench from HuggingFace into GuideLLM-compatible JSONL
2. **Model deployment** - Deploying vLLM via KServe (ServingRuntime + InferenceService)
3. **Benchmark execution** - Running GuideLLM with concurrent rate sweeps
4. **Results collection** - Copying JSON results from pods with checksum verification
5. **CSV conversion** - Converting GuideLLM JSON to performance dashboard CSV format

## Quick Links

- [Full Recipe (CLAUDE.md)](CLAUDE.md) - Step-by-step guide with command templates
- [SPEED-Bench Dataset Reference](datasets/speedbench.md) - Dataset splits, sample counts, preprocessing

## Related Tools

| Tool | Repository | Description |
|------|------------|-------------|
| [preprocess_speedbench.py](https://github.com/MML-coder/ai-tools/blob/main/preprocess_speedbench.py) | MML-coder/ai-tools | Preprocess SPEED-Bench HF dataset into JSONL |
| [analyze_speedbench_isl.py](https://github.com/MML-coder/ai-tools/blob/main/analyze_speedbench_isl.py) | MML-coder/ai-tools | Analyze input sequence length distribution |
| [run_speedbench_gpt_oss_120b.py](https://github.com/MML-coder/ai-tools/blob/main/run_speedbench_gpt_oss_120b.py) | MML-coder/ai-tools | Automated multi-concurrency benchmark runner |

## Models Benchmarked

| Model | Draft Model | Method |
|-------|-------------|--------|
| `openai/gpt-oss-120b` | `RedHatAI/gpt-oss-120b-speculator.eagle3` | Eagle3 (RedHat) |
| `openai/gpt-oss-120b` | `nvidia/gpt-oss-120b-Eagle3-v2` | Eagle3 (Nvidia) |
| `google/gemma-4-31B-it` | `RedHatAI/gemma-4-31B-it-speculator.eagle3` | Eagle3 (RedHat) |
