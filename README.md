# Speculative Decoding Performance Benchmark Recipe

End-to-end recipe for benchmarking vLLM inference with speculative decoding (Eagle3) using GuideLLM and SPEED-Bench datasets.

## What This Covers

1. **Dataset preparation** - Preprocessing SPEED-Bench from HuggingFace into GuideLLM-compatible JSONL
2. **Model deployment** - Deploying vLLM via KServe (ServingRuntime + InferenceService)
3. **Benchmark execution** - Running GuideLLM with concurrent rate sweeps
4. **Results collection** - Copying JSON results from pods with checksum verification
5. **CSV conversion** - Converting GuideLLM JSON to performance dashboard CSV format

## Quick Start (Automated via Claude Workflow)

```bash
git clone https://github.com/MML-coder/spec-decode-perf-benchmark-recipe.git
cd spec-decode-perf-benchmark-recipe
cp config-examples/gpt-oss-120b-nvidia-eagle3.yaml config.yaml
# Edit config.yaml with your cluster details
claude /workflows spec-decode-benchmark
```

See [Section 8 of CLAUDE.md](CLAUDE.md#8-running-via-claude-workflow-automated) for full workflow details.

## Quick Links

- [Full Recipe (CLAUDE.md)](CLAUDE.md) - Step-by-step guide with command templates
- [SPEED-Bench Dataset Reference](datasets/speedbench.md) - Dataset splits, sample counts, preprocessing
- [Example Configs](config-examples/) - Pre-filled configs for common model/draft combinations

## Related Tools

| Tool | Repository | Description |
|------|------------|-------------|
| [preprocess_speedbench.py](https://github.com/MML-coder/ai-tools/blob/main/preprocess_speedbench.py) | MML-coder/ai-tools | Preprocess SPEED-Bench HF dataset into JSONL |
| [analyze_speedbench_isl.py](https://github.com/MML-coder/ai-tools/blob/main/analyze_speedbench_isl.py) | MML-coder/ai-tools | Analyze input sequence length distribution |
| [run_speedbench_gpt_oss_120b.py](https://github.com/MML-coder/ai-tools/blob/main/run_speedbench_gpt_oss_120b.py) | MML-coder/ai-tools | Automated multi-concurrency benchmark runner |
| [import_manual_runs_json_v2.py](https://github.com/openshift-psap/performance-dashboard/blob/main/manual_runs/scripts/import_manual_runs_json_v2.py) | openshift-psap/performance-dashboard | Convert GuideLLM JSON to dashboard CSV |

## Models Benchmarked

| Model | Draft Model | Method |
|-------|-------------|--------|
| `openai/gpt-oss-120b` | `RedHatAI/gpt-oss-120b-speculator.eagle3` | Eagle3 (RedHat) |
| `openai/gpt-oss-120b` | `nvidia/gpt-oss-120b-Eagle3-v2` | Eagle3 (Nvidia) |
| `google/gemma-4-31B-it` | `RedHatAI/gemma-4-31B-it-speculator.eagle3` | Eagle3 (RedHat) |
