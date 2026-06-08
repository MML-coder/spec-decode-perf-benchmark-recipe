# Speculative Decoding Performance Benchmark Recipe

End-to-end process for benchmarking vLLM inference with speculative decoding using GuideLLM and SPEED-Bench datasets.

---

## Quick Start: Step-by-Step Overview

Follow these steps in order. Each step links to the detailed section below.

### Step 1: Prepare the Dataset
- Download and preprocess the SPEED-Bench dataset from HuggingFace
- Upload the processed JSONL files to a PVC on the cluster
- See [Section 1: SPEED-Bench Datasets](#1-speed-bench-datasets)

### Step 2: Create the GuideLLM Client Pod
- Create a PVC for datasets and results storage
- Deploy a lightweight Python pod with GuideLLM installed
- Upload preprocessed datasets to the PVC
- See [Section 2: GuideLLM Client Pod Setup](#2-guidellm-client-pod-setup)

### Step 3: Deploy the Model
- Deploy vLLM as a KServe InferenceService on your cluster
- Configure speculative decoding (or run baseline without it)
- Note the endpoint URL for the next step
- See [Section 3: vLLM Deployment](#3-vllm-deployment-kserve)

### Step 4: Run the Benchmark
- Execute GuideLLM from the client pod to hit the model endpoint
- Choose multi-concurrency (one run) or per-concurrency (separate runs)
- Wait for completion — output is a JSON file on the pod
- See [Section 4: GuideLLM Benchmark Execution](#4-guidellm-benchmark-execution)

### Step 5: Copy Results Locally
- Copy JSON result files from the pod to your local machine
- **Always verify MD5 checksums** — large file copies can silently corrupt
- See [Section 5: Copying Results from Pod](#5-copying-results-from-pod)

### Step 6: Convert JSON to CSV
- Run the import script to extract metrics into a dashboard-compatible CSV
- For per-concurrency JSONs, run once per file (script appends automatically)
- See [Section 6: JSON to CSV Conversion](#6-json-to-csv-conversion)

### Step 7: Review Results
- Open the CSV to compare throughput, latency, and TTFT across concurrency levels
- Compare baseline vs speculative decoding runs side by side
- See [Section 7: Concrete Examples](#7-concrete-examples) for real command examples

---

## 1. SPEED-Bench Datasets

**Source**: [nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) on HuggingFace

### Preprocessing

Raw SPEED-Bench has a `turns` column (list of strings). Preprocessing flattens turns into a single `prompt` string and splits into per-category JSONL files.

```bash
# Preprocess all splits
python3 preprocess_speedbench.py --config all --output-dir /tmp/speed-bench-processed
```

- Script: [`preprocess_speedbench.py`](https://github.com/MML-coder/ai-tools/blob/main/preprocess_speedbench.py)
- Analysis: [`analyze_speedbench_isl.py`](https://github.com/MML-coder/ai-tools/blob/main/analyze_speedbench_isl.py)

### Dataset Stats

| Split | Samples | Word min | Word median | Word mean | Word max |
|-------|---------|----------|-------------|-----------|----------|
| qualitative | 386 | - | - | - | - |
| throughput_1k | 897 | 61 | 508 | 525 | 998 |
| throughput_2k | 905 | 74 | 1,115 | 1,102 | 1,677 |
| throughput_8k | 808 | 277 | 4,418 | 4,444 | 6,353 |
| throughput_16k | 787 | 541 | 9,570 | 9,049 | 12,631 |
| throughput_32k | 824 | 1,130 | 18,924 | 17,910 | 25,512 |
| combined | 4,607 | - | - | - | - |
| throughput_1k_8k | 1,705 | - | - | - | - |

Word counts are whitespace-split approximations. For accurate token counts, use `analyze_speedbench_isl.py --tokenizer <model_id>`.

### JSONL Format

```json
{"prompt": "...", "category": "high_entropy"}
```

Categories: `high_entropy` (hard-to-predict tokens), `low_entropy` (predictable tokens).

### Pod Path Convention

Datasets are stored on a PVC mounted at `/data/datasets/`:

```
/data/datasets/speed-bench-processed/
  throughput_1k/all.jsonl     # 897 samples
  throughput_8k/all.jsonl     # 808 samples
  throughput_16k/all.jsonl    # 787 samples
  ...
```

See [`datasets/speedbench.md`](datasets/speedbench.md) for detailed per-split breakdown.

---

## 2. GuideLLM Client Pod Setup

GuideLLM runs from a lightweight client pod inside the cluster (so it can reach the vLLM endpoint via cluster-internal DNS). The pod needs a PVC for storing datasets and benchmark results.

### 2.1 Create the Storage PVC

The PVC stores preprocessed datasets and benchmark output JSON files. Create it once per namespace.

Manifest: [`manifests/client-storage-pvc.yaml`](manifests/client-storage-pvc.yaml)

```bash
# Edit the namespace in the manifest, then apply
oc apply -f manifests/client-storage-pvc.yaml
```

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: client-storage
  namespace: <NAMESPACE>
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs-provisioner
```

Adjust `storageClassName` for your cluster (e.g., `gp3-csi`, `ocs-storagecluster-cephfs`). 100Gi is sufficient for all SPEED-Bench splits plus benchmark results.

### 2.2 Deploy the GuideLLM Pod

The pod uses `python:3.11-slim` as a base and installs GuideLLM on startup. It stays running via `tail -f /dev/null` so you can `oc exec` into it.

Manifest: [`manifests/guidellm-pod.yaml`](manifests/guidellm-pod.yaml)

```bash
# Replace <USER> and <NAMESPACE> in the manifest, then apply
sed 's/<USER>/yourname/g; s/<NAMESPACE>/kserve-e2e-perf/g' manifests/guidellm-pod.yaml | oc apply -f -
```

What the pod does on startup:
1. Installs system tools: `tmux`, `vim`, `git`, `curl`, `wget`, `procps`
2. Installs GuideLLM v0.6.0 from the vllm-project GitHub repo
3. Mounts the PVC at `/data`
4. Stays running for interactive access

Wait for the pod to be ready (GuideLLM installation takes ~2 minutes):

```bash
# Watch pod status
oc get pod guidellm-<USER> -n <NAMESPACE> -w

# Check logs to confirm installation completed
oc logs guidellm-<USER> -n <NAMESPACE>

# Should see: "Installation complete!" and "Pod is ready."
```

### 2.3 Upload Datasets to the PVC

Once the pod is running, copy preprocessed SPEED-Bench datasets to it:

```bash
# Copy from local machine to pod PVC
oc cp /tmp/speed-bench-processed <NAMESPACE>/guidellm-<USER>:/data/datasets/speed-bench-processed

# Verify on pod
oc exec -n <NAMESPACE> guidellm-<USER> -- ls /data/datasets/speed-bench-processed/
# Expected: qualitative  throughput_1k  throughput_2k  throughput_8k  throughput_16k  throughput_32k
```

### 2.4 Create Results Directories

Create a directory on the PVC for benchmark output before running:

```bash
oc exec -n <NAMESPACE> guidellm-<USER> -- mkdir -p /data/results
```

### 2.5 Verify Setup

```bash
# Confirm guidellm is installed
oc exec -n <NAMESPACE> guidellm-<USER> -- guidellm --version

# Confirm datasets are available
oc exec -n <NAMESPACE> guidellm-<USER> -- wc -l /data/datasets/speed-bench-processed/throughput_1k/all.jsonl
# Expected: 897

# Interactive access (for running benchmarks manually)
oc exec -it -n <NAMESPACE> guidellm-<USER> -- bash
```

### Resource Notes

- **CPU/Memory**: 4 CPU / 8Gi memory limits. GuideLLM is CPU-bound (no GPU needed) — it sends HTTP requests to the vLLM endpoint. The original pod with 4Gi was OOMKilled under high concurrency, so 8Gi is recommended.
- **PVC**: `ReadWriteOnce` is fine — only this pod mounts it. If you need multiple pods sharing results, use `ReadWriteMany` with an NFS-backed storageClass.
- **SecurityContext**: `runAsUser: 0` is needed for `apt-get install`. If your cluster doesn't allow root, pre-build a custom image with the dependencies baked in.

---

## 3. vLLM Deployment (KServe)

### KServe Pattern

Deploy via **ServingRuntime** + **InferenceService** (two YAML documents, single `oc apply`).

- ServingRuntime: defines the vLLM container, command, args, env vars
- InferenceService: references the ServingRuntime, sets `storageUri` for model weights

### Deployment Naming Convention

Derive from HuggingFace model ID:

```
HF model ID → split on "/" → take second part → lowercase → replace "." with "-"
```

Examples:
- `Qwen/Qwen3-0.6B` -> `qwen3-0-6b`
- `google/gemma-4-31B-it` -> `gemma-4-31b-it`
- `openai/gpt-oss-120b` -> `gpt-oss-120b`

### Key vLLM Args

```yaml
# Common baseline args
--gpu-memory-utilization 0.85
--trust-remote-code
--disable-log-requests
--uvicorn-log-level debug
--tensor-parallel-size 1
--no-enable-prefix-caching
```

### Speculative Decoding Config

Pass as `--speculative-config` JSON:

```json
{
  "model": "RedHatAI/gpt-oss-120b-speculator.eagle3",
  "num_speculative_tokens": 3,
  "method": "eagle3",
  "draft_tensor_parallel_size": 1
}
```

Other draft models:
- `nvidia/gpt-oss-120b-Eagle3-v2` (Nvidia draft)
- `RedHatAI/gemma-4-31B-it-speculator.eagle3` (RedHat gemma draft)

### Container Images

| Type | Image |
|------|-------|
| RHAIIS (Red Hat) | `quay.io/aipcc/rhaiis/cuda-ubi9:<tag>` |
| Upstream vLLM | `vllm/vllm-openai:<tag>` |

### Endpoint URL Pattern

```
http://<deployment-name>-predictor.<namespace>.svc.cluster.local:8080/v1
```

---

## 4. GuideLLM Benchmark Execution

### Two Run Modes

#### Mode A: Multi-concurrency (single JSON output)

All concurrency levels in one run. Produces one JSON with N benchmark sections.

```bash
guidellm benchmark \
  --target http://<deployment>-predictor.<ns>.svc.cluster.local:8080/v1 \
  --data /data/datasets/speed-bench-processed/throughput_1k/all.jsonl \
  --rate-type concurrent \
  --rate 1,5,25,50,100 \
  --output-path /data/results/speedbench-throughput-1k.json
```

#### Mode B: Per-concurrency (separate JSON files)

One run per concurrency level. Useful when runs are long or you want to start/stop individually.

```bash
for c in 1 5 25 50 100; do
  guidellm benchmark \
    --target http://<deployment>-predictor.<ns>.svc.cluster.local:8080/v1 \
    --data /data/datasets/speed-bench-processed/throughput_1k/all.jsonl \
    --rate-type concurrent \
    --rate $c \
    --max-requests 100 \
    --output-path /data/results/c${c}.json
done
```

### Execution from GuideLLM Pod

```bash
oc exec -n <namespace> guidellm-<USER> -- \
  guidellm benchmark \
    --target <endpoint-url> \
    --data <dataset-path> \
    --rate-type concurrent \
    --rate <concurrency> \
    --output-path <output-path>
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--target` | vLLM endpoint URL (with `/v1`) | `http://gpt-oss-120b-predictor.ns.svc.cluster.local:8080/v1` |
| `--data` | Path to JSONL dataset on pod | `/data/datasets/speed-bench-processed/throughput_1k/all.jsonl` |
| `--rate-type` | Always `concurrent` for throughput tests | `concurrent` |
| `--rate` | Concurrency level(s) | `1,5,25,50,100` or single value |
| `--max-requests` | Cap requests per concurrency (per-c mode) | `100` |
| `--output-path` | JSON output file path on pod | `/data/results/output.json` |

### Automated Runner

For scripted multi-concurrency runs: [`run_speedbench_gpt_oss_120b.py`](https://github.com/MML-coder/ai-tools/blob/main/run_speedbench_gpt_oss_120b.py)

---

## 5. Copying Results from Pod

### Standard Copy (small-medium files, < ~100MB)

```bash
oc cp <namespace>/<pod>:/data/results/output.json ./output.json --retries=5
```

### Large Files (> 100MB)

`oc cp` may fail with EOF on large files. Use `--retries=5` which handles resume:

```bash
oc cp kserve-e2e-perf/guidellm-<USER>:/data/results/large-file.json ./large-file.json --retries=5
```

If `oc cp` still fails, fall back to `oc exec cat` (but always verify checksum — `cat` over `oc exec` can corrupt binary-ish content):

```bash
oc exec -n <ns> <pod> -- cat /data/results/output.json > ./output.json
```

### MD5 Verification (mandatory)

Always verify checksums after copy:

```bash
# Remote checksum
oc exec -n <ns> <pod> -- md5sum /data/results/output.json

# Local checksum (macOS)
md5 -r ./output.json

# Local checksum (Linux)
md5sum ./output.json
```

Checksums MUST match. If they don't, re-copy the file.

---

## 6. JSON to CSV Conversion

### Import Script

The conversion script lives in the [openshift-psap/performance-dashboard](https://github.com/openshift-psap/performance-dashboard) repo. Clone it first:

```bash
git clone https://github.com/openshift-psap/performance-dashboard.git
cd performance-dashboard/manual_runs/scripts
```

Script: [`import_manual_runs_json_v2.py`](https://github.com/openshift-psap/performance-dashboard/blob/main/manual_runs/scripts/import_manual_runs_json_v2.py)

Requires: `pandas` (use virtualenv with pandas installed)

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas
```

### Mode A: Multi-concurrency JSON (single file with all concurrency levels)

One JSON file containing 5 benchmark sections produces a 5-row CSV:

```bash
source ~/test_foo/python3_virt/bin/activate

python import_manual_runs_json_v2.py \
  speedbench-throughput-1k.json \
  --model "openai/gpt-oss-120b" \
  --version "RHAIIS-3.5.EA1" \
  --tp 1 \
  --accelerator "H200" \
  --runtime-args 'uvicorn-log-level: debug; trust-remote-code: True; tensor-parallel-size: 1; gpu-memory-utilization: 0.85; disable-log-requests: True; enable-expert-parallel: True; no-enable-prefix-caching: True; max-cudagraph-capture-size: 2048; speculative-config:{"model":"RedHatAI/gpt-oss-120b-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3","draft_tensor_parallel_size":1}' \
  --image-tag "quay.io/aipcc/rhaiis/cuda-ubi9:3.5.0-ea.1-1779124632" \
  --guidellm-version "v0.6.0" \
  --dataset "SPEEDBENCH-1k" \
  --spec-decoding "eagle3-redhat" \
  --prefix-caching "yes" \
  --csv-file "throughput_1k_rh_draft.csv"
```

### Mode B: Per-concurrency JSONs (separate files per concurrency)

Run the script once per file. It **appends** to the same CSV:

```bash
source ~/test_foo/python3_virt/bin/activate

for c in c1 c5 c25 c50 c100; do
  python import_manual_runs_json_v2.py \
    /path/to/results/${c}.json \
    --model "google/gemma-4-31B-it" \
    --version "vLLM-0.22.0" \
    --tp 1 \
    --accelerator "H200" \
    --runtime-args 'uvicorn-log-level: debug; trust-remote-code: True; tensor-parallel-size: 1; gpu-memory-utilization: 0.85; disable-log-requests: True; no-enable-prefix-caching: True' \
    --image-tag "vllm/vllm-openai:v0.22.0" \
    --guidellm-version "v0.6.0" \
    --dataset "SPEEDBENCH-1k" \
    --spec-decoding "baseline" \
    --prefix-caching "yes" \
    --csv-file "throughput_gemma-1k_baseline.csv"
done
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `json_file` | Path to GuideLLM JSON output | `speedbench-throughput-1k.json` |
| `--model` | HuggingFace model ID | `openai/gpt-oss-120b` |
| `--version` | Framework version identifier | `RHAIIS-3.5.EA1`, `vLLM-0.22.0` |
| `--tp` | Tensor parallelism size | `1`, `4`, `8` |
| `--accelerator` | GPU type | `H200`, `MI300X` |
| `--runtime-args` | vLLM server runtime config (semicolon-separated) | See examples above |
| `--image-tag` | Container image used | `vllm/vllm-openai:v0.22.0` |
| `--guidellm-version` | GuideLLM version | `v0.6.0` |

### Optional Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset name | `SPEEDBENCH-1k`, `SPEEDBENCH-8k` |
| `--spec-decoding` | Spec decoding method or `baseline` | `eagle3-redhat`, `baseline` |
| `--prefix-caching` | Prefix caching status | `yes`, `no` |
| `--csv-file` | Output CSV path (appends if exists) | `throughput_1k.csv` |
| `--dp` | Data parallelism (instead of `--tp`) | `2` |
| `--cluster` | Cluster name for run column | `hera` |

### Output CSV

48 columns compatible with the performance dashboard. Key columns:

| Column | Description |
|--------|-------------|
| `run` | `{accelerator}-{model}-{tp}` |
| `intended concurrency` | Requested concurrency (streams) |
| `output_tok/sec` | Output tokens per second |
| `total_tok/sec` | Total tokens per second (input + output) |
| `ttft_median` | Time to first token - median (ms) |
| `tpot_median` | Time per output token - median (ms) |
| `itl_median` | Inter-token latency - median (ms) |
| `successful_requests` | Number of successful requests |
| `errored_requests` | Number of errored requests |
| `spec_decoding` | Speculative decoding method |
| `dataset` | Dataset name |

Full column list: `run, accelerator, model, version, prompt toks, output toks, TP, measured concurrency, intended concurrency, measured rps, output_tok/sec, total_tok/sec, prompt_token_count_mean, prompt_token_count_p99, output_token_count_mean, output_token_count_p99, ttft_median, ttft_p95, ttft_p1, ttft_p999, tpot_median, tpot_p95, tpot_p99, tpot_p999, tpot_p1, itl_median, itl_p95, itl_p999, itl_p1, request_latency_median, request_latency_min, request_latency_max, successful_requests, errored_requests, uuid, ttft_mean, ttft_p99, itl_mean, itl_p99, runtime_args, guidellm_start_time_ms, guidellm_end_time_ms, image_tag, guidellm_version, DP, dataset, spec_decoding, prefix_caching`

---

## 7. Concrete Examples

### Example A: gpt-oss-120b with RedHat Eagle3 Draft (RHAIIS)

Model: `openai/gpt-oss-120b` with `RedHatAI/gpt-oss-120b-speculator.eagle3`

```bash
# 1. Run GuideLLM (multi-concurrency mode, from pod)
oc exec -n kserve-e2e-perf guidellm-<USER> -- \
  guidellm benchmark \
    --target http://gpt-oss-120b-predictor.kserve-e2e-perf.svc.cluster.local:8080/v1 \
    --data /data/datasets/speed-bench-processed/throughput_1k/all.jsonl \
    --rate-type concurrent \
    --rate 1,5,25,50,100 \
    --output-path /data/results-rh-spec/speedbench-throughput-1k.json

# 2. Copy results
oc cp kserve-e2e-perf/guidellm-<USER>:/data/results-rh-spec/speedbench-throughput-1k.json \
  ./speedbench-throughput-1k.json --retries=5

# 3. Verify MD5
oc exec -n kserve-e2e-perf guidellm-<USER> -- md5sum /data/results-rh-spec/speedbench-throughput-1k.json
md5 -r ./speedbench-throughput-1k.json

# 4. Convert to CSV
source ~/test_foo/python3_virt/bin/activate
cd performance-dashboard/manual_runs/scripts

python import_manual_runs_json_v2.py \
  ./speedbench-throughput-1k.json \
  --model "openai/gpt-oss-120b" \
  --version "RHAIIS-3.5.EA1" \
  --tp 1 \
  --accelerator "H200" \
  --runtime-args 'uvicorn-log-level: debug; trust-remote-code: True; tensor-parallel-size: 1; gpu-memory-utilization: 0.85; disable-log-requests: True; enable-expert-parallel: True; no-enable-prefix-caching: True; max-cudagraph-capture-size: 2048; speculative-config:{"model":"RedHatAI/gpt-oss-120b-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3","draft_tensor_parallel_size":1}' \
  --image-tag "quay.io/aipcc/rhaiis/cuda-ubi9:3.5.0-ea.1-1779124632" \
  --guidellm-version "v0.6.0" \
  --dataset "SPEEDBENCH-1k" \
  --spec-decoding "eagle3-redhat" \
  --prefix-caching "yes" \
  --csv-file "throughput_1k_rh_draft.csv"
```

Repeat with `throughput_8k/all.jsonl` and `--dataset "SPEEDBENCH-8k"` for the 8k split.

### Example B: gemma-4-31B-it Baseline vs Eagle3-RH (Upstream vLLM)

Per-concurrency mode (5 separate JSONs: c1.json, c5.json, c25.json, c50.json, c100.json).

```bash
# 1. Run GuideLLM (per-concurrency, from pod)
for c in 1 5 25 50 100; do
  oc exec -n kserve-e2e-perf guidellm-<USER> -- \
    guidellm benchmark \
      --target http://gemma-4-31b-it-predictor.kserve-e2e-perf.svc.cluster.local:8080/v1 \
      --data /data/datasets/speed-bench-processed/throughput_1k/all.jsonl \
      --rate-type concurrent \
      --rate $c \
      --max-requests 100 \
      --output-path /data/results_gemma-baseline/c${c}.json
done

# 2. Copy all files
for c in c1 c5 c25 c50 c100; do
  oc cp kserve-e2e-perf/guidellm-<USER>:/data/results_gemma-baseline/${c}.json \
    ./results_gemma-baseline/${c}.json --retries=5
done

# 3. Verify MD5 for all
oc exec -n kserve-e2e-perf guidellm-<USER> -- \
  md5sum /data/results_gemma-baseline/c1.json /data/results_gemma-baseline/c5.json \
         /data/results_gemma-baseline/c25.json /data/results_gemma-baseline/c50.json \
         /data/results_gemma-baseline/c100.json

md5 -r ./results_gemma-baseline/*.json

# 4. Convert to CSV (appends per file)
source ~/test_foo/python3_virt/bin/activate
cd performance-dashboard/manual_runs/scripts

for c in c1 c5 c25 c50 c100; do
  python import_manual_runs_json_v2.py \
    ./results_gemma-baseline/${c}.json \
    --model "google/gemma-4-31B-it" \
    --version "vLLM-0.22.0" \
    --tp 1 \
    --accelerator "H200" \
    --runtime-args 'uvicorn-log-level: debug; trust-remote-code: True; tensor-parallel-size: 1; gpu-memory-utilization: 0.85; disable-log-requests: True; enable-expert-parallel: True; no-enable-prefix-caching: True' \
    --image-tag "vllm/vllm-openai:v0.22.0" \
    --guidellm-version "v0.6.0" \
    --dataset "SPEEDBENCH-1k" \
    --spec-decoding "baseline" \
    --prefix-caching "yes" \
    --csv-file "throughput_gemma-1k_baseline.csv"
done

# 5. Same for eagle3-rh (add speculative-config to runtime-args, change spec-decoding)
for c in c1 c5 c25 c50 c100; do
  python import_manual_runs_json_v2.py \
    ./results_gemma-eagle3-rh/${c}.json \
    --model "google/gemma-4-31B-it" \
    --version "vLLM-0.22.0" \
    --tp 1 \
    --accelerator "H200" \
    --runtime-args 'uvicorn-log-level: debug; trust-remote-code: True; tensor-parallel-size: 1; gpu-memory-utilization: 0.85; disable-log-requests: True; enable-expert-parallel: True; no-enable-prefix-caching: True; speculative-config:{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3","num_speculative_tokens":3,"method":"eagle3","draft_tensor_parallel_size":1}' \
    --image-tag "vllm/vllm-openai:v0.22.0" \
    --guidellm-version "v0.6.0" \
    --dataset "SPEEDBENCH-1k" \
    --spec-decoding "eagle3-redhat" \
    --prefix-caching "yes" \
    --csv-file "throughput_gemma-1k_rh_draft.csv"
done
```

---

## 8. Tool References

| Tool | Location | Purpose |
|------|----------|---------|
| `preprocess_speedbench.py` | [MML-coder/ai-tools](https://github.com/MML-coder/ai-tools/blob/main/preprocess_speedbench.py) | Preprocess SPEED-Bench HF dataset into JSONL |
| `analyze_speedbench_isl.py` | [MML-coder/ai-tools](https://github.com/MML-coder/ai-tools/blob/main/analyze_speedbench_isl.py) | Analyze ISL distribution per split |
| `run_speedbench_gpt_oss_120b.py` | [MML-coder/ai-tools](https://github.com/MML-coder/ai-tools/blob/main/run_speedbench_gpt_oss_120b.py) | Automated benchmark runner (local machine, remote pod) |
| `import_manual_runs_json_v2.py` | [openshift-psap/performance-dashboard](https://github.com/openshift-psap/performance-dashboard/blob/main/manual_runs/scripts/import_manual_runs_json_v2.py) | Convert GuideLLM JSON to dashboard-compatible CSV |
