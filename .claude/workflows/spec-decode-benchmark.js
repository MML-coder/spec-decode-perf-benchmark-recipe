export const meta = {
  name: 'spec-decode-benchmark',
  description: 'End-to-end speculative decoding benchmark: deploy model, run GuideLLM, collect results, generate CSV',
  whenToUse: 'When the user wants to run a speculative decoding benchmark end-to-end on a Kubernetes cluster',
  phases: [
    { title: 'Validate', detail: 'Read config.yaml, verify cluster access' },
    { title: 'Setup', detail: 'Create PVC + GuideLLM pod if needed' },
    { title: 'Deploy', detail: 'Deploy vLLM model via KServe' },
    { title: 'Benchmark', detail: 'Run GuideLLM benchmarks for each dataset split' },
    { title: 'Collect', detail: 'Copy results from pod, verify MD5 checksums' },
    { title: 'Convert', detail: 'Convert JSON results to CSV' },
    { title: 'Summary', detail: 'Print performance summary with geomean comparison' },
  ],
}

const CONFIG_SCHEMA = {
  type: 'object',
  properties: {
    cluster: {
      type: 'object',
      properties: {
        kubeconfig: { type: 'string' },
        namespace: { type: 'string' },
        accelerator: { type: 'string' },
        storage_class: { type: 'string' },
        user: { type: 'string' },
      },
    },
    model: {
      type: 'object',
      properties: {
        hf_id: { type: 'string' },
        image: { type: 'string' },
        version: { type: 'string' },
        tp: { type: 'number' },
        gpu_count: { type: 'number' },
        storage_pvc: { type: 'string' },
        vllm_args: { type: 'object' },
      },
    },
    spec_decoding: {
      type: 'object',
      properties: {
        enabled: { type: 'boolean' },
        draft_model: { type: 'string' },
        num_speculative_tokens: { type: 'number' },
        method: { type: 'string' },
        draft_tensor_parallel_size: { type: 'number' },
        label: { type: 'string' },
      },
    },
    benchmark: {
      type: 'object',
      properties: {
        dataset: { type: 'string' },
        splits: { type: 'array', items: { type: 'string' } },
        concurrency: { type: 'array', items: { type: 'number' } },
        mode: { type: 'string' },
        guidellm_version: { type: 'string' },
        prefix_caching: { type: 'string' },
      },
    },
    output: {
      type: 'object',
      properties: {
        results_dir: { type: 'string' },
      },
    },
  },
}

const VALIDATE_SCHEMA = {
  type: 'object',
  properties: {
    valid: { type: 'boolean' },
    config: CONFIG_SCHEMA,
    cluster_user: { type: 'string' },
    deployment_name: { type: 'string' },
    endpoint_url: { type: 'string' },
    pod_name: { type: 'string' },
    run_date: { type: 'string' },
    errors: { type: 'array', items: { type: 'string' } },
  },
}

const STATUS_SCHEMA = {
  type: 'object',
  properties: {
    success: { type: 'boolean' },
    message: { type: 'string' },
    details: { type: 'string' },
  },
}

const COLLECT_SCHEMA = {
  type: 'object',
  properties: {
    success: { type: 'boolean' },
    files: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, local_path: { type: 'string' }, md5_match: { type: 'boolean' } } } },
    message: { type: 'string' },
  },
}

const SUMMARY_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    csv_files: { type: 'array', items: { type: 'string' } },
  },
}

// Phase 1: Validate
phase('Validate')
log('Reading config.yaml and validating cluster access...')

const validation = await agent(`
You are validating the benchmark configuration. Do these steps exactly:

1. Read the file config.yaml in the current working directory. Parse it as YAML.

2. Check that ALL required fields are non-empty:
   - cluster.kubeconfig
   - cluster.namespace
   - cluster.accelerator
   - cluster.storage_class
   - cluster.user
   - model.hf_id
   - model.image
   - model.version
   - model.storage_pvc
   - benchmark.splits (must have at least one entry)
   - benchmark.concurrency (must have at least one entry)
   If any are empty, set valid=false and list missing fields in errors.

3. Verify cluster access by running:
   KUBECONFIG=<cluster.kubeconfig> oc whoami
   If this fails, set valid=false and add the error.

4. Check the namespace exists:
   KUBECONFIG=<cluster.kubeconfig> oc get namespace <cluster.namespace>
   If this fails, set valid=false and add the error.

5. Check the storage class exists:
   KUBECONFIG=<cluster.kubeconfig> oc get sc <cluster.storage_class>
   If this fails, list available storage classes and set valid=false.

6. If model.node_selector is empty, auto-detect the GPU node:
   KUBECONFIG=<cluster.kubeconfig> oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{.items[0].metadata.name}'
   If that returns nothing, try:
   KUBECONFIG=<cluster.kubeconfig> oc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.capacity.nvidia\.com/gpu}{"\n"}{end}' | grep -v '<none>' | grep -v '^$'
   Pick the first node that has nvidia.com/gpu > 0.
   If no GPU node found, set valid=false with error "No GPU nodes found on cluster".
   Set the discovered node name as node_selector in the returned config.

7. Derive the deployment name from model.hf_id:
   - Split on "/"
   - Take the second part
   - Lowercase
   - Replace "." with "-"
   Example: "openai/gpt-oss-120b" -> "gpt-oss-120b"

8. Derive the endpoint URL:
   http://<deployment_name>-predictor.<namespace>.svc.cluster.local:8080/v1

9. Derive the pod name: guidellm-<cluster.user>

10. Get today's date by running: date +%Y%m%d
    Return this as the "run_date" field (e.g., "20260624").

Return the full parsed config, cluster_user from oc whoami, deployment_name, endpoint_url, pod_name, run_date, and any errors.
`, { label: 'validate-config', schema: VALIDATE_SCHEMA })

if (!validation || !validation.valid) {
  const errs = validation ? validation.errors.join(', ') : 'validation agent failed'
  log('VALIDATION FAILED: ' + errs)
  return { success: false, error: errs }
}

const cfg = validation.config
const deploymentName = validation.deployment_name
const endpointUrl = validation.endpoint_url
const podName = validation.pod_name
const kc = `KUBECONFIG=${cfg.cluster.kubeconfig}`
const ns = cfg.cluster.namespace

const runDate = validation.run_date || '00000000'

log(`Validated: model=${cfg.model.hf_id}, deployment=${deploymentName}, pod=${podName}`)
log(`Endpoint: ${endpointUrl}`)
log(`Cluster user: ${validation.cluster_user}, date: ${runDate}`)

// Phase 2: Setup Infrastructure
phase('Setup')
log('Creating PVC and GuideLLM pod if they do not already exist...')

const setup = await agent(`
You are setting up benchmark infrastructure on a Kubernetes cluster.
Do NOT assume anything exists — check first, then create only if missing.

Use this prefix for ALL oc commands: ${kc}
Namespace: ${ns}
Pod name: ${podName}
Storage class: ${cfg.cluster.storage_class}
User: ${cfg.cluster.user}

Steps:

1. Check if PVC "client-storage" exists in namespace ${ns}:
   ${kc} oc get pvc client-storage -n ${ns} --no-headers 2>/dev/null
   - If it exists, log "PVC already exists, skipping creation"
   - If it does NOT exist, create it:
     ${kc} oc apply -f - <<EOF
     apiVersion: v1
     kind: PersistentVolumeClaim
     metadata:
       name: client-storage
       namespace: ${ns}
     spec:
       accessModes: [ReadWriteOnce]
       resources:
         requests:
           storage: 100Gi
       storageClassName: ${cfg.cluster.storage_class}
     EOF

2. Check if pod "${podName}" exists in namespace ${ns}:
   ${kc} oc get pod ${podName} -n ${ns} --no-headers 2>/dev/null
   - If it exists and is Running, log "Pod already running, skipping creation"
   - If it does NOT exist, create it:
     ${kc} oc apply -f - <<EOF
     apiVersion: v1
     kind: Pod
     metadata:
       name: ${podName}
       namespace: ${ns}
       labels:
         app: guidellm
         user: ${cfg.cluster.user}
     spec:
       restartPolicy: Always
       securityContext:
         fsGroup: 0
         runAsUser: 0
       containers:
       - name: guidellm
         image: python:3.11-slim
         imagePullPolicy: IfNotPresent
         command: ["/bin/bash", "-c"]
         args:
         - |
           set -e
           apt-get update && apt-get install -y tmux vim git curl wget procps
           pip install --upgrade pip
           pip install 'guidellm[recommended] @ git+https://github.com/vllm-project/guidellm.git@${cfg.benchmark.guidellm_version}'
           echo "Installation complete!"
           tail -f /dev/null
         workingDir: /data
         resources:
           requests: { cpu: "1", memory: "4Gi" }
           limits: { cpu: "4", memory: "8Gi" }
         securityContext:
           runAsUser: 0
         volumeMounts:
         - { name: client-storage, mountPath: /data }
       volumes:
       - name: client-storage
         persistentVolumeClaim:
           claimName: client-storage
     EOF

3. Wait for pod to be Running and for GuideLLM installation to complete:
   - Poll: ${kc} oc get pod ${podName} -n ${ns} -o jsonpath='{.status.phase}'
   - Then check logs for "Installation complete!": ${kc} oc logs ${podName} -n ${ns} | tail -5
   - Retry up to 30 times with 10s sleep (5 minutes total)
   - If it never becomes ready, report failure.

4. Verify guidellm is installed:
   ${kc} oc exec -n ${ns} ${podName} -- guidellm --version

5. Create results directory on pod:
   ${kc} oc exec -n ${ns} ${podName} -- mkdir -p /data/results

Report success=true if pod is running with guidellm installed, or success=false with details.
`, { label: 'setup-infra', schema: STATUS_SCHEMA })

if (!setup || !setup.success) {
  log('SETUP FAILED: ' + (setup ? setup.message : 'agent failed'))
  return { success: false, error: setup ? setup.message : 'setup agent failed' }
}
log('Infrastructure ready: ' + setup.message)

// Phase 3: Deploy Model
phase('Deploy')
log('Deploying vLLM model via KServe...')

// Build vllm args as YAML list items for the template
const vllmArgItems = Object.entries(cfg.model.vllm_args || {}).map(([k, v]) => {
  if (v === 'true') return `    - --${k}`
  return `    - --${k}=${v}`
})

// Add speculative config if enabled
let specConfigJson = ''
if (cfg.spec_decoding && cfg.spec_decoding.enabled) {
  const specObj = {
    model: cfg.spec_decoding.draft_model,
    num_speculative_tokens: cfg.spec_decoding.num_speculative_tokens,
    method: cfg.spec_decoding.method,
    draft_tensor_parallel_size: cfg.spec_decoding.draft_tensor_parallel_size,
  }
  specConfigJson = JSON.stringify(specObj)
  vllmArgItems.push(`    - '--speculative-config=${specConfigJson}'`)
}

const vllmArgsYaml = vllmArgItems.join('\n')

// Build runtime-args string for CSV (semicolon-separated)
const runtimeArgsParts = Object.entries(cfg.model.vllm_args || {}).map(([k, v]) => `${k}: ${v}`)
if (specConfigJson) {
  runtimeArgsParts.push(`speculative-config:${specConfigJson}`)
}
const runtimeArgsStr = runtimeArgsParts.join('; ')

// Generate a deployment UUID
const deployUuid = [deploymentName, 'bench'].join('-').substring(0, 36)

const deploy = await agent(`
You are deploying a vLLM model on Kubernetes via KServe using the template at manifests/kserve-template.yaml.
Do NOT assume the model is already deployed — check first.
Do NOT generate YAML from scratch — use the template and replace placeholders.

Use this prefix for ALL oc commands: ${kc}
Namespace: ${ns}
Deployment name: ${deploymentName}

Steps:

1. Check if InferenceService "${deploymentName}" already exists:
   ${kc} oc get inferenceservice ${deploymentName} -n ${ns} --no-headers 2>/dev/null
   - If it exists and has Ready=True, log "Model already deployed and ready" and return success.
   - If it exists but is not ready, wait for it (step 3).
   - If it does NOT exist, proceed to step 2.

2. Read the template file manifests/kserve-template.yaml and replace these placeholders:
   __NAME__                  -> ${deploymentName}
   __NAMESPACE__             -> ${ns}
   __MODEL_ID__              -> ${cfg.model.hf_id}
   __IMAGE__                 -> ${cfg.model.image}
   __VLLM_ARGS__             -> (see below)
   __GPU_COUNT__             -> ${cfg.model.gpu_count}
   __NODE_SELECTOR__         -> ${cfg.model.node_selector}
   __STORAGE_PVC__           -> ${cfg.model.storage_pvc}
   __SERVICE_ACCOUNT__       -> ${cfg.model.service_account || 'sa'}
   __IMAGE_PULL_SECRET__     -> ${cfg.model.image_pull_secret || 'default-dockercfg'}
   __HF_TOKEN_SECRET__       -> ${cfg.model.hf_token_secret || 'storage-config'}
   __SHARED_MEMORY_SIZE__    -> ${cfg.model.shared_memory_size || '8Gi'}
   __MODEL_LOADING_TIMEOUT__ -> ${cfg.model.model_loading_timeout || 300000}
   __DEPLOYMENT_UUID__       -> ${deployUuid}

   For __VLLM_ARGS__, replace the placeholder line with these YAML list items:
${vllmArgsYaml}

   If node_selector is empty (""), REMOVE the entire nodeSelector block (both the key and value lines).
   If image_pull_secret is empty (""), REMOVE the imagePullSecrets block.

   Write the rendered manifest to a temp file, then apply:
   ${kc} oc apply -f /tmp/kserve-${deploymentName}.yaml

3. Wait for InferenceService to become Ready:
   - Poll: ${kc} oc get inferenceservice ${deploymentName} -n ${ns} -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}'
   - Should return "True"
   - Retry up to 360 times with 10s sleep (1 hour total)
   - If not ready after 1 hour, report failure with the current status and pod events.

4. Verify the endpoint is reachable from the GuideLLM pod:
   ${kc} oc exec -n ${ns} ${podName} -- curl -s -o /dev/null -w '%{http_code}' ${endpointUrl.replace('/v1', '/health')}
   - Should return 200
   - Retry up to 30 times with 10s sleep if not 200

Report success=true when the model is deployed and health check passes, or success=false with details about what failed.
`, { label: 'deploy-model', schema: STATUS_SCHEMA })

if (!deploy || !deploy.success) {
  log('DEPLOY FAILED: ' + (deploy ? deploy.message : 'agent failed'))
  return { success: false, error: deploy ? deploy.message : 'deploy agent failed' }
}
log('Model deployed: ' + deploy.message)

// Phase 4: Run Benchmarks
phase('Benchmark')
const splits = cfg.benchmark.splits || ['throughput_1k']
const concurrencyLevels = cfg.benchmark.concurrency || [1, 5, 25, 50, 100]
const concurrencyList = concurrencyLevels.join(',')
const benchmarkMode = cfg.benchmark.mode || 'per-concurrency'
const maxReqMultiplier = cfg.benchmark.max_requests_multiplier || 3
const maxReqMinimum = cfg.benchmark.max_requests_minimum || 100
const backendTimeout = cfg.benchmark.backend_timeout || 100000

log(`Running benchmarks: ${splits.length} split(s), concurrency=[${concurrencyList}], mode=${benchmarkMode}`)

const benchResults = []
for (const split of splits) {
  log(`Starting benchmark for split: ${split}`)

  let dataPath
  if (cfg.benchmark.dataset === 'synthetic') {
    dataPath = `prompt_tokens=1000,output_tokens=1000`
  } else {
    dataPath = `/data/datasets/speed-bench-processed/${split}/all.jsonl`
  }

  const resultsSubdir = `/data/results_${deploymentName}_${cfg.spec_decoding.label}_${runDate}`
  const dataFlag = cfg.benchmark.dataset === 'synthetic'
    ? `--data "${dataPath}"`
    : `--data ${dataPath}`

  // Build per-concurrency max_requests table for the agent prompt
  const perConcurrencyTable = concurrencyLevels.map(c => {
    const maxReq = Math.max(maxReqMinimum, maxReqMultiplier * c)
    return `  concurrency=${c} -> max_requests=${maxReq}`
  }).join('\n')

  const bench = await agent(`
You are running a GuideLLM benchmark on a Kubernetes cluster.
This is a LONG-RUNNING operation — benchmarks can take 30-120 minutes per split.

Use this prefix for ALL oc commands: ${kc}
Namespace: ${ns}
Pod: ${podName}
Target endpoint: ${endpointUrl}
Dataset: ${cfg.benchmark.dataset}
Data path/config: ${dataPath}
Results directory on pod: ${resultsSubdir}
Concurrency levels: ${concurrencyList}
Mode: ${benchmarkMode}
Backend timeout: ${backendTimeout}

Steps:

1. If dataset is "speedbench", verify the dataset file exists on the pod:
   ${kc} oc exec -n ${ns} ${podName} -- ls -la ${dataPath}
   If it does NOT exist, report failure immediately — the user needs to upload datasets first.

2. Create results directory on pod:
   ${kc} oc exec -n ${ns} ${podName} -- mkdir -p ${resultsSubdir}

3. Run the GuideLLM benchmark using mode "${benchmarkMode}":

   === MODE: multi-concurrency ===
   (Use this if mode is "multi-concurrency")
   Produces 1 JSON file with all concurrency levels inside:

   ${kc} oc exec -n ${ns} ${podName} -- \\
     guidellm benchmark run \\
       --target ${endpointUrl} \\
       --backend-kwargs '{"timeout":${backendTimeout}}' \\
       ${dataFlag} \\
       --rate-type concurrent \\
       --rate ${concurrencyList} \\
       --output-path ${resultsSubdir}/speedbench-${split.replace('throughput_', '')}.json

   === MODE: per-concurrency ===
   (Use this if mode is "per-concurrency")
   Produces 1 JSON per concurrency level. max_requests = max(${maxReqMinimum}, ${maxReqMultiplier} * concurrency):

${perConcurrencyTable}

   Run each concurrency level sequentially:
   for c in ${concurrencyList.replace(/,/g, ' ')}; do
     maxreq=$(( ${maxReqMultiplier} * c ))
     if [ $maxreq -lt ${maxReqMinimum} ]; then maxreq=${maxReqMinimum}; fi

     ${kc} oc exec -n ${ns} ${podName} -- \\
       guidellm benchmark run \\
         --target ${endpointUrl} \\
         --backend-kwargs '{"timeout":${backendTimeout}}' \\
         ${dataFlag} \\
         --rate-type concurrent \\
         --rate $c \\
         --max-requests $maxreq \\
         --output-path ${resultsSubdir}/${split.replace('throughput_', '')}_c$c.json
   done

   IMPORTANT: Each concurrency run can take 10-60 minutes. Run them sequentially and wait for each to complete.
   Do NOT set a short timeout — use at least 7200 seconds (2 hours) per command.

4. After completion, verify output files exist:
   ${kc} oc exec -n ${ns} ${podName} -- ls -la ${resultsSubdir}/

Report success=true with the results directory path, or success=false with error details.
Split name: ${split}
`, { label: 'benchmark-' + split, schema: STATUS_SCHEMA })

  benchResults.push({ split, result: bench })

  if (!bench || !bench.success) {
    log('BENCHMARK FAILED for ' + split + ': ' + (bench ? bench.message : 'agent failed'))
  } else {
    log('Benchmark complete for ' + split + ': ' + bench.message)
  }
}

const failedBench = benchResults.filter(b => !b.result || !b.result.success)
if (failedBench.length === benchResults.length) {
  log('ALL BENCHMARKS FAILED')
  return { success: false, error: 'all benchmarks failed' }
}

// Phase 5: Collect Results
phase('Collect')
log('Copying results from pod and verifying MD5 checksums...')

// Build per-concurrency file list for collect agent
const concurrencyFileList = concurrencyLevels.map(c => `<split_short>_c${c}.json`).join(', ')
const remoteResultsDir = `/data/results_${deploymentName}_${cfg.spec_decoding.label}_${runDate}`

const collect = await agent(`
You are collecting benchmark results from a Kubernetes pod to the local machine.
MD5 verification is MANDATORY — do not skip it.

Use this prefix for ALL oc commands: ${kc}
Namespace: ${ns}
Pod: ${podName}
Local results directory: ${cfg.output.results_dir}
Splits that were benchmarked: ${splits.join(', ')}
Benchmark mode: ${benchmarkMode}
Spec decoding label: ${cfg.spec_decoding.label}
Concurrency levels: ${concurrencyList}

The remote results directory is:
  /data/results_${deploymentName}_${cfg.spec_decoding.label}_${runDate}/

All splits share this single remote directory. Files are prefixed with the split name (e.g., 1k_c1.json, 8k_c5.json for per-concurrency, or speedbench-throughput_1k.json for multi-concurrency).

Steps:

1. Create local results directory:
   mkdir -p ${cfg.output.results_dir}/results_${deploymentName}_${cfg.spec_decoding.label}_${runDate}

2. For each split (${splits.join(', ')}):
   Remote dir: /data/results_${deploymentName}_${cfg.spec_decoding.label}_${runDate}/

   The split short name is derived by removing "throughput_" prefix (e.g., throughput_1k -> 1k, throughput_8k -> 8k).

   If mode is "multi-concurrency":
     - Single file: speedbench-<split_short>.json
     - Copy it, verify MD5

   If mode is "per-concurrency":
     - Multiple files per split: <split_short>_c1.json, <split_short>_c5.json, <split_short>_c25.json, etc.
     - For concurrency levels: ${concurrencyList}
     - Copy each file, verify MD5 for each

   For EACH file:
   a. Get remote MD5:
      ${kc} oc exec -n ${ns} ${podName} -- md5sum <remote_path>

   b. Copy:
      ${kc} oc cp ${ns}/${podName}:<remote_path> <local_path> --retries=5

   c. Get local MD5:
      md5 -r <local_path>  (macOS) OR md5sum <local_path> (Linux)

   d. If checksums do NOT match:
      - Delete local file
      - Retry copy up to 3 times
      - If still mismatched after 3 retries, report failure for that file

3. List all collected files with their sizes.

Report the list of files, their local paths, and whether MD5 matched for each.
`, { label: 'collect-results', schema: COLLECT_SCHEMA })

if (!collect || !collect.success) {
  log('COLLECT FAILED: ' + (collect ? collect.message : 'agent failed'))
  return { success: false, error: collect ? collect.message : 'collect agent failed' }
}
log('Results collected: ' + collect.message)

// Phase 6: Convert to CSV
phase('Convert')
log('Converting JSON results to CSV...')

const convert = await agent(`
You are converting GuideLLM benchmark JSON files to CSV format using the import_manual_runs_json_v2.py script.

The script is at: /Users/memehta/workspace/performance-dashboard/manual_runs/scripts/import_manual_runs_json_v2.py
Activate virtualenv first: source ~/test_foo/python3_virt/bin/activate
Run from: /Users/memehta/workspace/performance-dashboard/manual_runs/scripts

Configuration:
  Model: ${cfg.model.hf_id}
  Version: ${cfg.model.version}
  TP: ${cfg.model.tp}
  Accelerator: ${cfg.cluster.accelerator}
  Image tag: ${cfg.model.image}
  GuideLLM version: ${cfg.benchmark.guidellm_version}
  Spec decoding label: ${cfg.spec_decoding.label}
  Prefix caching: ${cfg.benchmark.prefix_caching}
  Runtime args: ${runtimeArgsStr}

Local results directory: ${cfg.output.results_dir}
Splits: ${splits.join(', ')}
Benchmark mode: ${benchmarkMode}
Concurrency levels: ${concurrencyList}

Steps:

For each split (${splits.join(', ')}):
  1. Determine the dataset label from the split name:
     - throughput_1k -> SPEEDBENCH-1k
     - throughput_2k -> SPEEDBENCH-2k
     - throughput_8k -> SPEEDBENCH-8k
     - throughput_16k -> SPEEDBENCH-16k
     - throughput_32k -> SPEEDBENCH-32k
     - If synthetic, use "synthetic"

  2. Determine the CSV filename (one CSV per split):
     ${cfg.output.results_dir}/<split>_${cfg.spec_decoding.label}/<split>_${cfg.spec_decoding.label}.csv

  3. Run the import script:

     If mode is "multi-concurrency":
       Single JSON file contains all concurrency levels. Run once:
       source ~/test_foo/python3_virt/bin/activate
       cd /Users/memehta/workspace/performance-dashboard/manual_runs/scripts
       python import_manual_runs_json_v2.py \\
         ${cfg.output.results_dir}/<split>_${cfg.spec_decoding.label}/speedbench-<split>.json \\
         --model "${cfg.model.hf_id}" \\
         --version "${cfg.model.version}" \\
         --tp ${cfg.model.tp} \\
         --accelerator "${cfg.cluster.accelerator}" \\
         --runtime-args '${runtimeArgsStr}' \\
         --image-tag "${cfg.model.image}" \\
         --guidellm-version "${cfg.benchmark.guidellm_version}" \\
         --dataset "<DATASET_LABEL>" \\
         --spec-decoding "${cfg.spec_decoding.label}" \\
         --prefix-caching "${cfg.benchmark.prefix_caching}" \\
         --csv-file "<CSV_PATH>"

     If mode is "per-concurrency":
       Multiple JSON files (c1.json, c5.json, etc). Run once per file — the script APPENDS to the same CSV:
       source ~/test_foo/python3_virt/bin/activate
       cd /Users/memehta/workspace/performance-dashboard/manual_runs/scripts
       for c in ${concurrencyList.replace(/,/g, ' ')}; do
         python import_manual_runs_json_v2.py \\
           ${cfg.output.results_dir}/<split>_${cfg.spec_decoding.label}/c$c.json \\
           --model "${cfg.model.hf_id}" \\
           --version "${cfg.model.version}" \\
           --tp ${cfg.model.tp} \\
           --accelerator "${cfg.cluster.accelerator}" \\
           --runtime-args '${runtimeArgsStr}' \\
           --image-tag "${cfg.model.image}" \\
           --guidellm-version "${cfg.benchmark.guidellm_version}" \\
           --dataset "<DATASET_LABEL>" \\
           --spec-decoding "${cfg.spec_decoding.label}" \\
           --prefix-caching "${cfg.benchmark.prefix_caching}" \\
           --csv-file "<CSV_PATH>"
       done

  4. Verify the CSV was created and has the expected number of rows.
     Expected rows = number of concurrency levels (${concurrencyLevels.length}).

Report the list of CSV files created and their row counts.
`, { label: 'convert-csv', schema: SUMMARY_SCHEMA })

if (!convert) {
  log('CONVERT FAILED: agent returned null')
  return { success: false, error: 'convert agent failed' }
}
log('CSVs generated: ' + (convert.csv_files || []).join(', '))

// Phase 7: Summary
phase('Summary')
log('Generating performance summary...')

const summary = await agent(`
You are generating a performance summary from benchmark CSV files.

Results directory: ${cfg.output.results_dir}
Model: ${cfg.model.hf_id}
Spec decoding: ${cfg.spec_decoding.label} (enabled=${cfg.spec_decoding.enabled})
Accelerator: ${cfg.cluster.accelerator}
Splits: ${splits.join(', ')}

Steps:

1. Read all CSV files in ${cfg.output.results_dir} that end with .csv

2. For each CSV, extract these key columns per concurrency level:
   - intended concurrency
   - output_tok/sec (throughput)
   - tpot_median (time per output token, ms)
   - itl_median (inter-token latency, ms)
   - ttft_median (time to first token, ms)
   - successful_requests
   - errored_requests

3. Print a formatted table showing these metrics per concurrency level, per split.

4. Print the absolute values — do NOT compute comparison percentages (we only have one run config here).
   Comparisons require a baseline run with a different config.

5. Print a summary line with:
   - Peak throughput (max output_tok/sec across all concurrency levels)
   - Best TPOT (min tpot_median, usually at lowest concurrency)
   - Total successful/errored requests

Format the output as a clean markdown table that can be pasted into Jira or Slack.

Return the formatted summary string and list of CSV files processed.
`, { label: 'summary', schema: SUMMARY_SCHEMA })

log('=== BENCHMARK COMPLETE ===')
if (summary) {
  log(summary.summary || 'Summary generated')
}

return {
  success: true,
  config: cfg,
  deployment_name: deploymentName,
  endpoint_url: endpointUrl,
  csv_files: convert ? convert.csv_files : [],
  summary: summary ? summary.summary : '',
}
