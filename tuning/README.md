# tuning/

Distillation pipeline for the SLM specialists described in
[`experiments/SLM_PLAN.md`](../experiments/SLM_PLAN.md). Two scripts plus a
clusters config:

- `clusters.yaml` — per-cluster definition (BQ source + match patterns + synth seed prompts).
- `dataset_exporter.py` — pulls (prompt, teacher_response) pairs from BigQuery, optionally augments with `gemini -p` synthesis, writes 80/20 SFTTrainer-format JSONL.
- `qlora_gemma4.py` — QLoRA fine-tune of Gemma 4 (default 2B-IT) using `peft` + `trl` + `bitsandbytes`. Runs locally or as a Vertex Custom Training job without code changes.

Install the extra deps:

```bash
pip install -e ".[tuning]"
```

## 1. Export the cluster dataset

```bash
python tuning/dataset_exporter.py \
  --cluster cloudrun \
  --out tuning/datasets/cloudrun/v1/

# Augment with 200 synthetic Q&A pairs via gemini CLI
python tuning/dataset_exporter.py \
  --cluster cloudrun \
  --out tuning/datasets/cloudrun/v1/ \
  --synthesize 200
```

Outputs `train.jsonl`, `eval.jsonl`, `manifest.json`. Each row is
`{"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}`.

Requires `bq` CLI authenticated to the project that owns
`wortz-project-352116.gemini_dreams.dream_raw_logs`, and `gemini` CLI for
`--synthesize`.

## 2. Train QLoRA locally

```bash
python tuning/qlora_gemma4.py \
  --train tuning/datasets/cloudrun/v1/train.jsonl \
  --eval  tuning/datasets/cloudrun/v1/eval.jsonl \
  --base  google/gemma-4-2b-it \
  --out   tuning/adapters/cloudrun/v1/ \
  --epochs 3
```

Writes `adapter_model.safetensors` + tokenizer files to `--out`. Expect
~30-60 min on a single L4 for ~250 examples.

## 3. Train on Vertex AI Custom Training

Package the script and submit:

```bash
ADAPTER_BUCKET=gs://router-adapters
DATA_BUCKET=gs://router-tuning
REGION=us-central1
PROJECT=wortz-project-352116

# Stage data
gsutil -m cp tuning/datasets/cloudrun/v1/*.jsonl ${DATA_BUCKET}/cloudrun/v1/

gcloud ai custom-jobs create \
  --project=${PROJECT} \
  --region=${REGION} \
  --display-name=qlora-gemma4-cloudrun-v1 \
  --worker-pool-spec=\
machine-type=g2-standard-12,\
accelerator-type=NVIDIA_L4,accelerator-count=1,\
replica-count=1,\
container-image-uri=us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-4.py311,\
local-package-path=tuning,\
script=qlora_gemma4.py \
  --args="--train=${DATA_BUCKET}/cloudrun/v1/train.jsonl,\
--eval=${DATA_BUCKET}/cloudrun/v1/eval.jsonl,\
--base=google/gemma-4-2b-it,\
--out=${ADAPTER_BUCKET}/cloudrun/v1/,\
--epochs=3"
```

The script accepts `gs://` paths for `--train`, `--eval`, and `--out` when
the runtime mounts them via Cloud Storage FUSE (Vertex managed datasets do
this automatically).

## 4. Eval and deploy with vLLM multi-LoRA

Eval gate (passes if `adapter_total >= 0.85 * teacher_total` on the rubric
in `experiments/judge.py`):

```bash
# (sketch — eval harness lives in experiments/, will accept --adapter)
python experiments/judge.py \
  --adapter tuning/adapters/cloudrun/v1/ \
  --eval-set tuning/datasets/cloudrun/v1/eval.jsonl \
  --out experiments/results_judge/cloudrun_v1
```

Serve with vLLM:

```bash
vllm serve google/gemma-4-2b-it \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 16 \
  --lora-modules cloudrun=tuning/adapters/cloudrun/v1/

# Hot-load additional adapters at runtime:
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'content-type: application/json' \
  -d '{"lora_name":"gcloud","lora_path":"gs://router-adapters/gcloud/v1"}'
```

Add the backend to `config/router.yaml`:

```yaml
- name: gemma4:cloudrun
  kind: vllm
  endpoint: http://localhost:8000/v1
  model: google/gemma-4-2b-it
  adapter: cloudrun     # served via --lora-modules above
  capabilities: [stream, local, "specialist:cloudrun"]
  cost_in_per_1m: 0.0
  cost_out_per_1m: 0.0
  expected_latency_ms_per_1k_out: 80.0
  max_context: 8192
```

Then add anchor exemplars to `config/anchors.yaml` under `gemma4:cloudrun:`
and run `router-eval rebuild-anchors`.
