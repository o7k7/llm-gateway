# Kubernetes Deployment

Manifests for deploying the LLM gateway stack on Kubernetes. Tested on:

- **Single-node k3s on RunPod** (NVIDIA A100 80GB) — production-class
  validation with all four components running. See top-level
  `docs/DEPLOYMENT.md` for the full RunPod walkthrough.

## File ordering

Files are numbered by apply order. Apply them all at once with
`kubectl apply -f deploy/k8s/`, or individually if you want to watch
each layer come up.

| File | Purpose |
|------|---------|
|namespace.yaml | Creates `llm-gateway` namespace |
|nvidia-device-plugin.yaml | GPU time-slicing (NVIDIA only; skip on k3d) |
|configmap.yaml | Non-sensitive config |
|secret.yaml.example | Template; create real secrets via kubectl |
|redis.yaml | Redis Stack StatefulSet + PVC |
|vllm-small.yaml | vLLM-small backend (Qwen2.5-7B-AWQ) |
|vllm-large.yaml | vLLM-large backend (Qwen2.5-14B-AWQ) |
|baseline.yaml | Naive HF baseline server (for benchmarks) |
|gateway.yaml | Gateway Deployment + Service + PDB |

## Prerequisites

### For NVIDIA GPU clusters (production-class)

- NVIDIA Container Toolkit installed on nodes
- `nvidia-container-runtime` configured as default in containerd
- The cluster must have at least one node with `nvidia.com/gpu.present`
  label and exposed GPU resources

### For all clusters

- `kubectl` configured against your target cluster
- Network access to GHCR (or your image registry) for pulling
  gateway/baseline images
- Network access to HuggingFace Hub for vLLM init containers to
  download model weights

## Quick start

```bash
# 1. Apply manifests (NVIDIA GPU cluster)
kubectl apply -f deploy/k8s/
