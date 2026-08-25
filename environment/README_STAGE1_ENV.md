# H9 Stage 1 server environment

Activate the verified environment with:

```bash
source /root/autodl-tmp/envs/h9-stage1/bin/activate
export HF_HOME=/root/autodl-tmp/huggingface
```

The target and semantic models are cached by immutable revisions listed in
`STAGE1_RUNTIME_MANIFEST.json`; formal scripts should pass those revisions or resolved
snapshot paths explicitly and use offline/local-only loading after the initial download.

To recreate third-party packages, create a Python 3.12 environment, install
`torch==2.5.1` from the CUDA 12.1 PyTorch wheel index, then install the exact versions in
`requirements-stage1.lock` and finally install this repository editable at the committed
Stage 1 revision. Do not recreate the environment from unconstrained extras alone.

The server has one 24 GB RTX 4090 D. Stage 1 is intentionally configured for the 0.5B
model and does not require the previous three-GPU 32B topology.
