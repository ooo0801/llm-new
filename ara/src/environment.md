# Environment

- Server repository: `/root/autodl-tmp/llm`
- Python: `/root/autodl-tmp/venvs/llm-integrity/bin/python` (Python 3.12 environment)
- Model: `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`
- Runtime: three NVIDIA GeForce RTX 4080 SUPER GPUs, 32,760 MiB each
- Offline cache: `/root/autodl-tmp/huggingface`
- H1 dependencies preflighted offline: sentence-transformers, bitsandbytes, SciPy, and the exact BGE snapshot
- F1 result commits: `9fe8ac7`, `fcd6bf0`, `03cf71e`
