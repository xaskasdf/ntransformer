# H100 training experiment results

Status: **failed**.

Experimental hybrid backend: native CUDA RMSNorm/SwiGLU and peer activation staging; PyTorch supplies GEMM, attention, AdamW and DDP.

Last supervisor message: CalledProcessError: Command '['/home/ubuntu/brandon-mini/.venv/bin/python', '/home/ubuntu/ntransformer-h100/source/tests/test_training_ops.py', '--brandon-root', '/lambda/nfs/btiny/brandon-mini/sources/stage1-v1', '--checkpoint', '/lambda/nfs/btiny/brandon-mini/stages/stage1-v1/pretrain/last.pt', '--output', '/home/ubuntu/ntransformer-h100/training/experiment-20261013/validation.json']' returned non-zero exit status 1.

See `validation.json`, `comparison-*.json`, per-case `summary.json` and `logs/` for diagnostics.
