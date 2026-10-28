"""Experimental NTransformer training kernels and activation pipeline.

PyTorch supplies GEMM, attention, autograd scheduling, AdamW and DDP.
The inference CLI and its read-only weight cache are independent.
"""
