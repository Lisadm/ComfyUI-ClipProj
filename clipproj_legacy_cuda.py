"""CUDA implementation of the reference Windows FP16 projector reduction."""
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


REFERENCE_TORCH = "cf30153c4c131c8164ee7798e5022d810682e2cb"


def supports(residual, x):
    return (
        sys.platform == "win32" and torch.version.git_version == REFERENCE_TORCH
        and x.device.type == "cuda" and torch.cuda.get_device_capability(x.device) == (12, 0)
        and x.dtype == torch.float16 and len(residual) == 3
        and isinstance(residual[1], torch.nn.GELU) and residual[1].approximate == "none"
        and residual[0].weight.shape == (32768, 2560)
        and residual[2].weight.shape == (5120, 32768)
        and residual[0].bias is not None and residual[2].bias is not None
    )


@triton.jit
def _linear_kernel(X, W, B, LUT, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   TILE: tl.constexpr, ACTIVATE: tl.constexpr):
    row = tl.program_id(1) * TILE + tl.arange(0, TILE)
    col = tl.program_id(0) * TILE + tl.arange(0, TILE)
    lane = tl.arange(0, 64)
    acc = tl.full((TILE, TILE, 64), 0, tl.float32)
    for k in range(K // 64):
        z = k * 64 + lane
        a = tl.load(X + row[:, None] * K + z[None, :], row[:, None] < M, other=0).to(tl.float32)
        b = tl.load(W + col[:, None] * K + z[None, :], col[:, None] < N, other=0).to(tl.float32)
        acc = tl.fma(a[:, None, :], b[None, :, :], acc)

    # Pair the eight FP32 registers in the reference order, then sum their lanes.
    left, right = tl.split(tl.trans(tl.reshape(acc, (TILE, TILE, 2, 32)), (0, 1, 3, 2)))
    left, right = tl.split(tl.trans(tl.reshape(left + right, (TILE, TILE, 2, 16)), (0, 1, 3, 2)))
    left, right = tl.split(tl.trans(tl.reshape(left + right, (TILE, TILE, 2, 8)), (0, 1, 3, 2)))
    acc = left + right
    dot = tl.gather(acc, tl.full((TILE, TILE, 1), 0, tl.int32), 2).reshape(TILE, TILE)
    for i in tl.static_range(1, 8):
        dot += tl.gather(acc, tl.full((TILE, TILE, 1), i, tl.int32), 2).reshape(TILE, TILE)
    bias = tl.load(B + col, col < N, other=0).to(tl.float32)
    result = (dot + bias[None, :]).to(tl.float16)
    if ACTIVATE:
        bits = result.to(tl.uint16, bitcast=True).to(tl.int32)
        result = tl.load(LUT + bits)
    tl.store(Y + row[:, None] * N + col[None, :], result,
             (row[:, None] < M) & (col[None, :] < N))


def _linear(x, weight, bias, lookup=None):
    m, k = x.shape
    n = weight.shape[0]
    result = torch.empty((m, n), device=x.device, dtype=x.dtype)
    tile = 8
    _linear_kernel[(triton.cdiv(n, tile), triton.cdiv(m, tile))](
        x, weight, bias, bias if lookup is None else lookup, result,
        m, n, k, tile, lookup is not None, num_warps=8)
    return result


def forward(residual, x):
    # A 128 KiB table keeps CPU GELU rounding for every possible FP16 input.
    half_bits = torch.arange(65536, device="cpu", dtype=torch.int32).to(torch.int16).view(torch.float16)
    lookup = F.gelu(half_bits).to(x.device)
    first, _, last = residual
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    activated = _linear(rows, first.weight, first.bias, lookup)
    output = _linear(activated, last.weight, last.bias)
    return output.reshape(*x.shape[:-1], last.out_features)
