"""Custom Triton-Ascend kernel: fused SwiGLU activation silu(x)*y (custom-kernel
seam — the lever to beat library-only paths). dtype-preserving."""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    tl.store(o_ptr + offs, (x * tl.sigmoid(x)) * y, mask=mask)   # cast-on-store to o dtype


def silu_mul(x, y):
    x = x.contiguous(); y = y.contiguous()
    o = torch.empty_like(x)
    n = x.numel()
    _silu_mul_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    return o
