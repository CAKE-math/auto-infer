"""NPU platform helpers (Ascend / torch_npu)."""
from importlib import import_module

import torch
def npu_device(index: int = 0) -> "torch.device":
    try:
        import_module("torch_npu")
    except ImportError as e:
        raise RuntimeError(
            "auto-infer requires an Ascend NPU (torch_npu), which was not found. "
            "Run inside the CANN container (image ascend/vllm-ascend). This is an "
            "NPU-only framework — there is no CPU fallback."
        ) from e
    torch.npu.set_device(index)
    return torch.device(f"npu:{index}")


def default_dtype(name: str = "bfloat16") -> "torch.dtype":
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]
