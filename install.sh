#!/usr/bin/env bash
# One-click install for auto-infer (Ascend-NPU LLM inference framework).
#
# Run this INSIDE the Ascend CANN container (image: ascend/vllm-ascend), which
# already provides torch + torch_npu. auto-infer only adds a small pure-Python
# layer (numpy/safetensors/transformers) + itself as an editable package.
#
#   bash install.sh            # install + verify (skips the slow model smoke)
#   bash install.sh --smoke /data0/models/Qwen2.5-0.5B-Instruct   # + NPU smoke
#
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
say() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

# 1) Python >= 3.11
say "Python: $($PY -c 'import sys;print(sys.version.split()[0])')"
$PY -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,11) else 1)' \
    || { warn "Python >= 3.11 required"; exit 1; }

# 2) editable install (pulls numpy/safetensors/transformers; NOT torch — vendor-provided)
say "pip install -e . (editable)"
$PY -m pip install -e . --quiet

# 3) verify imports
say "verifying auto_infer imports..."
$PY - <<'PYEOF'
import auto_infer                                    # noqa
from auto_infer.entrypoints.llm import LLM           # noqa
from auto_infer.layers.attention.backend import GqaFIABackend, GraphGqaBackend, MlaFIABackend, DenseBackend  # noqa
print("  auto_infer OK")
try:
    import torch, torch_npu                          # noqa
    print(f"  torch {torch.__version__} + torch_npu {torch_npu.__version__} — NPU env OK")
    print(f"  NPU device_count = {torch.npu.device_count()}")
except Exception as e:
    print(f"  [warn] torch_npu not importable ({e}). auto_infer installed, but you are")
    print(f"         NOT in an Ascend NPU env — run inside the ascend/vllm-ascend container.")
PYEOF

# 4) optional NPU smoke
if [ "${1:-}" = "--smoke" ]; then
    MODEL="${2:-/data0/models/Qwen2.5-0.5B-Instruct}"
    say "NPU smoke: scripts/smoke_qwen2.py $MODEL"
    $PY scripts/smoke_qwen2.py "$MODEL"
fi

say "done. Try:"
echo "  $PY scripts/smoke_qwen2.py /data0/models/Qwen2.5-0.5B-Instruct        # bare model + HF parity"
echo "  $PY scripts/smoke_engine_npu.py /data0/models/Qwen2.5-0.5B-Instruct   # full engine"
echo "  $PY scripts/run_deepseek_chat.py /data1/models/DeepSeek-V2-Lite-Chat  # DeepSeek MLA+MoE"
echo "  $PY -m pytest -q                                                      # host tests"
echo
echo "  NOTE: DeepSeek (16B MoE) needs PYTORCH_NPU_ALLOC_CONF=expandable_segments:True to"
echo "        avoid NPU allocator fragmentation OOMs. The deepseek scripts set this themselves;"
echo "        set it in the env for your own entrypoints."
