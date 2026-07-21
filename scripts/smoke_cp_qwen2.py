"""CP=2 Qwen2 prefill (torchrun). Prints CP last-token; compare to single-card run."""
import os, torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.distributed.parallel_state import init_distributed, tp_rank, tp_size
from auto_infer.models.qwen2 import Qwen2Model

PROMPT = "The capital of France is, in fact, the city of"

def main():
    single = os.environ.get("SINGLE") == "1"
    if not single:
        init_distributed()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    dev = torch.device(f"npu:{local}")
    tok = AutoTokenizer.from_pretrained("/data0/models/Qwen2.5-0.5B-Instruct")
    m = Qwen2Model.from_pretrained("/data0/models/Qwen2.5-0.5B-Instruct", dev, torch.bfloat16)
    ids = tok(PROMPT).input_ids
    if single:
        T = len(ids)
        out = m.forward_dense(torch.tensor(ids, device=dev), torch.arange(T, device=dev))[-1]
        print("SINGLE last:", int(out.argmax()))
        return
    cs = tp_size(); r = tp_rank()
    T = (len(ids) // cs) * cs; ids = ids[:T]; lt = T // cs
    li = torch.tensor(ids[r*lt:(r+1)*lt], device=dev)
    lp = torch.arange(r*lt, r*lt+lt, device=dev)
    cp = m.forward_cp(li, lp, T, r, cs)
    if r == cs - 1:
        print("CP last:", int(cp[-1].argmax()))

if __name__ == "__main__":
    main()
