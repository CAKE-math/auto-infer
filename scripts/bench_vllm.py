import time
from vllm import LLM, SamplingParams
path="/data0/models/Qwen2.5-0.5B-Instruct"
templates=["The capital of France is","Explain photosynthesis:","2+2 equals",
           "Once upon a time","The largest planet is","Write a haiku about",
           "The speed of light is","In the year 2050,"]
N=256
prompts=[templates[i%len(templates)] for i in range(N)]
llm=LLM(model=path, max_model_len=2048, gpu_memory_utilization=0.5)
sp=SamplingParams(max_tokens=32, temperature=0)
t0=time.perf_counter()
outs=llm.generate(prompts, sp)
dt=time.perf_counter()-t0
ntok=sum(len(o.outputs[0].token_ids) for o in outs)
print(f"VLLM_RESULT N={N} wall={dt:.2f}s tok_s={ntok/dt:.0f} req_s={N/dt:.1f}")
