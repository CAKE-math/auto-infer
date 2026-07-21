"""CLI to run a dataset eval on the NPU and print the score.

  docker exec -w <repo> -e PYTHONPATH=<repo> -e HF_ENDPOINT=https://hf-mirror.com \
    auto-infer-dev-20260624 python scripts/run_eval.py mmlu \
    --model /data1/models/Qwen3-0.6B --limit 200

Tasks: mmlu (mc, 5-shot), gsm8k (gen, 8-shot), cmmlu (mc, 5-shot),
humaneval (gen, 0-shot). --n-shot / --limit override the defaults.
"""
import argparse
import importlib
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TASKS = {                                  # name -> (module, class, default n_shot)
    "mmlu": ("auto_infer.eval.tasks.mmlu", "MMLU", 5),
    "gsm8k": ("auto_infer.eval.tasks.gsm8k", "GSM8K", 8),
    "ceval": ("auto_infer.eval.tasks.ceval", "CEval", 0),
    "humaneval": ("auto_infer.eval.tasks.humaneval", "HumanEval", 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-shot", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-gen", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from auto_infer.eval.runner import evaluate
    mod, cls, default_shot = TASKS[args.task]
    task = getattr(importlib.import_module(mod), cls)()
    n_shot = default_shot if args.n_shot is None else args.n_shot
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    res = evaluate(task, args.model, tok, n_shot=n_shot, limit=args.limit,
                   max_gen_tokens=args.max_gen, device_index=args.device)
    print(f"\n=== {res['task']}  n_shot={n_shot}  "
          f"acc={res['acc']:.4f}  ({res['correct']}/{res['n']}) ===")


if __name__ == "__main__":
    main()
