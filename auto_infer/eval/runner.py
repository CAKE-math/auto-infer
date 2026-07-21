"""Generic offline eval loop over our engine. Tasks (eval/tasks/*.py) provide
the dataset + prompt formatting + scoring; this runner drives the model and
aggregates accuracy. Depends only on the model, not on `datasets` (that's in the
task's `load`).

A Task is any object with:
  name: str
  mode: "mc" | "gen"
  fewshot_prompt(n_shot) -> str            # GLOBAL few-shot preamble prepended to
                                           # every doc; return "" when the shots are
                                           # per-doc (e.g. MMLU's per-subject shots,
                                           # which the task folds into doc_to_prompt).
                                           # Also the task's chance to load/cache
                                           # whatever few-shot state it needs.
  test_docs(limit) -> list[dict]           # the graded rows (test split)
  doc_to_prompt(doc) -> str                # this question's prompt (no answer);
                                           # may itself embed per-doc few-shot
  # mc mode:
  doc_to_choices(doc) -> list[str]         # continuation string per choice
  doc_to_label(doc) -> int                 # index of the correct choice
  # gen mode:
  doc_to_target(doc) -> str                # gold answer (for is_correct)
  extract(text) -> str                     # pull the answer out of generated text
  is_correct(pred: str, target: str) -> bool
"""
import torch

from auto_infer.config import (CacheConfig, EngineConfig, ExecutionConfig,
                               ModelConfig, SchedulerConfig)
from auto_infer.entrypoints.llm import LLM


def _token_batches(lengths, budget):
    """Yield index chunks whose summed `lengths` stay under `budget` tokens
    (always >=1 item per chunk, so an over-budget item runs alone)."""
    batch, tot = [], 0
    for i, L in enumerate(lengths):
        if batch and tot + L > budget:
            yield batch
            batch, tot = [], 0
        batch.append(i)
        tot += L
    if batch:
        yield batch


def _score_mc_singletok(model, prompts, choice_ids, labels, budget):
    """Fast path when every choice is ONE token (MMLU/CEval letters): batch the
    prompts, one forward per batch, read the log-prob of each candidate token at
    the last prompt position, argmax. `choice_ids[d]` = [tok_c0, tok_c1, ...]."""
    correct = 0
    for batch in _token_batches([len(p) for p in prompts], budget):
        hidden, bounds = model.forward_dense_batch([prompts[b] for b in batch])
        last = torch.tensor([bounds[r][1] - 1 for r in range(len(batch))], device=hidden.device)
        logp = torch.log_softmax(model.logits(hidden[last]).float(), dim=-1)   # (B, vocab)
        cand = torch.tensor([choice_ids[b] for b in batch], device=hidden.device)  # (B, C)
        preds = logp.gather(1, cand).argmax(dim=1).tolist()
        correct += sum(int(preds[r] == labels[b]) for r, b in enumerate(batch))
    return correct


def _score_mc_general(model, prompts, choices_tok, labels, budget):
    """General path (multi-token continuations): score each prompt+choice
    sequence's teacher-forced continuation log-prob, batched, then argmax over a
    doc's choices. `choices_tok[d]` = [cont_ids_c0, cont_ids_c1, ...]."""
    seqs, meta = [], []                       # meta[j] = (doc, choice, plen, clen)
    for di, (p, chs) in enumerate(zip(prompts, choices_tok)):
        for ci, cont in enumerate(chs):
            seqs.append(p + cont)
            meta.append((di, ci, len(p), len(cont)))
    scores = {}                               # (doc, choice) -> summed log-prob
    for batch in _token_batches([len(s) for s in seqs], budget):
        hidden, bounds = model.forward_dense_batch([seqs[j] for j in batch])
        rows, tgts, segs = [], [], []
        for r, j in enumerate(batch):
            s, _ = bounds[r]
            _, _, plen, clen = meta[j]
            for t in range(clen):             # position s+plen+t-1 predicts cont token t
                rows.append(s + plen + t - 1)
                tgts.append(seqs[j][plen + t])
            segs.append((j, clen))
        logp = torch.log_softmax(
            model.logits(hidden[torch.tensor(rows, device=hidden.device)]).float(), dim=-1)
        lp = logp.gather(1, torch.tensor(tgts, device=hidden.device)[:, None]).squeeze(1).tolist()
        k = 0
        for j, clen in segs:
            di, ci, _, _ = meta[j]
            scores[(di, ci)] = sum(lp[k:k + clen])
            k += clen
    return sum(int(max(range(len(choices_tok[di])), key=lambda ci: scores[(di, ci)]) == labels[di])
               for di in range(len(prompts)))


def evaluate(task, model_path, tok, *, n_shot=0, limit=None, max_gen_tokens=256,
             device_index=0, mc_batch_tokens=16384):
    """Run `task` on the model at `model_path`; return {"acc", "n", "correct"}."""
    model = None
    llm = None
    preamble = task.fewshot_prompt(n_shot)      # global preamble ("" if per-doc)
    docs = task.test_docs(limit)

    if task.mode == "gen":
        cfg = EngineConfig(model=ModelConfig(model_path=model_path),
                           cache=CacheConfig(block_size=16, num_blocks=2048),
                           scheduler=SchedulerConfig(max_num_batched_tokens=8192),
                           execution=ExecutionConfig(mode="paged", device_index=device_index))
        llm = LLM(cfg)
        prompts = [tok(preamble + task.doc_to_prompt(d)).input_ids for d in docs]
        outs = llm.generate(prompts, max_tokens=max_gen_tokens, eos_token_id=tok.eos_token_id)
        correct = sum(task.is_correct(task.extract(tok.decode(o, skip_special_tokens=True)),
                                      task.doc_to_target(d)) for o, d in zip(outs, docs))
        llm.close()
    else:  # mc — log-prob over choices, teacher-forced (leaderboard-aligned)
        import json
        import os

        from importlib import import_module
        import_module("torch_npu")  # registers torch.npu and Ascend kernels

        from auto_infer.models.registry import get_model_class
        from auto_infer.platform import npu_device
        arch = json.load(open(os.path.join(model_path, "config.json")))["architectures"][0]
        model = get_model_class(arch).from_pretrained(
            model_path, device=npu_device(device_index), dtype=torch.bfloat16)

        # tokenize once; decide fast (single-token continuations) vs general path
        prompts = [tok(preamble + task.doc_to_prompt(d)).input_ids for d in docs]
        choices_tok = [[tok(c, add_special_tokens=False).input_ids
                        for c in task.doc_to_choices(d)] for d in docs]
        labels = [task.doc_to_label(d) for d in docs]
        if all(len(c) == 1 for chs in choices_tok for c in chs):
            choice_ids = [[c[0] for c in chs] for chs in choices_tok]
            correct = _score_mc_singletok(model, prompts, choice_ids, labels, mc_batch_tokens)
        else:
            correct = _score_mc_general(model, prompts, choices_tok, labels, mc_batch_tokens)

    n = len(docs)
    return {"task": task.name, "acc": correct / n if n else 0.0, "correct": correct, "n": n}
