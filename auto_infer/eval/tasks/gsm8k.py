"""GSM8K grade-school math word problems, chain-of-thought + numeric match.

Generative (gen mode): greedy-decode a CoT, then pull the final number. 8-shot
by default, shots taken from the train split (their gold answers already carry
step-by-step reasoning ending in ``#### <n>``; we strip the ``<<calc>>``
annotations). Matches the standard GSM8K CoT setup — gold is the number after
``####``; a prediction counts if its extracted number equals gold.

Dataset: ``openai/gsm8k`` config ``main`` (train=7473, test=1319).
"""
import re

_CALC = re.compile(r"<<[^>]*>>")                 # "<<16-3-4=9>>" calculator spans
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")        # 1,234  -5  3.14
_HASH_ANS = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")   # the "#### N" answer marker


def _gold(answer):
    """The reference number: text after ``####`` in a GSM8K answer."""
    return answer.split("####")[-1].strip().replace(",", "")


def _to_float(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


class GSM8K:
    name = "gsm8k"
    mode = "gen"

    def _load(self, split):
        from datasets import load_dataset
        return load_dataset("openai/gsm8k", "main", split=split)

    def fewshot_prompt(self, n_shot):
        if not n_shot:
            return ""
        blocks = []
        for d in list(self._load("train"))[:n_shot]:
            cot = _CALC.sub("", d["answer"]).strip()
            blocks.append(f"Question: {d['question'].strip()}\nAnswer: {cot}\n\n")
        return "".join(blocks)

    def test_docs(self, limit):
        docs = list(self._load("test"))
        return docs[:limit] if limit else docs

    def doc_to_prompt(self, doc):
        return f"Question: {doc['question'].strip()}\nAnswer:"

    def doc_to_target(self, doc):
        return _gold(doc["answer"])

    def extract(self, text):
        """The predicted number: the FIRST ``#### N`` marker (the current
        question's answer — the model keeps going and hallucinates further
        Q&A blocks with their own ``####`` since our engine has no string-stop,
        so the *last* marker would be a made-up question's answer). Fall back to
        the last bare number when the model never emits ``####``."""
        m = _HASH_ANS.search(text)
        if m:
            return m.group(1).replace(",", "")
        nums = _NUM.findall(text)
        return nums[-1].replace(",", "") if nums else ""

    def is_correct(self, pred, target):
        p, t = _to_float(pred), _to_float(target)
        if p is not None and t is not None:
            return abs(p - t) < 1e-4
        return pred.strip() == target.strip()
