"""HumanEval — Python function synthesis, pass@1 by unit-test execution (gen).

The prompt is the function signature + docstring; the model completes the body.
We truncate the completion at the first line that starts a new top-level
construct, splice ``prompt + completion + test + check(entry_point)`` into one
program, and run it in a subprocess with a timeout — pass iff it exits 0.

Generated code is executed. It runs in a fresh subprocess with a wall-clock
timeout so a hang or crash can't take down the eval; this is the standard
HumanEval harness and is meant for a trusted research box, not untrusted input.

Dataset: ``openai/openai_humaneval`` (split ``test``, 164 problems).
"""
import subprocess
import sys

# a completion ends when the model starts a new top-level construct
_STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n@", "\n```", "\nassert "]
_TIMEOUT_S = 15


class HumanEval:
    name = "humaneval"
    mode = "gen"

    def fewshot_prompt(self, n_shot):
        return ""      # HumanEval is 0-shot (the stub+docstring is the prompt)

    def test_docs(self, limit):
        from datasets import load_dataset
        docs = list(load_dataset("openai/openai_humaneval", split="test"))
        return docs[:limit] if limit else docs

    def doc_to_prompt(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        # passed straight through to is_correct (protocol is duck-typed) — the
        # unit-test harness needs the test body + entry point, not just a string.
        return {"prompt": doc["prompt"], "test": doc["test"], "entry_point": doc["entry_point"]}

    def extract(self, text):
        """The function body: cut the generation at the first stop marker so a
        chatty model's trailing prose/next-function doesn't break exec."""
        cut = len(text)
        for s in _STOPS:
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    def is_correct(self, pred, target):
        program = (target["prompt"] + pred + "\n"
                   + target["test"] + f"\ncheck({target['entry_point']})\n")
        try:
            r = subprocess.run([sys.executable, "-c", program],
                               capture_output=True, timeout=_TIMEOUT_S)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
