"""MMLU (Massive Multitask Language Understanding), 57 subjects, 4-choice.

Scored the way the Open LLM Leaderboard does it: the prompt lists the four
options as ``A./B./C./D.`` and ends with ``Answer:``; we score the log-prob of
each single-letter continuation (`` A``/`` B``/`` C``/`` D``) and argmax. 5-shot,
where the shots are drawn per-subject from the ``dev`` split (so the few-shot
block is folded into ``doc_to_prompt`` rather than a single global preamble).

Dataset: ``cais/mmlu`` config ``all`` — one stream carrying every subject, with
a ``subject`` field on each row (test=14042, dev=5×57, validation=1531).
"""

_LETTERS = ["A", "B", "C", "D"]


def _fmt_q(doc, with_answer):
    """One question block: stem, the four labelled options, then ``Answer:``.
    When `with_answer`, append the gold letter (used for few-shot examples)."""
    lines = [doc["question"].strip()]
    for letter, choice in zip(_LETTERS, doc["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    text = "\n".join(lines)
    if with_answer:
        text += f" {_LETTERS[doc['answer']]}\n\n"
    return text


class MMLU:
    name = "mmlu"
    mode = "mc"

    def __init__(self):
        self._shots_by_subject = {}   # subject -> preformatted few-shot block

    def _load(self, split):
        from datasets import load_dataset
        return load_dataset("cais/mmlu", "all", split=split)

    def fewshot_prompt(self, n_shot):
        """Build one few-shot block per subject from the dev split; returns ""
        because the shots are per-doc (chosen by the test row's subject)."""
        self._n_shot = n_shot
        if n_shot:
            by_subject = {}
            for d in self._load("dev"):
                by_subject.setdefault(d["subject"], []).append(d)
            for subject, docs in by_subject.items():
                header = ("The following are multiple choice questions (with "
                          f"answers) about {subject.replace('_', ' ')}.\n\n")
                shots = "".join(_fmt_q(d, with_answer=True) for d in docs[:n_shot])
                self._shots_by_subject[subject] = header + shots
        return ""

    def test_docs(self, limit):
        docs = list(self._load("test"))
        if not limit:
            return docs
        # round-robin by subject so a partial run stays stratified across all 57
        # subjects (the raw stream is grouped by subject — docs[:limit] would be
        # one subject). Group, then interleave.
        by_subject = {}
        for d in docs:
            by_subject.setdefault(d["subject"], []).append(d)
        interleaved = []
        groups = list(by_subject.values())
        for j in range(max(len(g) for g in groups)):
            for g in groups:
                if j < len(g):
                    interleaved.append(g[j])
        return interleaved[:limit]

    def doc_to_prompt(self, doc):
        return self._shots_by_subject.get(doc["subject"], "") + _fmt_q(doc, with_answer=False)

    def doc_to_choices(self, doc):
        return [f" {ltr}" for ltr in _LETTERS]     # leading space: "Answer: A"

    def doc_to_label(self, doc):
        return doc["answer"]
