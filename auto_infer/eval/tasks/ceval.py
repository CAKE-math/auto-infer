"""C-Eval — Chinese multi-discipline multiple-choice (4-choice), log-prob (mc).

Scored like MMLU: options listed ``A./B./C./D.``, prompt ends with ``答案：``,
score the log-prob of each single-letter continuation and argmax.

Dataset: ``ceval/ceval-exam`` — pulled via the HF auto-converted parquet
(``revision="refs/convert/parquet"``, the only form the current ``datasets``
loads since script datasets were dropped). That parquet exposes ``validation``
(1606 rows, WITH gold answers) and ``test`` (answers hidden — unusable for
self-grading), and drops the per-subject split, so this runs 0-shot over the
validation set. Fields: ``question``, ``A``/``B``/``C``/``D``, ``answer`` (letter).
"""

_LETTERS = ["A", "B", "C", "D"]
_REVISION = "refs/convert/parquet"


class CEval:
    name = "ceval"
    mode = "mc"

    def fewshot_prompt(self, n_shot):
        return ""      # C-Eval parquet has no dev split -> 0-shot only

    def test_docs(self, limit):
        from datasets import load_dataset
        docs = list(load_dataset("ceval/ceval-exam", "default",
                                 revision=_REVISION, split="validation"))
        return docs[:limit] if limit else docs

    def doc_to_prompt(self, doc):
        lines = [doc["question"].strip()]
        for ltr in _LETTERS:
            lines.append(f"{ltr}. {doc[ltr]}")
        lines.append("答案：")
        return "\n".join(lines)

    def doc_to_choices(self, doc):
        return [f" {ltr}" for ltr in _LETTERS]

    def doc_to_label(self, doc):
        return _LETTERS.index(doc["answer"].strip())
