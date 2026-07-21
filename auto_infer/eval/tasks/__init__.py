"""Concrete eval tasks. Each module exposes a task class implementing the Task
protocol in ``auto_infer.eval.runner``. Datasets are pulled with the HuggingFace
``datasets`` library; set ``HF_ENDPOINT=https://hf-mirror.com`` for the domestic
mirror (the default the run scripts export).

  * mmlu       — MMLU, 4-choice, log-prob (mc), 5-shot per subject
  * gsm8k      — grade-school math, CoT generate + numeric match (gen), 8-shot
  * ceval      — C-Eval Chinese multi-discipline, 4-choice, log-prob (mc), 0-shot
  * humaneval  — Python code synthesis, generate + unit-test exec (gen), 0-shot
"""
