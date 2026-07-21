"""Offline evaluation harness (dataset benchmarks, to compare against official
HF/leaderboard scores). Lightweight, self-written, ModelScope datasets.

Two scoring modes (see runner.py):
  * "mc"  — multiple choice (MMLU / CEval / CMMLU): score each choice by the
    log-prob its continuation gets, argmax vs the labelled answer. Matches the
    log-likelihood accuracy the Open LLM Leaderboard reports.
  * "gen" — generative (GSM8K / HumanEval): greedy-decode via LLM.generate, then
    extract + check the answer (numeric / code-exec).
"""
