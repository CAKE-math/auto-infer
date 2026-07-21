import re
from pathlib import Path

import pytest

from auto_infer.entrypoints.cli import build_parser


ROOT = Path(__file__).parents[1]


def test_cli_help_and_execution_modes(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    assert "serve" in capsys.readouterr().out


def test_local_markdown_links_exist():
    for document in (ROOT / "README.md", ROOT / "docs/SPEC-ALIGNMENT.md"):
        text = document.read_text()
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" not in target and not target.startswith("#"):
                path = (document.parent / target.split("#", 1)[0]).resolve()
                assert path.exists(), f"{document.relative_to(ROOT)} -> {target}"


def test_decode_validation_report_records_all_three_frameworks_and_gates():
    report = ROOT / "docs/VLLM-FIRST-VALIDATION-2026-07-20.md"
    text = report.read_text()
    for term in ("auto-infer", "omni-npu", "vllm-ascend", "B16", "TPOT",
                 "output digest", "npu-smi", "path counters",
                 "14,464", "20 measured", "coefficient of variation",
                 "prefill graph"):
        assert term in text
