import json

from scripts.verify_model_package import (
    build_parser,
    compare_outputs,
    output_digest,
)


def test_output_digest_is_stable_and_order_sensitive():
    outputs = [[11, 12], [21]]

    assert output_digest(outputs) == output_digest([[11, 12], [21]])
    assert output_digest(outputs) != output_digest([[21], [11, 12]])


def test_compare_outputs_reports_exact_mismatch():
    comparison = compare_outputs([[1, 2], [3]], [[1, 2], [4]])

    assert comparison["exact_match"] is False
    assert comparison["mismatched_requests"] == [1]
    assert comparison["reference_digest"] != comparison["candidate_digest"]


def test_cli_requires_reference_candidate_and_package():
    args = build_parser().parse_args([
        "--reference-model", "/models/reference",
        "--candidate-model", "/models/candidate",
        "--package", "/packages/candidate",
        "--prompt-ids", "[[1, 2, 3], [4]]",
    ])

    assert args.reference_model == "/models/reference"
    assert args.candidate_model == "/models/candidate"
    assert json.loads(args.prompt_ids) == [[1, 2, 3], [4]]
