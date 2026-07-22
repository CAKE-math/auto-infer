"""Validate the machine-readable outputs of the three-framework benchmark."""
import json
import sys
from pathlib import Path

from benchmarks.common import (
    validate_comparable_results, validate_comparison_result)


EXPECTED_FRAMEWORKS = {"auto-infer", "omni-npu", "vllm-ascend"}


def load_comparable_results(
        paths, *, legacy_schema: str | None = None) -> list[dict]:
    results = [json.loads(Path(path).read_text()) for path in paths]
    for result in results:
        validate_comparison_result(
            result, legacy_schema=legacy_schema)
    frameworks = {result["framework"] for result in results}
    if len(results) != 3 or frameworks != EXPECTED_FRAMEWORKS:
        raise ValueError(
            f"expected one result for each of {sorted(EXPECTED_FRAMEWORKS)}; "
            f"got {sorted(frameworks)}")
    canonical_manifest = results[0]["manifest"]
    if any(result["manifest"] != canonical_manifest for result in results[1:]):
        raise ValueError("benchmark manifests differ across framework results")
    validate_comparable_results(results)
    return results


def main() -> None:
    results = load_comparable_results(sys.argv[1:])
    capacity = results[0]["kv_capacity"]["usable_tokens"]
    print(f"comparison valid: 3 frameworks, {capacity} usable KV tokens")


if __name__ == "__main__":
    main()
