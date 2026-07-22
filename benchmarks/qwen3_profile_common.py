"""Shared artifact contract for the bounded Qwen3 profiler capture."""

import hashlib
import json
from pathlib import Path


REQUIRED_METADATA_FIELDS = {
    "framework",
    "trace",
    "workload",
    "environment",
    "output_digest",
    "output_length",
}


def load_chrome_trace_events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    events = payload if isinstance(payload, list) else payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace must contain a traceEvents array")
    return events


def validate_chrome_trace(path: Path) -> dict:
    events = load_chrome_trace_events(path)
    return {"event_count": len(events), "size_bytes": path.stat().st_size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_metadata(path: Path, metadata: dict) -> None:
    missing = REQUIRED_METADATA_FIELDS - metadata.keys()
    if missing:
        raise ValueError(f"profile metadata missing fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
