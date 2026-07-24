"""StepEnvelope-compatible Harness results and provenance."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys


_EXIT_CODES = {"ok": 0, "partial": 2, "failed": 1, "skipped": 0}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_git_dir(marker: Path) -> Path:
    if marker.is_dir():
        return marker
    line = marker.read_text().strip()
    prefix = "gitdir: "
    if not line.startswith(prefix):
        raise ValueError(f"invalid git marker: {marker}")
    target = Path(line[len(prefix):])
    return target if target.is_absolute() else (marker.parent / target).resolve()


def _read_ref(git_dir: Path, ref: str) -> str | None:
    candidates = [git_dir / ref]
    common = git_dir / "commondir"
    common_dir = None
    if common.exists():
        value = Path(common.read_text().strip())
        common_dir = value if value.is_absolute() else (git_dir / value).resolve()
        candidates.append(common_dir / ref)
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text().strip()
    for root in (git_dir, common_dir):
        if root is None:
            continue
        packed = root / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    sha, name = line.split(" ", 1)
                    if name == ref:
                        return sha
    return None


def git_commit(start: Path) -> str | None:
    for directory in (start.resolve(), *start.resolve().parents):
        marker = directory / ".git"
        if not marker.exists():
            continue
        try:
            git_dir = _resolve_git_dir(marker)
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref: "):
                return _read_ref(git_dir, head[5:])
            return head or None
        except (OSError, ValueError):
            return None
    return None


def build_provenance(config_path: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[2]
    provenance = {
        "framework_commit": git_commit(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if config_path is not None:
        provenance["config_sha256"] = sha256_file(config_path)
    return provenance


@dataclass(frozen=True)
class HarnessResult:
    status: str
    step_id: str
    result: dict = field(default_factory=dict)
    artifacts: dict[str, str | None] = field(default_factory=dict)
    error_summary: str | None = None
    provenance: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        exit_code(self.status)
        if not self.step_id:
            raise ValueError("step_id must not be empty")

    def to_dict(self) -> dict:
        return asdict(self)


def write_result(result: HarnessResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    )


def exit_code(status: str) -> int:
    try:
        return _EXIT_CODES[status]
    except KeyError:
        raise ValueError(f"unknown harness status: {status}") from None
