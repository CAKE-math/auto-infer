"""Distributed sharded weight loading (spec §15.1).

For 671B-class models the full state dict cannot be materialized on any single
host. This loader lets each rank read ONLY the tensors it owns (its TP/EP shard),
pulling individual tensors from the safetensors shard files via ``safe_open``
(header-only key scan + per-tensor reads) — the whole shard file is never loaded
into RAM, and non-local experts are never read at all.

Model-agnostic mechanism (index parsing + selective streaming read); the
weight-name → shard predicate is supplied by the model (model-relevant, spec §7).
"""
import glob
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from safetensors import safe_open


def build_weight_index(path: str) -> dict[str, str]:
    """Map ``tensor_name -> shard_file_abspath``.

    Uses ``model.safetensors.index.json`` when present (the multi-shard layout
    DeepSeek-V3 671B ships). Otherwise scans each ``*.safetensors`` file's header
    for its keys — ``safe_open(...).keys()`` reads only the metadata header, not
    the tensor bytes, so this stays cheap even for large single shards.
    """
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
        return {name: os.path.join(path, fn) for name, fn in weight_map.items()}
    mapping: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                mapping[name] = f
    if not mapping:
        raise FileNotFoundError(f"no *.safetensors found under {path}")
    return mapping


def start_prefetch(path: str, num_threads: int = 8, chunk_size: int = 8 * 1024 * 1024):
    """Warm the OS page cache for ``path``'s ``*.safetensors`` shards in the
    background, so the sequential/threaded reads ``load_sharded`` does next
    mostly hit page cache instead of cold disk.

    Mirrors vLLM's checkpoint-prefetch mechanism (``weight_utils.py``
    ``maybe_prefetch_checkpoint`` / ``_prefetch_all_checkpoints`` /
    ``_prefetch_checkpoint``, ~line 800): a background daemon thread owns a
    small thread pool that reads each shard file start-to-end in fixed-size
    chunks (vLLM uses 16MB blocks; here 8MB), discarding the bytes — the OS
    page cache does the actual caching. This call returns immediately
    (launches the daemon thread and returns) so it can run concurrently with
    config parsing / module construction while the checkpoint warms.

    Best-effort only: any failure — path doesn't exist, permission denied,
    file removed mid-read, whatever — is swallowed silently. This function
    must never raise and never block the caller.
    """
    def _prefetch_file(fp: str) -> None:
        try:
            with open(fp, "rb") as f:
                while f.read(chunk_size):
                    pass
        except Exception:
            pass  # best-effort warm; a failed prefetch just falls back to cold reads

    def _run() -> None:
        try:
            files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
            with ThreadPoolExecutor(max_workers=num_threads) as ex:
                list(ex.map(_prefetch_file, files))
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def load_sharded(path, wanted, device="cpu", dtype=None, max_workers: int = 8) -> dict:
    """Load only tensors for which ``wanted(name)`` is True.

    Reads are grouped by shard file so each file is opened once and only the
    needed tensors are pulled (``get_tensor`` reads one tensor without loading
    the rest of the file). Never builds a full state dict → bounded host memory.
    Returns ``{name: tensor}`` already on ``device`` (and cast to ``dtype`` for
    floating tensors, leaving int8/quant tensors untouched).

    ``max_workers`` mirrors vLLM's ``multi_thread_safetensors_weights_iterator``
    (``weight_utils.py`` ~line 930): shard FILES are read in parallel via a
    ``ThreadPoolExecutor`` (each worker opens one file and reads only its
    wanted tensors, returning a private per-file dict — no shared-dict writes,
    so there is no cross-thread race merging into ``out``). ``max_workers=1``
    (or a single matched file) falls back to the original sequential loop, so
    both paths are always available and return identical results.
    """
    index = build_weight_index(path)
    names = [n for n in index if wanted(n)]
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)

    def _read_file(item: tuple[str, list[str]]) -> dict:
        f, ns = item
        result: dict = {}
        with safe_open(f, framework="pt", device=str(device)) as sf:
            for n in ns:
                t = sf.get_tensor(n)
                if dtype is not None and t.is_floating_point():
                    t = t.to(dtype)
                result[n] = t
        return result

    items = list(by_file.items())
    out: dict = {}
    if max_workers <= 1 or len(items) <= 1:
        for item in items:
            out.update(_read_file(item))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for result in ex.map(_read_file, items):
                out.update(result)
    return out


_EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")


def expert_shard_predicate(n_routed_experts: int, ep_size: int, ep_rank: int,
                           base=None):
    """Predicate keeping every non-expert weight + only THIS ep_rank's experts.

    MoE expert weights are named ``model.layers.{i}.mlp.experts.{e}.*``; experts
    outside ``[ep_rank*n_local, (ep_rank+1)*n_local)`` are dropped so each rank
    loads only its slice of the 256 experts (spec §6/§15.1). ``base`` optionally
    composes an extra filter (e.g. a TP predicate for attention/dense weights).
    """
    if ep_size <= 1:
        return base or (lambda name: True)
    n_local = n_routed_experts // ep_size
    lo, hi = ep_rank * n_local, (ep_rank + 1) * n_local

    def keep(name: str) -> bool:
        if base is not None and not base(name):
            return False
        m = _EXPERT_RE.search(name)
        if m is None:
            return True                       # non-expert weight: every rank keeps
        return lo <= int(m.group(1)) < hi     # expert: only local slice

    return keep
