"""Cheap host-side telemetry for the engine execution loop."""
import logging
import time


logger = logging.getLogger("auto_infer.metrics")


def _ensure_stderr_handler() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class StatLogger:
    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        _ensure_stderr_handler()
        self._t0 = time.monotonic()
        self._reset_window()
        self._finished = 0

    def _reset_window(self) -> None:
        self._prefill_toks = 0
        self._gen_toks = 0
        self._ttfts: list[float] = []
        self._spec_steps = 0
        self._spec_accepted = 0
        self._spec_accepted_per_position: list[int] = []

    def record_step(self, prefill_toks: int, gen_toks: int) -> None:
        self._prefill_toks += prefill_toks
        self._gen_toks += gen_toks

    def record_ttft(self, dt_s: float) -> None:
        self._ttfts.append(dt_s)

    def record_finished(self, n: int) -> None:
        self._finished += n

    def record_spec(self, steps: int, accepted: int,
                    accepted_per_position=()) -> None:
        self._spec_steps += steps
        self._spec_accepted += accepted
        missing = len(accepted_per_position) - len(
            self._spec_accepted_per_position)
        if missing > 0:
            self._spec_accepted_per_position.extend([0] * missing)
        for position, count in enumerate(accepted_per_position):
            self._spec_accepted_per_position[position] += count

    def maybe_log(self, now: float, *, running: int, waiting: int, kv,
                  num_preemptions: int) -> None:
        elapsed = now - self._t0
        if elapsed < self.interval_s:
            return
        kv_used = (1.0 - kv.num_free_blocks() / kv.num_blocks
                   if kv.num_blocks else 0.0)
        prefix_rate = (kv.prefix_hit_blocks / kv.prefix_queried_blocks
                       if kv.prefix_queried_blocks else 0.0)
        parts = [
            f"prefill {self._prefill_toks / elapsed:8.1f} tok/s",
            f"decode {self._gen_toks / elapsed:8.1f} tok/s",
            f"running {running} waiting {waiting}",
            f"kv {kv_used * 100:5.1f}%",
            (f"ttft(avg) {1000 * sum(self._ttfts) / len(self._ttfts):7.1f}ms"
             if self._ttfts else "ttft(avg)     n/a"),
            f"prefix-hit {prefix_rate * 100:5.1f}%",
            f"preempt {num_preemptions}",
            f"finished {self._finished}",
        ]
        if self._spec_steps:
            parts.append(
                f"spec-accept {self._spec_accepted / self._spec_steps + 1:.2f} tok/step")
            rates = ",".join(
                f"{count / self._spec_steps:.1%}"
                for count in self._spec_accepted_per_position)
            if rates:
                parts.append(f"spec-by-pos [{rates}]")
        logger.info("[engine] " + " | ".join(parts))
        self._t0 = now
        self._reset_window()
