"""HTTP-serving Prometheus metrics."""
import resource
import sys
import time

from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                               generate_latest)

_SERVING_STAGES = (
    "http_parse",
    "admission_wait",
    "tokenize",
    "engine_queue",
    "prefill",
    "decode",
    "detokenize",
    "sse_send",
    "ttft",
    "itl",
    "e2e",
)


class ServingMetrics:
    """Per-runtime Prometheus metrics backed only by host observations."""

    def __init__(self):
        self.registry = CollectorRegistry()
        self._stages = {
            stage: Histogram(
                f"auto_infer_serving_{stage}_seconds",
                f"Host-observed {stage.replace('_', ' ')} duration in seconds.",
                registry=self.registry,
            )
            for stage in _SERVING_STAGES
        }
        self._requests = Counter(
            "auto_infer_serving_requests_total",
            "Completed Serving requests.",
            ("endpoint", "status"),
            registry=self.registry,
        )
        self._rejections = Counter(
            "auto_infer_serving_rejections_total",
            "Requests rejected before Engine admission.",
            ("stage",),
            registry=self.registry,
        )
        self._errors = Counter(
            "auto_infer_serving_errors_total",
            "Serving request failures.",
            ("type",),
            registry=self.registry,
        )
        self._aborts = Counter(
            "auto_infer_serving_aborts_total",
            "Requests aborted after admission.",
            registry=self.registry,
        )
        self._running = Gauge(
            "auto_infer_serving_running_requests",
            "Requests currently running in the engine.",
            registry=self.registry,
        )
        self._waiting = Gauge(
            "auto_infer_serving_waiting_requests",
            "Requests currently waiting for engine work.",
            registry=self.registry,
        )
        self._kv_utilization = Gauge(
            "auto_infer_serving_kv_cache_utilization",
            "Fraction of KV blocks currently in use.",
            registry=self.registry,
        )
        self._tokens = Counter(
            "auto_infer_serving_tokens_total",
            "Serving tokens processed by kind.",
            ("kind",),
            registry=self.registry,
        )
        self._process_cpu = Gauge(
            "auto_infer_serving_process_cpu_seconds",
            "Serving process CPU time in seconds.",
            registry=self.registry,
        )
        self._process_peak_rss = Gauge(
            "auto_infer_serving_process_peak_rss_bytes",
            "Serving process peak resident set size in bytes.",
            registry=self.registry,
        )

    def observe_stage(self, stage: str, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("stage duration must be >= 0")
        try:
            metric = self._stages[stage]
        except KeyError as error:
            raise ValueError(f"unknown Serving stage: {stage}") from error
        metric.observe(seconds)

    def record_request(self, endpoint: str, status: str) -> None:
        self._requests.labels(endpoint=endpoint, status=status).inc()

    def record_tokens(self, *, prompt: int, generated: int) -> None:
        self._tokens.labels(kind="prompt").inc(prompt)
        self._tokens.labels(kind="generated").inc(generated)

    def record_rejection(self, stage: str) -> None:
        self._rejections.labels(stage=stage).inc()

    def record_error(self, error_type: str) -> None:
        self._errors.labels(type=error_type).inc()

    def record_abort(self) -> None:
        self._aborts.inc()

    def set_load(self, *, running: int, waiting: int,
                 kv_utilization: float) -> None:
        self._running.set(running)
        self._waiting.set(waiting)
        self._kv_utilization.set(kv_utilization)

    def render(self) -> str:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak_rss *= 1024
        self._process_cpu.set(time.process_time())
        self._process_peak_rss.set(peak_rss)
        return generate_latest(self.registry).decode("utf-8")
