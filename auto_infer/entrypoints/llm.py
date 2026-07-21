import itertools

from auto_infer.config import EngineConfig
from auto_infer.engine.engine_core import EngineCore
from auto_infer.engine.executor import Executor, MockExecutor
from auto_infer.engine.request import Request, SamplingParams


class LLM:
    def __init__(self, config: EngineConfig, executor: Executor | None = None):
        self.config = config
        if executor is None:
            from auto_infer.engine.factory import build_executor
            executor = build_executor(config)
        self.engine = EngineCore(config, executor)

    @classmethod
    def for_testing(cls, config: EngineConfig, executor: Executor | None = None):
        """Explicit host-only constructor; production never silently uses a mock."""
        return cls(config, executor or MockExecutor())

    def close(self) -> None:
        self.engine.executor.close()

    def generate(self, prompts: list[list[int]], max_tokens: int = 16,
                 eos_token_id: int | None = None) -> list[list[int]]:
        ids = [f"req-{i}" for i in range(len(prompts))]
        for rid, prompt in zip(ids, prompts):
            self.engine.add_request(
                Request(request_id=rid, prompt_token_ids=list(prompt),
                        sampling=SamplingParams(max_tokens=max_tokens, eos_token_id=eos_token_id))
            )
        results: dict[str, list[int]] = {}
        for _ in itertools.count():
            if not self.engine.has_unfinished():
                break
            for req in self.engine.step():
                results[req.request_id] = list(req.output_token_ids)
        return [results[rid] for rid in ids]

    def generate_stream(self, prompt_ids: list[int], max_tokens: int = 16,
                        eos_token_id: int | None = None):
        """Yield output token ids one at a time as they are produced (for SSE)."""
        rid = "stream-0"
        self.engine.add_request(
            Request(request_id=rid, prompt_token_ids=list(prompt_ids),
                    sampling=SamplingParams(max_tokens=max_tokens, eos_token_id=eos_token_id)))
        emitted = 0
        while self.engine.has_unfinished():
            finished = self.engine.step()
            req = self.engine.scheduler.get_request_or_none(rid)
            if req is None:
                req = next((r for r in finished if r.request_id == rid), None)
            if req is not None:
                for tid in req.output_token_ids[emitted:]:
                    yield tid
                emitted = len(req.output_token_ids)
