"""Bounded asynchronous micro-batching for blocking tokenizer backends."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any


class TokenizerOverloaded(RuntimeError):
    pass


class TokenizerClosed(RuntimeError):
    pass


@dataclass(frozen=True)
class _Work:
    operation: str
    payload: Any
    kwargs: dict[str, Any]
    key: tuple[Any, ...]
    future: asyncio.Future


_STOP = object()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


class AsyncTokenizer:
    """Run tokenizer work off-loop while merging compatible adjacent calls."""

    def __init__(
        self,
        tokenizer,
        *,
        max_batch_size: int = 32,
        wait_s: float = 0.002,
        queue_capacity: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if wait_s < 0:
            raise ValueError("wait_s must be >= 0")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be > 0")
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.wait_s = wait_s
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[_Work | object] = asyncio.Queue(
            maxsize=queue_capacity
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="AutoInferTokenizer"
        )
        self._closed = False
        self._executor_closed = False
        self._batch_task = self._loop.create_task(
            self._batch_loop(), name="AutoInferTokenizerBatcher"
        )

    async def encode(self, prompt: str, **kwargs) -> list[int]:
        return await self._submit("encode", prompt, kwargs)

    async def decode(self, token_ids: list[int], **kwargs) -> str:
        return await self._submit("decode", list(token_ids), kwargs)

    async def render_chat(self, messages: list[dict], **kwargs):
        return await self._submit("chat", messages, kwargs)

    async def _submit(self, operation: str, payload: Any, kwargs: dict[str, Any]):
        if self._closed:
            raise TokenizerClosed("tokenizer is closed")
        future = self._loop.create_future()
        work = _Work(
            operation=operation,
            payload=payload,
            kwargs=dict(kwargs),
            key=(operation, _freeze(kwargs)),
            future=future,
        )
        try:
            self._queue.put_nowait(work)
        except asyncio.QueueFull as error:
            raise TokenizerOverloaded("tokenizer queue is full") from error
        return await future

    async def _batch_loop(self) -> None:
        carry: _Work | object | None = None
        while True:
            item = carry if carry is not None else await self._queue.get()
            carry = None
            if item is _STOP:
                return
            assert isinstance(item, _Work)
            batch = [item]
            deadline = self._loop.time() + self.wait_s
            while len(batch) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(
                        self._queue.get(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    break
                if candidate is _STOP:
                    carry = candidate
                    break
                assert isinstance(candidate, _Work)
                if candidate.key != item.key:
                    carry = candidate
                    break
                batch.append(candidate)
            try:
                results = await self._loop.run_in_executor(
                    self._executor, partial(self._execute_batch, batch)
                )
            except BaseException as error:
                for work in batch:
                    if not work.future.done():
                        work.future.set_exception(error)
            else:
                for work, result in zip(batch, results):
                    if not work.future.done():
                        work.future.set_result(result)

    def _execute_batch(self, batch: list[_Work]) -> list[Any]:
        operation = batch[0].operation
        kwargs = batch[0].kwargs
        if operation == "encode":
            encoded = self.tokenizer(
                [work.payload for work in batch], **kwargs
            )
            input_ids = encoded["input_ids"]
            return [list(ids) for ids in input_ids]
        if operation == "decode":
            return list(
                self.tokenizer.batch_decode(
                    [work.payload for work in batch], **kwargs
                )
            )
        if operation == "chat":
            return [
                self.tokenizer.apply_chat_template(work.payload, **kwargs)
                for work in batch
            ]
        raise RuntimeError(f"unknown tokenizer operation: {operation}")

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._queue.put(_STOP)
        await self._batch_task
        if not self._executor_closed:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor_closed = True
