import asyncio
import threading

import pytest

from auto_infer.serving.tokenizer import (AsyncTokenizer, TokenizerClosed,
                                          TokenizerOverloaded)


class RecordingTokenizer:
    def __init__(self):
        self.batch_calls = []
        self.decode_calls = []
        self.chat_calls = []

    def __call__(self, prompts, **kwargs):
        self.batch_calls.append((list(prompts), kwargs))
        return {"input_ids": [[len(prompt)] for prompt in prompts]}

    def batch_decode(self, token_ids, **kwargs):
        self.decode_calls.append(([list(ids) for ids in token_ids], kwargs))
        return ["-".join(map(str, ids)) for ids in token_ids]

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return [len(messages)]


def test_compatible_encodes_are_microbatched():
    async def scenario():
        backend = RecordingTokenizer()
        tokenizer = AsyncTokenizer(
            backend, max_batch_size=4, wait_s=0.01, queue_capacity=8
        )

        outputs = await asyncio.gather(
            tokenizer.encode("a", add_special_tokens=False),
            tokenizer.encode("bb", add_special_tokens=False),
        )

        assert outputs == [[1], [2]]
        assert backend.batch_calls == [
            (["a", "bb"], {"add_special_tokens": False})
        ]
        await tokenizer.aclose()

    asyncio.run(scenario())


def test_decode_and_chat_template_run_through_async_worker():
    async def scenario():
        backend = RecordingTokenizer()
        tokenizer = AsyncTokenizer(
            backend, max_batch_size=4, wait_s=0, queue_capacity=8
        )

        decoded = await tokenizer.decode([1, 2], skip_special_tokens=True)
        rendered = await tokenizer.render_chat(
            [{"role": "user", "content": "x"}],
            add_generation_prompt=True,
            tokenize=True,
        )

        assert decoded == "1-2"
        assert rendered == [1]
        assert backend.decode_calls == [
            ([[1, 2]], {"skip_special_tokens": True})
        ]
        assert backend.chat_calls[0][1] == {
            "add_generation_prompt": True,
            "tokenize": True,
        }
        await tokenizer.aclose()

    asyncio.run(scenario())


def test_full_tokenizer_queue_rejects_without_waiting():
    class BlockingTokenizer(RecordingTokenizer):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def __call__(self, prompts, **kwargs):
            self.entered.set()
            assert self.release.wait(timeout=5)
            return super().__call__(prompts, **kwargs)

    async def scenario():
        backend = BlockingTokenizer()
        tokenizer = AsyncTokenizer(
            backend, max_batch_size=1, wait_s=0, queue_capacity=1
        )
        first = asyncio.create_task(tokenizer.encode("active"))
        assert await asyncio.to_thread(backend.entered.wait, 2)
        second = asyncio.create_task(tokenizer.encode("queued"))
        await asyncio.sleep(0)

        with pytest.raises(TokenizerOverloaded, match="queue is full"):
            await tokenizer.encode("rejected")

        backend.release.set()
        assert await first == [6]
        assert await second == [6]
        await tokenizer.aclose()

    asyncio.run(scenario())


def test_close_is_idempotent_and_rejects_new_work():
    async def scenario():
        tokenizer = AsyncTokenizer(
            RecordingTokenizer(), max_batch_size=1, wait_s=0, queue_capacity=1
        )

        await tokenizer.aclose()
        await tokenizer.aclose()

        with pytest.raises(TokenizerClosed, match="closed"):
            await tokenizer.encode("late")

    asyncio.run(scenario())
