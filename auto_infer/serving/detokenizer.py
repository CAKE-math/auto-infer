"""Bounded incremental detokenization and stop-string handling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextDelta:
    text: str
    token_count: int
    finish_reason: str | None = None
    finished: bool = False


class _SuffixDecoder:
    """Decode against a small stable token context instead of the full output."""

    def __init__(self, tokenizer, *, context_tokens: int = 8,
                 skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self.skip_special_tokens = skip_special_tokens
        self._uses_pieces = all(
            hasattr(tokenizer, name)
            for name in ("convert_ids_to_tokens", "convert_tokens_to_string")
        )
        self._context: list = []
        self._pending: list = []

    def push(self, token_ids: tuple[int, ...]) -> str:
        if self._uses_pieces:
            pieces = self.tokenizer.convert_ids_to_tokens(
                list(token_ids), skip_special_tokens=self.skip_special_tokens
            )
            if isinstance(pieces, str):
                pieces = [pieces]
            self._pending.extend(pieces)
        else:
            self._pending.extend(token_ids)
        return self._commit(stable_only=True)

    def flush(self) -> str:
        return self._commit(stable_only=False)

    def _render(self, values: list) -> str:
        if not values:
            return ""
        if self._uses_pieces:
            return self.tokenizer.convert_tokens_to_string(values)
        return self.tokenizer.decode(
            values, skip_special_tokens=self.skip_special_tokens
        )

    def _commit(self, *, stable_only: bool) -> str:
        if not self._pending:
            return ""
        before = self._render(self._context)
        combined_values = self._context + self._pending
        after = self._render(combined_values)
        if after.startswith(before):
            delta = after[len(before):]
        else:
            # Some tokenizers normalize a context boundary. Keeping the window
            # small bounds fallback work while avoiding a full-prefix decode.
            delta = self._render(self._pending)
        if stable_only and delta.endswith("�"):
            return ""
        self._context = combined_values[-self.context_tokens:]
        self._pending = []
        return delta


class IncrementalTextDecoder:
    def __init__(self, tokenizer, stop: list[str] | tuple[str, ...],
                 *, skip_special_tokens: bool = True):
        self._decoder = _SuffixDecoder(
            tokenizer, skip_special_tokens=skip_special_tokens
        )
        self._stops = tuple(value for value in stop if value)
        self._pending_text = ""
        self._token_count = 0
        self._finished = False
        self._finish_reason: str | None = None

    def push(self, token_ids: tuple[int, ...]) -> TextDelta:
        if self._finished:
            return self._terminal_delta("")
        self._token_count += len(token_ids)
        self._pending_text += self._decoder.push(token_ids)
        stop_index = self._first_stop_index(self._pending_text)
        if stop_index is not None:
            text = self._pending_text[:stop_index]
            self._pending_text = ""
            self._finished = True
            self._finish_reason = "stop"
            return self._terminal_delta(text)
        holdback = self._stop_prefix_holdback(self._pending_text)
        safe_length = len(self._pending_text) - holdback
        text = self._pending_text[:safe_length]
        self._pending_text = self._pending_text[safe_length:]
        return TextDelta(text=text, token_count=self._token_count)

    def finish(self, reason: str) -> TextDelta:
        if self._finished:
            return self._terminal_delta("")
        self._pending_text += self._decoder.flush()
        text = self._pending_text
        self._pending_text = ""
        self._finished = True
        self._finish_reason = reason
        return self._terminal_delta(text)

    def _terminal_delta(self, text: str) -> TextDelta:
        return TextDelta(
            text=text,
            token_count=self._token_count,
            finish_reason=self._finish_reason,
            finished=True,
        )

    def _first_stop_index(self, text: str) -> int | None:
        indexes = [text.find(stop) for stop in self._stops]
        found = [index for index in indexes if index >= 0]
        return min(found) if found else None

    def _stop_prefix_holdback(self, text: str) -> int:
        maximum = 0
        for stop in self._stops:
            upper = min(len(text), len(stop) - 1)
            for size in range(1, upper + 1):
                if text.endswith(stop[:size]):
                    maximum = max(maximum, size)
        return maximum
