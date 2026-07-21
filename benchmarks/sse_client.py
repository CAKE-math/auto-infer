"""Client-side helpers shared by serving benchmark and validation tools."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    text: str
    done: bool
    completion_tokens: int | None = None


def parse_sse_event(line: str) -> SSEEvent | None:
    if not line.startswith("data: "):
        return None
    data = line.removeprefix("data: ")
    if data == "[DONE]":
        return SSEEvent("", True)
    payload = json.loads(data)
    choice = payload.get("choices", [{}])[0]
    text = choice.get("text")
    if text is None:
        text = choice.get("delta", {}).get("content", "")
    count = payload.get("completion_tokens")
    return SSEEvent(
        str(text or ""), False, int(count) if count is not None else None
    )


def parse_sse_line(line: str) -> tuple[str, bool] | None:
    event = parse_sse_event(line)
    if event is None:
        return None
    return event.text, event.done
