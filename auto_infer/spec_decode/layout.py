"""Value objects for compacting confirmed speculative-decoding rows."""
from dataclasses import dataclass
from typing import Sequence

from auto_infer.spec_decode.geometry import MtpGeometry


@dataclass(frozen=True)
class ConfirmedLayout:
    source_rows: tuple[int, ...]
    query_lengths: tuple[int, ...]
    cumulative_query_lengths: tuple[int, ...]
    final_rows: tuple[int, ...]
    active_tokens: int


def confirmed_layout(
    accepted: Sequence[int], geometry: MtpGeometry
) -> ConfirmedLayout:
    """Pack each request's confirmed target rows in request order."""
    counts = tuple(int(value) for value in accepted)
    if any(not 0 <= value <= geometry.draft_depth for value in counts):
        raise ValueError(
            f"MTP accepted counts must be between 0 and {geometry.draft_depth}"
        )
    source_rows = []
    cumulative = []
    final_rows = []
    active_tokens = 0
    for request_row, value in enumerate(counts):
        width = value + 1
        start = geometry.query_width * request_row
        source_rows.extend(range(start, start + width))
        active_tokens += width
        cumulative.append(active_tokens)
        final_rows.append(active_tokens - 1)
    return ConfirmedLayout(
        tuple(source_rows),
        tuple(1 + value for value in counts),
        tuple(cumulative),
        tuple(final_rows),
        active_tokens,
    )
