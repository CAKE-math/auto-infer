"""Model-derived speculative-decoding shape contract."""
from dataclasses import dataclass
import re


_MTP_INPUT_PROJECTION = re.compile(
    r"model\.mtp_layers\.(\d+)\.input_proj\.weight"
)
MAX_VERIFIED_GRAPH_MTP_DRAFT_DEPTH = 2


def validate_graph_mtp_depth(draft_depth: int) -> None:
    """Reject graph recurrence beyond the retained NPU token-parity gate."""
    if draft_depth > MAX_VERIFIED_GRAPH_MTP_DRAFT_DEPTH:
        raise ValueError(
            "graph MTP NPU-verified maximum is "
            f"{MAX_VERIFIED_GRAPH_MTP_DRAFT_DEPTH}; requested {draft_depth}")


@dataclass(frozen=True)
class MtpGeometry:
    """Separates requested proposal depth from checkpoint MTP topology."""
    draft_depth: int
    trained_layer_count: int = 1

    def __post_init__(self) -> None:
        if self.draft_depth <= 0:
            raise ValueError("MTP draft depth must be positive")
        if self.trained_layer_count <= 0:
            raise ValueError("trained MTP layer count must be positive")

    @property
    def proposal_depth(self) -> int:
        return self.draft_depth

    @property
    def query_width(self) -> int:
        return self.draft_depth + 1

    def layer_prefix(self, index: int) -> str:
        if not 0 <= index < self.trained_layer_count:
            raise IndexError("MTP layer index outside model geometry")
        return f"model.mtp_layers.{index}."

    @classmethod
    def recurrent_from_weights(cls, weights, draft_depth: int) -> "MtpGeometry":
        trained = cls.from_weights(weights)
        if trained.trained_layer_count != 1:
            raise NotImplementedError(
                "recurrent MTP requires exactly one trained layer; "
                f"checkpoint exposes {trained.trained_layer_count}")
        return cls(draft_depth, trained_layer_count=1)

    @classmethod
    def from_weights(cls, weights) -> "MtpGeometry":
        indices = sorted(
            int(match.group(1))
            for name in weights
            if (match := _MTP_INPUT_PROJECTION.fullmatch(name))
        )
        if not indices:
            raise ValueError("model has no MTP layers")
        if indices != list(range(len(indices))):
            raise ValueError(f"MTP layer indices must be contiguous: {indices}")
        return cls(len(indices), trained_layer_count=len(indices))
