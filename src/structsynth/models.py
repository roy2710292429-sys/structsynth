from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SeedQuestion:
    id: str
    text: str
    type: str = "unknown"
    difficulty: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any], index: int = 0) -> "SeedQuestion":
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"Question {index} has no non-empty 'text'.")
        return cls(
            id=str(row.get("id") or f"q{index + 1:03d}"),
            text=text,
            type=str(row.get("type", "unknown")),
            difficulty=str(row.get("difficulty", "unknown")),
            metadata=dict(row.get("metadata", {})),
        )


@dataclass
class Segment:
    id: str
    question_id: str
    order: int
    text: str
    embedding: list[float] | None = None
    role: str = ""
    operation: str = ""
    abstraction: str = ""

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        row = asdict(self)
        if not include_embedding:
            row.pop("embedding", None)
        return row


@dataclass
class CandidateUnit:
    id: str
    label: str
    definition: str
    segment_ids: list[str]
    exemplars: list[str]
    activation: float
    order_hint: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateSlot:
    id: str
    name: str
    definition: str
    required: bool = True
    source_unit_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Template:
    name: str
    description: str
    slots: list[TemplateSlot]
    invariants: list[str] = field(default_factory=list)
    reasoning_profile: dict[str, Any] = field(default_factory=dict)
    source: str = "spectral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "slots": [slot.to_dict() for slot in self.slots],
            "invariants": self.invariants,
            "reasoning_profile": self.reasoning_profile,
            "source": self.source,
        }


@dataclass
class GeneratedQuestion:
    id: str
    arm: str
    text: str
    slot_texts: dict[str, str] = field(default_factory=dict)
    question_type: str = "unknown"
    depth_level: int | None = None
    reasoning_outline: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
