"""Typed source fragments shared by retrieval, routing and citation checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Chunk:
    """An exact source slice; offsets count Unicode characters, end exclusive.

    Corpus chunks have no score. Search returns separate instances carrying a
    query-specific score, so ranking cannot change the stored source fragment.
    """

    id: str
    doc_id: str
    title: str
    clause: str
    text: str
    start: int
    end: int
    suspicious: bool
    section: str
    kind: Literal["company", "law"]
    content_role: Literal["metadata", "heading", "editorial", "body"]
    searchable: bool
    score: float | None = None

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("Chunk offsets must be integer character positions.")
        if self.start < 0 or self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("Chunk text must match the length of its nonempty source span.")

    def to_dict(self) -> dict[str, Any]:
        """Export the existing JSON payload used by the document viewer."""
        result = asdict(self)
        if self.score is None:
            result.pop("score")
        return result
