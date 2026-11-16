"""Document model; dictionaries are used only when exporting to the HTTP API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Document:
    """One source document and the metadata extracted during corpus loading.

    Source fields cannot be reassigned, preserving citation offsets. Metadata
    remains a dictionary for the different document types; consumers treat it
    as read-only. Exporting creates a separate copy, including nested metadata.
    """

    id: str
    title: str
    filename: str
    text: str
    kind: Literal["company", "law"]
    snapshot_date: str | None
    metadata: dict[str, Any]
    sha256: str

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Return the existing JSON schema without exposing mutable metadata."""
        result = asdict(self)
        if not include_text:
            result.pop("text")
        return result
