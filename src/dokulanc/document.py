from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class OcrLine:
    id: int
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "bbox": list(self.bbox), "confidence": round(self.confidence, 4)}


@dataclass
class Check:
    field: str
    level: str
    message: str

    OK = "ok"
    WARNING = "figyelmeztetes"
    ERROR = "hiba"


@dataclass
class Document:
    source: str
    image: np.ndarray | None = None
    original_image: np.ndarray | None = None
    doc_type: str | None = None
    type_probs: dict[str, float] = field(default_factory=dict)
    type_source: str = ""
    llm_type: str | None = None
    ocr_profile: str = ""
    ocr_engine: str = ""
    ocr_lines: list[OcrLine] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    parsed: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.source).name

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.ocr_lines)

    @property
    def top_probability(self) -> float:
        return max(self.type_probs.values()) if self.type_probs else 0.0

    def line(self, line_id: int | None) -> OcrLine | None:
        if line_id is None:
            return None
        for line in self.ocr_lines:
            if line.id == line_id:
                return line
        return None

    def status(self) -> str:
        levels = {c.level for c in self.checks}
        if Check.ERROR in levels:
            return Check.ERROR
        if Check.WARNING in levels:
            return Check.WARNING
        return Check.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "type_probs": {k: round(v, 4) for k, v in self.type_probs.items()},
            "type_source": self.type_source,
            "llm_type": self.llm_type,
            "ocr_profile": self.ocr_profile,
            "ocr_engine": self.ocr_engine,
            "ocr_lines": [line.to_dict() for line in self.ocr_lines],
            "fields": self.fields,
            "parsed": _jsonable(self.parsed),
            "checks": [asdict(c) for c in self.checks],
            "status": self.status(),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "preprocessing": self.preprocessing,
            "errors": self.errors,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)
