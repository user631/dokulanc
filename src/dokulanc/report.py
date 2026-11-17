from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import CLASS_LABELS
from .document import Check, Document
from .schemas import LIST_FIELDS


def flat_fields(doc: Document) -> dict:
    out = {}
    for name, raw in (doc.fields or {}).items():
        if isinstance(raw, dict):
            out[name] = raw.get("ertek")
        elif name in LIST_FIELDS:
            out[name] = json.dumps(raw, ensure_ascii=False) if raw else ""
        else:
            out[name] = raw
    return out


def summary_row(doc: Document) -> dict:
    row = {
        "fajl": doc.name,
        "tipus": CLASS_LABELS.get(doc.doc_type or "", doc.doc_type or ""),
        "tipus_valoszinuseg": round(doc.top_probability, 3),
        "tipus_forras": doc.type_source,
        "statusz": doc.status(),
        "hibak": sum(1 for c in doc.checks if c.level == Check.ERROR),
        "figyelmeztetesek": sum(1 for c in doc.checks if c.level == Check.WARNING),
        "ido_mp": round(sum(doc.timings.values()), 2),
    }
    row.update(flat_fields(doc))
    return row


def write_json(docs: list[Document], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([d.to_dict() for d in docs], ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(docs: list[Document], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(d) for d in docs]
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        w.writeheader()
        w.writerows(rows)
