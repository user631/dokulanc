from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .normalize import normalize_text, normalize_words, parse_amount, parse_date
from .schemas import scalar_fields
from .validation import AMOUNT_FIELDS, DATE_FIELDS


def clean_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" | ", " ")).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = clean_ocr_text(ref), clean_ocr_text(hyp)
    return Levenshtein.distance(r, h) / max(len(r), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = clean_ocr_text(ref).split(), clean_ocr_text(hyp).split()
    return Levenshtein.distance(r, h) / max(len(r), 1)


def word_recall(ref: str, hyp: str) -> float:
    r = Counter(normalize_words(clean_ocr_text(ref)).split())
    h = Counter(normalize_words(clean_ocr_text(hyp)).split())
    total = sum(r.values())
    return sum(min(c, h[w]) for w, c in r.items()) / max(total, 1)


def value_matches(name: str, gt, pred, threshold: int = 90) -> bool:
    if gt in (None, "") and pred in (None, ""):
        return True
    if gt in (None, "") or pred in (None, ""):
        return False
    if name in AMOUNT_FIELDS or name in {"egysegar", "netto", "afa", "brutto", "osszeg"}:
        a, b = parse_amount(gt), parse_amount(pred)
        return a is not None and b is not None and abs(a - b) < 0.01
    if name in DATE_FIELDS:
        a, b = parse_date(gt), parse_date(pred)
        if a is not None or b is not None:
            return a == b
    a, b = normalize_text(gt), normalize_text(pred)
    return a == b or fuzz.ratio(a, b) >= threshold


def _value(v):
    return v.get("ertek") if isinstance(v, dict) else v


def compare_fields(doc_type: str, gt: dict, pred: dict) -> dict:
    result = {}
    for name in scalar_fields(doc_type):
        if name not in gt:
            continue
        result[name] = value_matches(name, _value(gt.get(name)), _value((pred or {}).get(name)))
    return result


def compare_items(doc_type: str, gt: dict, pred: dict) -> dict:
    pred = pred or {}
    out = {}
    if doc_type in ("szamla", "nyugta"):
        keys = ["megnevezes", "netto", "brutto"] if doc_type == "szamla" else ["megnevezes", "osszeg"]
        g, p = gt.get("tetelek") or [], pred.get("tetelek") or []
        out["tetelszam_egyezik"] = len(g) == len(p)
        hits = total = 0
        for i, item in enumerate(g):
            for k in keys:
                total += 1
                if i < len(p) and value_matches(k, item.get(k), p[i].get(k)):
                    hits += 1
        out["tetel_mezo_pontossag"] = hits / max(total, 1)
    if doc_type == "oneletrajz":
        for list_name, key in (("munkahelyek", "ceg"), ("tanulmanyok", "intezmeny")):
            g = [x.get(key) for x in gt.get(list_name) or []]
            p = [x.get(key) for x in pred.get(list_name) or [] if isinstance(x, dict)]
            out[f"{list_name}_felidezes"] = sum(any(value_matches(key, a, b, 85) for b in p) for a in g) / max(len(g), 1)
        for list_name in ("nyelvek", "keszsegek"):
            g, p = gt.get(list_name) or [], pred.get(list_name) or []
            out[f"{list_name}_felidezes"] = sum(any(value_matches(list_name, a, b, 80) for b in p) for a in g) / max(len(g), 1)
    return out
