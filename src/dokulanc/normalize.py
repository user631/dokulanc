from __future__ import annotations

import re
import unicodedata
from datetime import date

MONTHS = {
    "január": 1, "február": 2, "március": 3, "április": 4, "május": 5, "június": 6,
    "július": 7, "augusztus": 8, "szeptember": 9, "október": 10, "november": 11, "december": 12,
    "jan": 1, "febr": 2, "feb": 2, "márc": 3, "ápr": 4, "máj": 5, "jún": 6, "júl": 7,
    "aug": 8, "szept": 9, "szep": 9, "okt": 10, "nov": 11, "dec": 12,
}
MONTH_NAMES = [None, "január", "február", "március", "április", "május", "június",
               "július", "augusztus", "szeptember", "október", "november", "december"]


def parse_amount(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace(chr(160), " ").lower()
    m = re.search(r"-?\d(?:[\d .,]*\d)?", s)
    if not m:
        return None
    num = m.group(0).replace(" ", "")
    if "," in num and "." in num:
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif "," in num:
        num = num.replace(",", "") if num.count(",") > 1 else num.replace(",", ".")
    elif "." in num:
        parts = num.split(".")
        if len(parts) > 2 or len(parts[-1]) == 3:
            num = num.replace(".", "")
    try:
        return float(num)
    except ValueError:
        return None


def parse_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    s = str(value).strip().lower()
    m = re.search(r"(\d{4})\s*[.\-/]\s*(\d{1,2})(?:\s*[.\-/]\s*|\s+)(\d{1,2})(?!\d)", s)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})\.?\s*([a-záéíóöőúüű]+)\.?\s*(\d{1,2})", s)
    if m and m.group(2) in MONTHS:
        return _safe_date(int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))
    m = re.search(r"(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{4})", s)
    if m:
        return _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def format_amount(value: float, decimals: int = 0, unit: str = "Ft", sep: str = " ") -> str:
    s = f"{value:,.{decimals}f}".replace(",", "\u0001").replace(".", ",").replace("\u0001", sep)
    return f"{s} {unit}".strip()


def format_date(d: date, style: int = 0) -> str:
    if style == 1:
        return f"{d.year}.{d.month:02d}.{d.day:02d}."
    if style == 2:
        return d.isoformat()
    if style == 3:
        return f"{d.year}. {MONTH_NAMES[d.month]} {d.day}."
    return f"{d.year}. {d.month:02d}. {d.day:02d}."


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_text(s: str) -> str:
    s = strip_accents(str(s or "")).lower()
    return re.sub(r"[^0-9a-z]+", "", s)


def normalize_words(s: str) -> str:
    s = strip_accents(str(s or "")).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z@.+\- ]+", " ", s)).strip()
