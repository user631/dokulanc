from __future__ import annotations

import re

from .normalize import format_amount, normalize_words, parse_amount

VAT_RATES = (27, 18, 5, 0)


def _number_spans(groups: list[str], end: int, k: int):
    if k == 0:
        yield []
        return
    for start in range(end - 1, -1, -1):
        size = end - start
        if size > 4:
            break
        head = groups[start]
        if size > 1 and (len(head) > 3 or any(len(g) != 3 for g in groups[start + 1:end])):
            continue
        if len(head) > 1 and head.startswith("0"):
            continue
        value = int("".join(groups[start:end]))
        for rest in _number_spans(groups, start, k - 1):
            yield rest + [value]


def candidates(line_text: str, qty: float | None, rate_hint: float | None):
    variants = []
    text = line_text.replace("|", " ")
    m = re.search(r"(\d{1,2})\s*%", text)
    if m:
        variants.append((text[:m.start()] + " " + text[m.end():], [float(m.group(1))]))
    else:
        for m in re.finditer(r"(?<!\d)(27|18|5)(?:96|9o|9)(?!\d)", text):
            variants.append((text[:m.start()] + " " + text[m.end():], [float(m.group(1))]))
        variants.append((text, [rate_hint] if rate_hint is not None and rate_hint in VAT_RATES else list(VAT_RATES)))
    found = set()
    for variant_text, rates in variants:
        found |= _solve(re.findall(r"\d+", variant_text), rates, qty)
    return found


def _solve(groups: list[str], rates: list[float], qty: float | None) -> set:
    found = set()
    for r in rates:
        for k in (4, 3):
            for nums in _number_spans(groups, len(groups), k):
                unit, net, gross = (nums[0], nums[1], nums[-1]) if k == 4 else (nums[0], nums[1], nums[2])
                if net <= 0:
                    continue
                if abs(net * (1 + r / 100) - gross) > 1.0:
                    continue
                if qty is not None and abs(qty * unit - net) > 1.0:
                    continue
                if qty is None and unit > net:
                    continue
                found.add((float(unit), float(net), float(gross - net), float(gross), float(r)))
    return found


def item_consistent(item: dict, rate: float | None, qty: float | None) -> bool:
    unit, net, vat, gross = (parse_amount(item.get(k)) for k in ("egysegar", "netto", "afa", "brutto"))
    if None in (net, vat, gross) or abs(net + vat - gross) > 1.0:
        return False
    if rate is not None and abs(net * rate / 100 - vat) > 1.0:
        return False
    if qty is not None and unit is not None and abs(qty * unit - net) > 1.0:
        return False
    return True


def repair_invoice_items(items: list[dict], line_lookup) -> list[tuple[int, dict, dict]]:
    repaired = []
    for i, item in enumerate(items):
        qty = parse_amount(item.get("mennyiseg"))
        rate_raw = item.get("afa_kulcs") or ""
        rate = parse_amount(rate_raw) if "%" in rate_raw or rate_raw.strip().isdigit() else None
        if item_consistent(item, rate, qty):
            continue
        line = line_lookup(item.get("sor"))
        if line is None:
            continue
        sols = candidates(line.text, qty, rate)
        if len(sols) != 1:
            continue
        unit, net, vat, gross, r = sols.pop()
        new = {
            "egysegar": format_amount(unit, 0, ""), "netto": format_amount(net, 0, ""), "afa": format_amount(vat, 0, ""),
            "brutto": format_amount(gross, 0, ""), "afa_kulcs": f"{r:g}%",
        }
        old = {k: item.get(k) for k in new}
        item.update(new)
        repaired.append((i, old, new))
    return repaired


TOTAL_LABELS = {
    "szamla": {
        "netto_osszesen": r"netto\s*osszesen",
        "afa_osszesen": r"afa\s*osszesen",
        "brutto_osszesen": r"brutto\s*vegosszeg|fizetendo(?:\s*osszesen)?|vegosszeg",
    },
    "nyugta": {
        "vegosszeg": r"osszesen|fizetendo|vegosszeg",
    },
}


def label_fallback(doc_type: str, fields: dict, lines, is_present) -> list[tuple[str, str, str, int]]:
    changes = []
    for name, pattern in TOTAL_LABELS.get(doc_type, {}).items():
        raw = fields.get(name)
        if not isinstance(raw, dict):
            continue
        current = raw.get("ertek")
        if current is not None and parse_amount(current) is not None and is_present(current):
            continue
        for line in lines:
            norm = normalize_words(line.text)
            m = re.search(pattern, norm)
            if not m:
                continue
            value = parse_amount(norm[m.end():])
            if value is None:
                continue
            new = format_amount(value, 0)
            raw["ertek"], raw["sor"] = new, line.id
            changes.append((name, current, new, line.id))
            break
    return changes


def _side(value: str, text: str) -> str | None:
    if "|" not in text or not value:
        return None
    left, right = text.split("|", 1)
    v = normalize_words(value)
    in_left, in_right = v in normalize_words(left), v in normalize_words(right)
    if in_left == in_right:
        return None
    return "left" if in_left else "right"


def fix_party_swap(fields: dict, lines) -> bool:
    label_line = None
    for line in lines:
        norm = normalize_words(line.text)
        seller = re.search(r"elado|szallito", norm)
        buyer = re.search(r"vevo", norm)
        if seller and buyer:
            label_line = "left" if seller.start() < buyer.start() else "right"
            break
    if label_line is None:
        return False
    by_id = {line.id: line for line in lines}
    seller_name, buyer_name = fields.get("elado_nev") or {}, fields.get("vevo_nev") or {}
    s_line, b_line = by_id.get(seller_name.get("sor")), by_id.get(buyer_name.get("sor"))
    if s_line is None or b_line is None:
        return False
    s_side = _side(seller_name.get("ertek"), s_line.text)
    b_side = _side(buyer_name.get("ertek"), b_line.text)
    if s_side is None or b_side is None or s_side == b_side or s_side == label_line:
        return False
    for key in ("nev", "cim", "adoszam"):
        a, b = f"elado_{key}", f"vevo_{key}"
        if a in fields and b in fields:
            fields[a], fields[b] = fields[b], fields[a]
    return True
