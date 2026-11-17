from __future__ import annotations

import re
from datetime import date, timedelta

from rapidfuzz import fuzz

from . import config
from . import identifiers as ids
from .document import Check, Document
from .normalize import normalize_text, parse_amount, parse_date
from .repair import fix_party_swap, label_fallback, repair_invoice_items
from .schemas import FREE_TEXT_FIELDS, LIST_FIELDS

AMOUNT_FIELDS = {"netto_osszesen", "afa_osszesen", "brutto_osszesen", "vegosszeg"}
DATE_FIELDS = {"kiallitas_datuma", "teljesites_datuma", "fizetesi_hatarido", "datum", "szuletesi_datum", "hatarido", "kelt"}
TAX_FIELDS = {"elado_adoszam", "vevo_adoszam", "adoszam"}
VAT_RATES = {0.0, 5.0, 18.0, 27.0}
VAT_TEXT = {"aam", "tam", "eu", "fad", "ka", "mentes"}

REQUIRED = {
    "szamla": ["szamlaszam", "kiallitas_datuma", "elado_nev", "elado_adoszam", "brutto_osszesen"],
    "nyugta": ["uzlet_nev", "datum", "vegosszeg"],
    "level": ["felado", "cimzett", "datum", "targy"],
    "oneletrajz": ["nev"],
    "urlap": ["nev", "szuletesi_datum"],
}

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def value_of(fields: dict, name: str):
    v = fields.get(name)
    if isinstance(v, dict):
        v = v.get("ertek")
    if isinstance(v, str) and not v.strip():
        return None
    return v


def numbers_in(text: str) -> list[float]:
    out = []
    for part in text.split("|"):
        for m in re.finditer(r"-?\d(?:[\d .,]*\d)?", part):
            v = parse_amount(m.group(0))
            if v is not None:
                out.append(v)
            for tok in m.group(0).split():
                t = parse_amount(tok)
                if t is not None:
                    out.append(t)
    return out


def found_in_text(value, text: str, kind: str = "text") -> bool:
    if value is None or not text:
        return False
    if kind == "amount":
        target = parse_amount(value)
        if target is None:
            return False
        if any(abs(n - target) < 0.01 for n in numbers_in(text)):
            return True
        digits_text = re.sub(r"\D", "", text)
        return str(int(abs(target))) in digits_text
    nv = normalize_text(value)
    nt = normalize_text(text)
    if not nv:
        return True
    if nv in nt:
        return True
    return fuzz.partial_ratio(nv, nt) >= config.FUZZY_FOUND_THRESHOLD


class Validator:
    def __init__(self, today: date | None = None):
        self.today = today or date.today()

    def validate(self, doc: Document) -> tuple[dict, list[Check]]:
        fields = doc.fields or {}
        checks: list[Check] = []
        parsed: dict = {}
        if not fields:
            return parsed, [Check("dokumentum", Check.ERROR, "Nincs kinyert adat.")]
        self._pre_repair(doc, fields, checks)
        if doc.ocr_lines:
            mean_conf = sum(l.confidence for l in doc.ocr_lines) / len(doc.ocr_lines)
            if mean_conf < config.LOW_IMAGE_QUALITY:
                checks.append(Check("dokumentum", Check.WARNING,
                                    f"Gyenge képminőség: az átlagos OCR-konfidencia {mean_conf:.2f}, ezért a kinyert adatok "
                                    "megbízhatatlanok lehetnek. Érdemes élesebb, jobban megvilágított fotót készíteni."))
        for name in REQUIRED.get(doc.doc_type, []):
            if value_of(fields, name) is None:
                checks.append(Check(name, Check.WARNING, "Hiányzó kötelező mező."))
        self._grounding(doc, fields, checks)
        for name, raw in fields.items():
            v = value_of(fields, name)
            if v is None or name in LIST_FIELDS or name in FREE_TEXT_FIELDS:
                continue
            if name in AMOUNT_FIELDS:
                parsed[name] = parse_amount(v)
                if parsed[name] is None:
                    checks.append(Check(name, Check.WARNING, f"Nem értelmezhető összeg, ellenőrizendő: {v}"))
            elif name in DATE_FIELDS:
                parsed[name] = parse_date(v)
                if parsed[name] is None:
                    checks.append(Check(name, Check.WARNING, f"Nem értelmezhető dátum, ellenőrizendő: {v}"))
                else:
                    self._date_plausible(name, parsed[name], checks)
            else:
                parsed[name] = v
        for name in TAX_FIELDS:
            v = value_of(fields, name)
            if v is not None:
                identifier_check(name, v, ids.is_valid_tax_number, "Az adószám formátuma és ellenőrző számjegye helyes.",
                                 "Érvénytelen adószám (formátum vagy ellenőrző számjegy)", checks)
        if value_of(fields, "bankszamla") is not None:
            v = value_of(fields, "bankszamla")
            is_iban = re.sub(r"\s", "", v)[:2].isalpha()
            identifier_check("bankszamla", v, valid_bank,
                             "Az IBAN ellenőrző összege helyes." if is_iban else "A bankszámlaszám ellenőrző számjegyei helyesek.",
                             "Érvénytelen IBAN" if is_iban else "Érvénytelen bankszámlaszám", checks, keep_prefix=2 if is_iban else 0)
        if value_of(fields, "email") is not None:
            ok = bool(EMAIL_RE.match(str(value_of(fields, "email")).replace(" ", "")))
            checks.append(Check("email", Check.OK if ok else Check.WARNING,
                                "Az e-mail-cím formátuma helyes." if ok else "Hibás formátumú e-mail-cím."))
        if value_of(fields, "telefon") is not None:
            d = ids.digits(str(value_of(fields, "telefon")))
            ok = 8 <= len(d) <= 12 and (d.startswith("36") or d.startswith("06"))
            checks.append(Check("telefon", Check.OK if ok else Check.WARNING,
                                "A telefonszám formátuma helyes." if ok else "Szokatlan formátumú telefonszám."))
        handler = getattr(self, f"_check_{doc.doc_type}", None)
        if handler:
            handler(doc, fields, parsed, checks)
        return parsed, checks

    def _pre_repair(self, doc: Document, fields: dict, checks: list[Check]):
        if doc.doc_type == "szamla" and fix_party_swap(fields, doc.ocr_lines):
            checks.append(Check("elado_nev", Check.WARNING,
                                "Az eladó és a vevő adatai a címkék sorrendje alapján fel voltak cserélve; javítva."))
        text = doc.text
        for name, old, new, line_id in label_fallback(doc.doc_type, fields, doc.ocr_lines,
                                                      lambda v: found_in_text(v, text, "amount")):
            checks.append(Check(name, Check.WARNING,
                                f"Az LLM értéke ({old}) nem volt használható; a címke alapján a(z) {line_id}. sorból kiolvasva: {new}."))

    def _grounding(self, doc: Document, fields: dict, checks: list[Check]):
        text = doc.text
        for name, raw in fields.items():
            if name in FREE_TEXT_FIELDS or name in LIST_FIELDS or not isinstance(raw, dict):
                continue
            v = raw.get("ertek")
            if v is None or not str(v).strip():
                continue
            kind = "amount" if name in AMOUNT_FIELDS else "text"
            line = doc.line(raw.get("sor"))
            if raw.get("sor") is not None and line is None:
                checks.append(Check(name, Check.WARNING, f"A megadott forrássor ({raw.get('sor')}) nem létezik."))
            if line is not None and found_in_text(v, line.text, kind):
                if line.confidence < config.LOW_OCR_CONFIDENCE:
                    checks.append(Check(name, Check.WARNING,
                                        f"Alacsony OCR-konfidenciájú sorból származik ({line.confidence:.2f})."))
                continue
            if not found_in_text(v, text, kind):
                checks.append(Check(name, Check.WARNING,
                                    "Az érték nem található a felismert szövegben (lehetséges hallucináció vagy OCR-hiba)."))
        for list_name in ("tetelek",):
            for i, item in enumerate(fields.get(list_name) or []):
                name = item.get("megnevezes")
                if name and not found_in_text(name, text):
                    checks.append(Check(f"{list_name}[{i + 1}]", Check.WARNING,
                                        f"A tétel megnevezése nem található a szövegben: {name}"))

    def _date_plausible(self, name: str, d: date, checks: list[Check]):
        if name == "szuletesi_datum":
            age = (self.today - d).days / 365.25
            if not 14 <= age <= 110:
                checks.append(Check(name, Check.WARNING, f"Valószínűtlen születési dátum ({d.isoformat()})."))
        elif d > self.today + timedelta(days=400) or d.year < 1990:
            checks.append(Check(name, Check.WARNING, f"Valószínűtlen dátum ({d.isoformat()})."))

    def _tolerance(self, n: int) -> float:
        return max(config.AMOUNT_TOLERANCE, 0.5 * n + 0.5)

    def _check_szamla(self, doc: Document, fields: dict, parsed: dict, checks: list[Check]):
        net, vat, gross = parsed.get("netto_osszesen"), parsed.get("afa_osszesen"), parsed.get("brutto_osszesen")
        items = fields.get("tetelek") or []
        for i, old, new in repair_invoice_items(items, doc.line):
            changed = ", ".join(f"{k}: {old[k]} → {new[k]}" for k in new if parse_amount(old[k]) != parse_amount(new[k]))
            checks.append(Check(f"tetelek[{i + 1}]", Check.WARNING,
                                f"A tétel számai az aritmetikai összefüggések alapján a forrássorból javítva ({changed})."))
        totals_ok = None
        item_level = Check.ERROR
        if None not in (net, vat, gross):
            diff = net + vat - gross
            totals_ok = abs(diff) <= self._tolerance(len(items))
            item_level = Check.WARNING if totals_ok else Check.ERROR
            if totals_ok:
                checks.append(Check("brutto_osszesen", Check.OK, "Nettó + ÁFA = bruttó végösszeg."))
            else:
                checks.append(Check("brutto_osszesen", Check.ERROR,
                                    f"Nettó + ÁFA ({net + vat:,.0f}) ≠ bruttó végösszeg ({gross:,.0f}), eltérés: {diff:,.0f}."))
        parsed_items = []
        for i, it in enumerate(items):
            p = {k: parse_amount(it.get(k)) for k in ("egysegar", "netto", "afa", "brutto")}
            qty = parse_amount(it.get("mennyiseg"))
            rate_raw = (it.get("afa_kulcs") or "").strip().lower()
            rate = parse_amount(rate_raw) if rate_raw and not any(t in rate_raw for t in VAT_TEXT) else None
            label = f"tetelek[{i + 1}]"
            if rate_raw:
                if rate is not None and rate not in VAT_RATES:
                    checks.append(Check(label, item_level, f"Nem létező ÁFA-kulcs: {it.get('afa_kulcs')}"))
            if p["netto"] is not None and rate is not None and p["afa"] is not None:
                if abs(p["netto"] * rate / 100 - p["afa"]) > 1.0:
                    checks.append(Check(label, item_level,
                                        f"A tétel ÁFA-összege ({p['afa']:,.0f}) nem egyezik a nettó × kulcs értékkel ({p['netto'] * rate / 100:,.0f})."))
            if p["netto"] is not None and p["afa"] is not None and p["brutto"] is not None:
                if abs(p["netto"] + p["afa"] - p["brutto"]) > 1.0:
                    checks.append(Check(label, item_level, "A tétel nettó + ÁFA összege nem egyezik a bruttóval."))
            if qty is not None and p["egysegar"] is not None and p["netto"] is not None:
                if abs(qty * p["egysegar"] - p["netto"]) > 1.0:
                    checks.append(Check(label, item_level, "Mennyiség × egységár ≠ nettó érték."))
            parsed_items.append({**p, "mennyiseg": qty, "afa_kulcs": rate})
        parsed["tetelek"] = parsed_items
        if parsed_items:
            for key, total_name, total in (("netto", "netto_osszesen", net), ("afa", "afa_osszesen", vat), ("brutto", "brutto_osszesen", gross)):
                vals = [it[key] for it in parsed_items]
                if total is None or any(v is None for v in vals):
                    continue
                s = sum(vals)
                if abs(s - total) <= self._tolerance(len(vals)):
                    checks.append(Check(total_name, Check.OK, f"A tételek {key} értékeinek összege egyezik."))
                else:
                    hint = " A végösszegek egymással egyeznek, ezért valószínűleg a tételsorok felismerése hibás." if totals_ok else ""
                    checks.append(Check(total_name, item_level,
                                        f"A tételek {key} összege ({s:,.0f}) ≠ {total_name} ({total:,.0f}).{hint}"))
        issue, due, perf = parsed.get("kiallitas_datuma"), parsed.get("fizetesi_hatarido"), parsed.get("teljesites_datuma")
        if issue and due:
            if due < issue:
                checks.append(Check("fizetesi_hatarido", Check.ERROR, "A fizetési határidő korábbi, mint a kiállítás dátuma."))
            else:
                checks.append(Check("fizetesi_hatarido", Check.OK, "A fizetési határidő a kiállítás után van."))
        if issue and perf and (perf - issue).days > 31:
            checks.append(Check("teljesites_datuma", Check.WARNING, "A teljesítés jóval a kiállítás után van."))

    def _check_nyugta(self, doc: Document, fields: dict, parsed: dict, checks: list[Check]):
        total = parsed.get("vegosszeg")
        vals = [parse_amount(it.get("osszeg")) for it in fields.get("tetelek") or []]
        parsed["tetelek"] = vals
        if total is not None and vals and all(v is not None for v in vals):
            s = sum(vals)
            if abs(s - total) <= config.AMOUNT_TOLERANCE:
                checks.append(Check("vegosszeg", Check.OK, "A tételek összege egyezik a végösszeggel."))
            else:
                checks.append(Check("vegosszeg", Check.ERROR, f"A tételek összege ({s:,.0f}) ≠ végösszeg ({total:,.0f})."))

    def _check_level(self, doc: Document, fields: dict, parsed: dict, checks: list[Check]):
        d, h = parsed.get("datum"), parsed.get("hatarido")
        if d and h:
            if h < d:
                checks.append(Check("hatarido", Check.WARNING, "A határidő korábbi, mint a levél dátuma."))
            else:
                checks.append(Check("hatarido", Check.OK, f"Határidő: {(h - d).days} nappal a levél dátuma után."))

    def _check_urlap(self, doc: Document, fields: dict, parsed: dict, checks: list[Check]):
        raw_tax_id = value_of(fields, "adoazonosito_jel")
        if raw_tax_id is not None:
            tax_id = identifier_check("adoazonosito_jel", raw_tax_id, ids.is_valid_tax_id,
                                      "Az adóazonosító jel ellenőrző számjegye helyes.", "Érvénytelen adóazonosító jel", checks)
            born = parsed.get("szuletesi_datum")
            encoded = ids.tax_id_birth_date(tax_id) if tax_id else None
            if born and encoded:
                if born == encoded:
                    checks.append(Check("szuletesi_datum", Check.OK, "A születési dátum egyezik az adóazonosító jelben kódolttal."))
                else:
                    checks.append(Check("szuletesi_datum", Check.ERROR,
                                        f"A születési dátum nem egyezik az adóazonosító jelben kódolttal ({encoded.isoformat()})."))
        taj = value_of(fields, "taj_szam")
        if taj is not None:
            identifier_check("taj_szam", taj, ids.is_valid_taj, "A TAJ-szám ellenőrző számjegye helyes.",
                             "Érvénytelen TAJ-szám", checks)


OCR_DIGIT_MAP = str.maketrans({"O": "0", "o": "0", "Q": "0", "I": "1", "l": "1", "|": "1"})


def ocr_fix(value: str, keep_prefix: int = 0) -> str:
    v = str(value)
    return v[:keep_prefix] + v[keep_prefix:].translate(OCR_DIGIT_MAP)


def valid_bank(value: str) -> bool:
    s = re.sub(r"\s", "", value).upper()
    if s[:2].isalpha():
        return ids.is_valid_iban(s) and (not s.startswith("HU") or ids.is_valid_hu_account(s[4:]))
    return ids.is_valid_hu_account(value)


def identifier_check(field: str, value: str, is_valid, ok_msg: str, error_msg: str, checks: list[Check],
                     keep_prefix: int = 0) -> str | None:
    if is_valid(value):
        checks.append(Check(field, Check.OK, ok_msg))
        return value
    fixed = ocr_fix(value, keep_prefix)
    if fixed != value and is_valid(fixed):
        checks.append(Check(field, Check.WARNING, f"{ok_msg} OCR-karaktercsere javítása után: {value} → {fixed}"))
        return fixed
    checks.append(Check(field, Check.ERROR, f"{error_msg}: {value}"))
    return None
