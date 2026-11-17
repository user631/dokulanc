import random
from datetime import date

import pytest

from dokulanc import identifiers as ids
from dokulanc.config import CLASSES
from dokulanc.document import Check, Document, OcrLine
from dokulanc.synth import Generator, ground_truth_fields, ground_truth_lines
from dokulanc.validation import Validator, found_in_text


def make_doc(doc_type, fields, lines):
    return Document(source="teszt", doc_type=doc_type, fields=fields, ocr_lines=lines)


def gt_doc(rendered):
    meta = {"doc_type": rendered.doc_type, "fields": rendered.fields, "lines": rendered.lines}
    return make_doc(rendered.doc_type, ground_truth_fields(meta), ground_truth_lines(meta))


@pytest.fixture(scope="module")
def validator():
    return Validator(today=date(2026, 11, 13))


@pytest.mark.parametrize("doc_type", CLASSES)
def test_clean_generated_documents_have_no_errors(doc_type, validator):
    gen = Generator(7)
    for _ in range(15):
        doc = gt_doc(gen.generate(doc_type))
        _, checks = validator.validate(doc)
        errors = [c for c in checks if c.level == Check.ERROR]
        assert not errors, errors


@pytest.mark.parametrize("doc_type", ["szamla", "nyugta", "urlap"])
def test_corrupted_documents_are_detected(doc_type, validator):
    gen = Generator(11)
    for _ in range(20):
        r = gen.generate(doc_type, corrupt=True)
        doc = gt_doc(r)
        _, checks = validator.validate(doc)
        error_fields = {c.field for c in checks if c.level == Check.ERROR}
        assert set(r.errors) & error_fields, (r.errors, error_fields)


def test_hallucinated_value_is_flagged(validator):
    lines = [OcrLine(1, "NYUGTA", (0, 0, 10, 10), 0.99), OcrLine(2, "ÖSSZESEN: | 1 200 Ft", (0, 20, 10, 30), 0.97)]
    fields = {"uzlet_nev": {"ertek": "Kitalált Bolt Kft.", "sor": 1}, "uzlet_cim": {"ertek": None, "sor": None},
              "adoszam": {"ertek": None, "sor": None}, "datum": {"ertek": None, "sor": None},
              "ido": {"ertek": None, "sor": None}, "nyugtaszam": {"ertek": None, "sor": None},
              "fizetesi_mod": {"ertek": None, "sor": None}, "vegosszeg": {"ertek": "1200 Ft", "sor": 2}, "tetelek": []}
    _, checks = validator.validate(make_doc("nyugta", fields, lines))
    msgs = {c.field: c.message for c in checks if c.level == Check.WARNING}
    assert "uzlet_nev" in msgs and "hallucináció" in msgs["uzlet_nev"]
    assert "vegosszeg" not in msgs


def test_low_confidence_source_line(validator):
    lines = [OcrLine(1, "Név: | Kiss Anna", (0, 0, 10, 10), 0.3)]
    fields = {"nev": {"ertek": "Kiss Anna", "sor": 1}}
    _, checks = validator.validate(make_doc("oneletrajz", fields, lines))
    assert any(c.field == "nev" and "konfidenci" in c.message for c in checks)


def test_found_in_text_amounts():
    assert found_in_text("4320 Ft", "Irodaszer | 1 db | 4 320 | 27%", "amount")
    assert not found_in_text("9999 Ft", "Irodaszer | 1 db | 4 320 | 27%", "amount")
    assert found_in_text("Szabó Kft", "Eladó: Szabo Kft.")


def test_ocr_letter_o_in_iban_is_fixed_with_warning(validator):
    acct = ids.generate_hu_account(random.Random(3), 16)
    iban = ids.iban_from_hu_account(acct)
    garbled = iban[:-9] + "OOOO OOOo"
    fields = {"bankszamla": {"ertek": garbled, "sor": None}}
    _, checks = validator.validate(make_doc("szamla", fields, []))
    bank = [c for c in checks if c.field == "bankszamla" and "OCR-karaktercsere" in c.message]
    assert bank and bank[0].level == Check.WARNING


def test_poor_image_quality_warning(validator):
    lines = [OcrLine(i, f"sor {i}", (0, 10 * i, 10, 10 * i + 8), 0.3) for i in range(1, 6)]
    _, checks = validator.validate(make_doc("level", {"targy": {"ertek": "sor 1", "sor": 1}}, lines))
    assert any(c.field == "dokumentum" and "képminőség" in c.message for c in checks)
    good = [OcrLine(i, f"sor {i}", (0, 10 * i, 10, 10 * i + 8), 0.95) for i in range(1, 6)]
    _, checks = validator.validate(make_doc("level", {"targy": {"ertek": "sor 1", "sor": 1}}, good))
    assert not any(c.field == "dokumentum" for c in checks)
