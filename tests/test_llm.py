import pytest

from dokulanc.extractor import Extractor
from dokulanc.normalize import normalize_text, parse_amount
from dokulanc.synth import Generator, ground_truth_lines

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def extractor():
    ex = Extractor()
    ok, msg = ex.client.available()
    if not ok:
        pytest.skip(msg)
    return ex


def test_invoice_extraction_from_ideal_text(extractor):
    r = Generator(21).generate("szamla")
    lines = ground_truth_lines({"lines": r.lines})
    out = extractor.extract("szamla", lines)
    assert normalize_text(out["szamlaszam"]["ertek"]) == normalize_text(r.fields["szamlaszam"])
    assert parse_amount(out["brutto_osszesen"]["ertek"]) == parse_amount(r.fields["brutto_osszesen"])
    assert len(out["tetelek"]) == len(r.fields["tetelek"])


def test_llm_classification(extractor):
    r = Generator(22).generate("nyugta")
    tipus, _ = extractor.classify(ground_truth_lines({"lines": r.lines}))
    assert tipus == "nyugta"
