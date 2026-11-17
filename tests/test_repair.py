from dokulanc.document import OcrLine
from dokulanc.normalize import parse_amount
from dokulanc.repair import candidates, fix_party_swap, label_fallback, repair_invoice_items


def test_merged_columns_are_reconstructed():
    sols = candidates("Takarítási szolgáltatás | 10 alkalom | 37 660 376 600 27% 101 682 478 282", 10, None)
    assert sols == {(37660.0, 376600.0, 101682.0, 478282.0, 27.0)}


def test_lost_digit_in_vat_is_recovered_from_net_and_gross():
    sols = candidates("Szoftverfejlesztési szolgáltatás | 2 óra | 17 900 | 35 800 27% | 666 45 466", 2, None)
    assert sols == {(17900.0, 35800.0, 9666.0, 45466.0, 27.0)}


def test_repair_updates_only_inconsistent_items():
    lines = {1: OcrLine(1, "Takarítás | 10 alkalom | 37 660 376 600 27% 101 682 478 282", (0, 0, 1, 1), 0.9),
             2: OcrLine(2, "Kenyér | 5 db | 730 | 3 650 5% | 182 | 3 832", (0, 0, 1, 1), 0.9)}
    items = [
        {"megnevezes": "Takarítás", "mennyiseg": "10 alkalom", "egysegar": "37 660 376 600", "netto": "101 682 478 282",
         "afa_kulcs": "27%", "afa": "101 682 478 282", "brutto": "101 682 478 282", "sor": 1},
        {"megnevezes": "Kenyér", "mennyiseg": "5 db", "egysegar": "730", "netto": "3 650", "afa_kulcs": "5%",
         "afa": "182", "brutto": "3 832", "sor": 2},
    ]
    repaired = repair_invoice_items(items, lines.get)
    assert [r[0] for r in repaired] == [0]
    assert parse_amount(items[0]["netto"]) == 376600 and parse_amount(items[0]["brutto"]) == 478282
    assert items[1]["netto"] == "3 650"



def test_percent_read_as_96_is_handled():
    sols = candidates("Könyv | 3 db | 6 830 | 20 490 596 | 1 024 | 21 514", 3, None)
    assert sols == {(6830.0, 20490.0, 1024.0, 21514.0, 5.0)}


def test_percent_read_as_9_suffix():
    sols = candidates("Domain regisztráció | 12 db | 7 920 | 95 040 279 25 661 120 701", 12, None)
    assert sols == {(7920.0, 95040.0, 25661.0, 120701.0, 27.0)}


def _lines(*texts):
    return [OcrLine(i + 1, t, (0, 10 * i, 100, 10 * i + 8), 0.9) for i, t in enumerate(texts)]


def test_label_fallback_reads_totals_when_llm_fails():
    lines = _lines("Nettó összesen: | 849 340 Ft", "ÁFA összesen: | 229 322 Ft", "Végösszeg: | 1 078 662 Ft")
    fields = {"netto_osszesen": {"ertek": "sorokban nincs", "sor": 13}, "afa_osszesen": {"ertek": "229 322 Ft", "sor": 2},
              "brutto_osszesen": {"ertek": None, "sor": None}}
    changes = label_fallback("szamla", fields, lines, lambda v: True)
    assert {c[0] for c in changes} == {"netto_osszesen", "brutto_osszesen"}
    assert parse_amount(fields["netto_osszesen"]["ertek"]) == 849340
    assert parse_amount(fields["brutto_osszesen"]["ertek"]) == 1078662 and fields["brutto_osszesen"]["sor"] == 3


def test_party_swap_is_detected_from_label_order():
    lines = _lines("Vevő: | Eladó:", "Lukács Tech Zrt. | Varga és Társa Bt.", "Adószám: 84942496-2-05 | Adószám: 10471436-5-38")
    fields = {"elado_nev": {"ertek": "Lukács Tech Zrt.", "sor": 2}, "vevo_nev": {"ertek": "Varga és Társa Bt.", "sor": 2},
              "elado_adoszam": {"ertek": "84942496-2-05", "sor": 3}, "vevo_adoszam": {"ertek": "10471436-5-38", "sor": 3},
              "elado_cim": {"ertek": None, "sor": None}, "vevo_cim": {"ertek": None, "sor": None}}
    assert fix_party_swap(fields, lines)
    assert fields["elado_nev"]["ertek"] == "Varga és Társa Bt." and fields["elado_adoszam"]["ertek"] == "10471436-5-38"
    assert not fix_party_swap(fields, lines)
