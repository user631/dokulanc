from datetime import date

import pytest

from dokulanc.normalize import format_amount, format_date, normalize_text, parse_amount, parse_date


@pytest.mark.parametrize("raw,expected", [
    ("12 345 Ft", 12345.0),
    ("12 345,50 Ft", 12345.5),
    ("12.345,50", 12345.5),
    ("12.345", 12345.0),
    ("1,234.50", 1234.5),
    ("12345,5", 12345.5),
    ("-1 200 Ft", -1200.0),
    ("Összesen: 45 720 Ft", 45720.0),
    ("HUF 999", 999.0),
    (1500, 1500.0),
    ("nincs", None),
    (None, None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw", ["2026. 11. 13.", "2026.11.13", "2026-11-13", "2026. november 13.", "13.11.2026",
                                 "2026. nov. 13.", "Budapest, 2026. 11. 13."])
def test_parse_date(raw):
    assert parse_date(raw) == date(2026, 11, 13)


def test_parse_date_invalid():
    assert parse_date("2026. 02. 31.") is None
    assert parse_date("valami") is None


def test_format_roundtrip():
    for style in range(4):
        d = date(2025, 3, 7)
        assert parse_date(format_date(d, style)) == d
    for v in (0, 15, 1234567, 99.5):
        assert parse_amount(format_amount(v, 2)) == pytest.approx(v)


def test_normalize_text():
    assert normalize_text("Árvíztűrő Tükörfúrógép!") == "arvizturotukorfurogep"


def test_parse_date_with_missing_separator_from_ocr():
    assert parse_date("1980. 01 07") == date(1980, 1, 7)
