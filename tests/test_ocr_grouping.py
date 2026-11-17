from dokulanc.ocr import Segment, group_lines


def test_segments_on_same_row_are_merged():
    segs = [
        Segment("Nettó", 10, 100, 80, 120, 0.9),
        Segment("összesen:", 85, 101, 190, 121, 0.8),
        Segment("12 000 Ft", 600, 99, 700, 119, 0.95),
        Segment("Számla", 10, 10, 120, 40, 0.99),
    ]
    lines = group_lines(segs)
    assert [l.text for l in lines] == ["Számla", "Nettó összesen: | 12 000 Ft"]
    assert [l.id for l in lines] == [1, 2]
    assert lines[1].bbox == (10, 99, 700, 121)
    assert 0.8 < lines[1].confidence < 0.95


def test_separate_rows_stay_separate():
    segs = [Segment("A", 10, 10, 30, 30, 1.0), Segment("B", 10, 40, 30, 60, 1.0), Segment("C", 10, 70, 30, 90, 1.0)]
    assert [l.text for l in group_lines(segs)] == ["A", "B", "C"]


def test_empty_input():
    assert group_lines([]) == []
    assert group_lines([Segment("  ", 0, 0, 1, 1, 1.0)]) == []


def test_sidebar_layout_is_read_column_by_column():
    left = [Segment(t, 10, 20 + 40 * i, 150, 40 + 40 * i, 1.0) for i, t in enumerate(["E-mail", "a@b.hu", "Telefon", "+36 1", "Lakcím"])]
    right = [Segment(t, 300, 35 + 55 * i, 700, 55 + 55 * i, 1.0) for i, t in enumerate(["Kiss Anna", "Önéletrajz", "Tapasztalat", "2020 – 2022", "fejlesztő"])]
    texts = [l.text for l in group_lines(left + right)]
    assert texts == ["E-mail", "a@b.hu", "Telefon", "+36 1", "Lakcím", "Kiss Anna", "Önéletrajz", "Tapasztalat", "2020 – 2022", "fejlesztő"]


def test_table_rows_are_not_split_into_columns():
    names = [Segment(t, 10, 20 + 30 * i, 200, 40 + 30 * i, 1.0) for i, t in enumerate(["TEJ", "KENYER", "VAJ", "ALMA", "SAJT"])]
    prices = [Segment(t, 400, 20 + 30 * i, 460, 40 + 30 * i, 1.0) for i, t in enumerate(["329", "549", "899", "499", "1899"])]
    texts = [l.text for l in group_lines(names + prices)]
    assert texts[0] == "TEJ | 329" and len(texts) == 5
