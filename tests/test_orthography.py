import pytest

from shakara import normalize, normalize_digits


@pytest.mark.parametrize("raw,expected", [
    ("doki", "doki"),
    ("b'aki", "ɓaki"),
    ("d'an", "ɗan"),
    ("k'asa", "ƙasa"),
    ("y'anci", "ƴanci"),
    ("'yanci", "ƴanci"),
    ("gab'a", "gaɓa"),          # mid-word
    ("ts'", "ts"),
])
def test_lowercase(raw, expected):
    assert normalize(raw) == expected


def test_case_preserved():
    assert normalize("K'asa") == "Ƙasa"
    assert normalize("'Yanci") == "Ƴanci"
    assert normalize("B'aki D'an") == "Ɓaki Ɗan"
    assert normalize("TS'") == "TS"


def test_typographic_apostrophes():
    assert normalize("k’asa ʼyan") == "ƙasa ƴan"


def test_real_apostrophes_survive():
    assert normalize("al'ada a'a ɗan'uwa") == "al'ada a'a ɗan'uwa"


def test_digits():
    assert normalize_digits("٢٠١٥ ۱۲") == "2015 12"
    assert normalize("Zaben ٢٠١٥") == "Zaben 2015"
    assert normalize("٢", digits=False) == "٢"


def test_empty():
    assert normalize("") == ""
