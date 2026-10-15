import pytest

from gyara import normalize, normalize_digits


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("doki", "doki"),
        ("b'aki", "ɓaki"),
        ("d'an", "ɗan"),
        ("k'asa", "ƙasa"),
        ("y'anci", "ƴanci"),
        ("'yanci", "ƴanci"),
        ("gab'a", "gaɓa"),  # mid-word
        ("ts'a", "tsa"),
    ],
)
def test_lowercase(raw, expected):
    assert normalize(raw) == expected


def test_case_preserved():
    assert normalize("K'asa") == "Ƙasa"
    assert normalize("'Yanci") == "Ƴanci"
    assert normalize("B'aki D'an") == "Ɓaki Ɗan"
    assert normalize("TS'a") == "TSa"


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


@pytest.mark.parametrize("word", ["ya'ya", "ya'yan", "ba'a", "ma'ana", "mu'amala"])
def test_glottal_stop_before_y_is_not_a_hook(word):
    # "ya'ya" (children) used to become "yaƴa".
    assert normalize(word) == word


@pytest.mark.parametrize("text", ["'yaro'", "she said 'k'", "'b' da 'd'", "the 'ts'"])
def test_quotes_and_closing_apostrophes_survive(text):
    assert normalize(text) == text


def test_word_initial_y_still_works_next_to_text():
    assert normalize("Ina son 'yanci, 'yan uwa") == "Ina son ƴanci, ƴan uwa"
    assert normalize("('yan)") == "(ƴan)"


def test_hook_inside_quoted_phrase_still_fixed():
    assert normalize("'k'asa'") == "'ƙasa'"
