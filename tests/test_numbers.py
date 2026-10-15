import pytest

from shakara import hausa_cardinal, spell_out_numbers


@pytest.mark.parametrize("n,words", [
    (0, "sifili"), (1, "ɗaya"), (10, "goma"), (11, "goma sha ɗaya"),
    (21, "ashirin da ɗaya"), (100, "ɗari"), (101, "ɗari da ɗaya"),
    (250, "ɗari biyu da hamsin"), (1000, "dubu"),
    (2026, "dubu biyu da ashirin da shida"),
    (1_000_000, "miliyan"),
])
def test_cardinals(n, words):
    assert hausa_cardinal(n) == words


def test_billions_digit_by_digit():
    assert hausa_cardinal(1_000_000_000).startswith("ɗaya sifili")


def test_negative():
    with pytest.raises(ValueError):
        hausa_cardinal(-1)


def test_spell_text():
    assert spell_out_numbers("Ina da 3.5") == "Ina da uku digo biyar"
    assert spell_out_numbers("1,000") == "dubu"
    assert spell_out_numbers("0500") == "sifili biyar sifili sifili"   # code, not 500
    assert spell_out_numbers("08012") == "sifili takwas sifili ɗaya biyu"
    assert spell_out_numbers("a1b") == "a1b"                           # identifiers untouched
