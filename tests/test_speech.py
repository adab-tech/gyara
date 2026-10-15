import pytest

from gyara import prepare_for_speech, strip_english_glosses
from gyara.cli import main


def test_currency_before_number():
    assert prepare_for_speech("₦500") == "naira ɗari biyar"
    assert prepare_for_speech("$2") == "dala biyu"
    assert prepare_for_speech("€10 £3") == "yuro goma fam uku"


def test_dollar_never_survives():
    assert "$" not in prepare_for_speech("Cost $5 or $6")


def test_percent():
    assert prepare_for_speech("50%") == "hamsin cikin ɗari"
    assert prepare_for_speech("kashi 50 %") == "kashi hamsin cikin ɗari"


def test_slash_and_at():
    assert prepare_for_speech("maka/miki") == "maka ko miki"
    assert prepare_for_speech("me@x") == "me a x"


def test_markup_becomes_space_not_deleted():
    assert prepare_for_speech("**ruwa**\n# gida") == "ruwa gida"


def test_hooked_letters_and_digits_first():
    assert prepare_for_speech("K'asa ٢") == "Ƙasa biyu"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ruwa (water) yana da amfani", "ruwa yana da amfani"),
        ("ilimin halittu (Biology)", "ilimin halittu"),
        ("sai (misali) wani", "sai (misali) wani"),  # known Hausa aside kept
        ("abu (ƙanana)", "abu (ƙanana)"),  # hooked letter => Hausa
        ("abu (a very long english aside here)", "abu (a very long english aside here)"),
    ],
)
def test_glosses(text, expected):
    assert strip_english_glosses(text) == expected


def test_keep_glosses_flag():
    assert prepare_for_speech("ruwa (water)", strip_glosses=False) == "ruwa water"


def test_cli_speech(tmp_path, capsys):
    f = tmp_path / "t.txt"
    f.write_text("₦500 ne\n", encoding="utf-8")
    assert main(["speech", str(f)]) == 0
    assert capsys.readouterr().out == "naira ɗari biyar ne"
