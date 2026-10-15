import pytest

from shakara import annotate_tones, syllabify


def texts(word):
    return [s.text for s in syllabify(word)]


def test_syllables_and_weight():
    assert texts("kwanduna") == ["kwan", "du", "na"]
    assert [s.weight for s in syllabify("kwanduna")] == ["heavy", "light", "light"]
    assert texts("al'ada") == ["al", "'a", "da"]
    assert [s.weight for s in syllabify("daa")] == ["heavy"]


def test_empty_and_punctuation():
    assert syllabify("") == []
    assert texts("sannu,") == ["san", "nu"]


@pytest.mark.parametrize(
    "word,expected",
    [
        ("kwanduna", "kwándùnà"),
        ("motoci", "mótòcí"),
        ("yatsotsi", "yátsòtsí"),
        ("gidaye", "gìdáyè"),
        ("dawakai", "dàwákái"),
        ("litattafai", "lìtàttáfái"),
    ],
)
def test_plural_rules(word, expected):
    assert annotate_tones(word) == expected


def test_lexicon_and_punctuation():
    assert annotate_tones("Sannu!") == "Sánnù!"
    assert annotate_tones("sannu") == "sánnù"


def test_unknown_words_left_alone_by_default():
    assert annotate_tones("zzzxqk") == "zzzxqk"
    assert annotate_tones("tabbata") == "tabbata"
    assert annotate_tones("tabbata", guess=True) != "tabbata"
