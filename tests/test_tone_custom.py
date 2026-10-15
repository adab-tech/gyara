import pytest

from gyara import annotate_tones
from gyara.tone import melody_for


def test_falling_tone_gets_a_circumflex():
    assert annotate_tones("abada", lexicon={"abada": ["L", "L", "F"]}) == "àbàdâ"
    assert annotate_tones("kai", lexicon={"kai": ["F"]}) == "kâi"


def test_custom_lexicon_beats_the_builtin_list():
    assert annotate_tones("sannu") == "sánnù"  # built-in: H-L
    assert annotate_tones("sannu", lexicon={"sannu": ["L", "H"]}) == "sànnú"


def test_lookup_is_case_insensitive_and_keeps_punctuation():
    assert annotate_tones("Mutum!", lexicon={"mutum": ["L", "F"]}) == "Mùtûm!"


def test_melody_longer_than_the_word_keeps_its_last_tones():
    # "rana" has two syllables: a four-tone melody is right-aligned, so only H-L is used.
    lex = {"rana": ["H", "H", "H", "L"]}
    assert melody_for("rana", lexicon=lex) == ["H", "H", "H", "L"]
    assert annotate_tones("rana", lexicon=lex) == "ránà"


@pytest.mark.parametrize("bad", [[], ["X"], ["H", "low"]])
def test_invalid_melodies_are_rejected(bad):
    with pytest.raises(ValueError):
        annotate_tones("abada", lexicon={"abada": bad})


def test_words_missing_from_the_custom_lexicon_fall_back():
    assert annotate_tones("motoci", lexicon={"abada": ["L", "L", "F"]}) == "mótòcí"
    assert annotate_tones("zzzxqk", lexicon={"abada": ["L", "L", "F"]}) == "zzzxqk"
