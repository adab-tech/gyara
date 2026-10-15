from hypothesis import given
from hypothesis import strategies as st

from gyara import audit_text, normalize, prepare_for_speech, syllabify

TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=80)
HAUSA_ISH = st.text(alphabet="abdkyBDKYstTSaeiou '’ʼ.,!?0123456789", max_size=60)


@given(HAUSA_ISH)
def test_normalize_is_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once


@given(TEXT)
def test_text_without_apostrophes_or_foreign_digits_is_untouched(text):
    if not any(c in text for c in "'’ʼ") and normalize(text, digits=False) != text:
        raise AssertionError(repr(text))


@given(HAUSA_ISH)
def test_normalized_text_audits_clean(text):
    assert audit_text(normalize(text)).ok


@given(HAUSA_ISH)
def test_speech_prep_never_emits_unspeakable_symbols(text):
    out = prepare_for_speech(text)
    assert not any(c in out for c in '$@%*#_^~|<>[]{}()"')
    assert out == out.strip() and "  " not in out


@given(st.text(alphabet="abdfghjklmnrstwyzɓɗƙƴaeiou'", min_size=1, max_size=20))
def test_syllables_never_drop_letters(word):
    assert "".join(s.text for s in syllabify(word)) == word
