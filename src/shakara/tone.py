"""Heuristic tone annotation for Hausa.

Standard Hausa spelling does not mark tone, so tone here is a *heuristic*:
a curated lexicon of common words, a few plural-suffix melodies, and an
optional low-confidence guess for everything else. Unknown words are left
unmarked unless ``guess=True``. Melodies are aligned to syllables
right-to-left; the leftmost tone spreads over any extra syllables.
"""

import re
import unicodedata
from typing import Dict, List, Optional

from .syllables import syllabify

ACUTE = "́"  # high
GRAVE = "̀"  # low

LEXICON: Dict[str, List[str]] = {
    # greetings & courtesy
    "sannu": ["H", "L"], "sannun": ["H", "L"], "barka": ["H", "H"], "barkan": ["H", "H"],
    "lafiya": ["H", "H", "L"], "lafiyayye": ["H", "H", "L", "H"], "madalla": ["L", "H", "L"],
    "godiya": ["L", "H", "L"], "gode": ["H", "H"],
    # pronouns & particles
    "na": ["L"], "naa": ["H"], "ni": ["H"], "kai": ["H"], "ke": ["H"], "shi": ["H"],
    "ita": ["H", "L"], "mu": ["H"], "ku": ["H"], "su": ["H"], "mun": ["H"], "kun": ["H"],
    "sun": ["H"], "ya": ["L"], "ta": ["L"], "kuma": ["H", "L"], "don": ["H"], "da": ["L"],
    "a": ["L"], "ne": ["H"], "ce": ["H"], "ko": ["H"], "mai": ["H"], "fi": ["H"],
    # time & questions
    "yanzu": ["H", "L"], "yau": ["H"], "gobe": ["H", "L"], "dare": ["L", "H"], "rana": ["H", "L"],
    "safe": ["L", "H"], "yaushe": ["H", "L"], "ina": ["L", "H"], "yaya": ["H", "L"],
    "mene": ["H", "L"], "menene": ["H", "L", "L"],
    # numbers
    "ɗan": ["H"], "ɗaya": ["L", "H"], "biyu": ["H", "L"], "uku": ["H", "L"], "huɗu": ["H", "L"],
    "biyar": ["L", "H"], "shida": ["H", "L"], "bakwai": ["H", "H"], "takwas": ["L", "H"],
    "tara": ["H", "L"], "goma": ["H", "L"], "ɗari": ["H", "L"], "dubu": ["H", "L"],
    # nouns
    "doki": ["H", "L"], "ƙasa": ["L", "H"], "ƙasar": ["L", "H"], "girma": ["H", "L"],
    "kunya": ["H", "L"], "al'ada": ["L", "H", "L"], "mutunci": ["L", "H", "L"],
    "yaro": ["H", "L"], "yara": ["H", "L"], "mace": ["H", "L"], "namiji": ["L", "H", "L"],
    "mutum": ["L", "H"], "mutane": ["H", "H", "L"], "abinci": ["L", "H", "L"], "ruwa": ["L", "H"],
    "wuta": ["L", "H"], "gida": ["L", "H"], "hanya": ["H", "L"], "aiki": ["H", "L"],
    "baba": ["H", "L"], "mama": ["H", "L"], "daji": ["H", "L"], "kogi": ["H", "L"],
    "tsauni": ["H", "L"], "iska": ["L", "H"], "ƙwai": ["H"], "hausa": ["H", "H"],
    "lafiyar": ["L", "L", "H"], "matuƙar": ["L", "H", "L"], "muhimmanci": ["L", "H", "H", "L"],
    "fada": ["H", "L"],   # palace
    "fadaa": ["L", "H"],  # fighting / speaking
    # verbs
    "yi": ["H"], "yiwuwa": ["L", "H", "L"], "samu": ["H", "L"], "tafi": ["H", "L"], "zo": ["H"],
    "ci": ["H"], "sha": ["H"], "gani": ["H", "L"], "ji": ["H"], "laɓewa": ["L", "L", "H"],
    "cikin": ["L", "H"], "ɗumi": ["H", "L"],
}

_PLURAL_SUFFIXES = (
    (("una",), ["H", "L", "L"]),
    (("oshi", "oshe"), ["H", "L", "H"]),
    (("aye", "ayi"), ["L", "H", "L"]),
    (("oci", "otsi"), ["H", "L", "H"]),
)
_AI_PLURAL = re.compile(r"[bdfghjklmnrstwyzɓɗƙƴ']ai$")
_LETTERS = r"\wƁƊƘƳɓɗƙƴ'’ʼ"
_WORD_RE = re.compile(rf"^([^{_LETTERS}]*)([{_LETTERS}]+)([^{_LETTERS}]*)$")


def melody_for(word: str, *, guess: bool = False) -> Optional[List[str]]:
    """Tone melody (list of "H"/"L") for a word, or None if unknown."""
    low = word.lower()
    if low in LEXICON:
        return LEXICON[low]
    for suffixes, melody in _PLURAL_SUFFIXES:
        if low.endswith(suffixes):
            return melody
    if _AI_PLURAL.search(low):
        return ["L", "H", "H"]
    if guess:
        sylls = syllabify(low)
        if sylls:
            return ["L", "H"] if sylls[-1].heavy else ["H", "L"]
    return None


def _align(melody: List[str], n: int) -> List[str]:
    """Right-align `melody` to n syllables, spreading the leftmost tone."""
    if n <= len(melody):
        return melody[len(melody) - n:]
    return [melody[0]] * (n - len(melody)) + melody


def _mark(syllable: str, tone: str) -> str:
    m = re.search(r"[aeiouAEIOU]", syllable)
    if not m:
        return syllable
    return syllable[: m.end()] + (ACUTE if tone == "H" else GRAVE) + syllable[m.end():]


def annotate_tones(text: str, *, guess: bool = False) -> str:
    """Mark known words with acute (H) / grave (L) accents on each syllable.

    >>> annotate_tones("kwanduna motoci")
    'kwándùnà mótòcí'
    """
    out = []
    for token in text.split():
        m = _WORD_RE.match(token)
        if not m:
            out.append(token)
            continue
        left, word, right = m.groups()
        sylls = syllabify(word)
        melody = melody_for(word, guess=guess)
        if not sylls or melody is None:
            out.append(token)
            continue
        tones = _align(melody, len(sylls))
        marked = "".join(_mark(s.text, t) for s, t in zip(sylls, tones))
        out.append(left + unicodedata.normalize("NFC", marked) + right)
    return " ".join(out)
