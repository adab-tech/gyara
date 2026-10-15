"""Hausa syllabification with syllable weight (light CV, heavy CVV / CVC)."""

import re
from dataclasses import dataclass
from typing import List

_WORD_CHARS = r"\wƁƊƘƳɓɗƙƴ'’ʼ"
_CLEAN_RE = re.compile(rf"[^{_WORD_CHARS}]")

_DIGRAPH = re.compile(r"(ts|sh|ch|gy|ky|ƙy|fy|gw|kw|ƙw|hw)")
_CONSONANT = re.compile(r"[ɓɗƙƴbdfghjklmnrstwyz'’ʼ]")
_LONG_VOWEL = re.compile(r"(aa|ee|ii|oo|uu|ai|au)")
_VOWEL = re.compile(r"[aeiou]")


@dataclass(frozen=True)
class Syllable:
    text: str
    weight: str  # "light" or "heavy"

    @property
    def heavy(self) -> bool:
        return self.weight == "heavy"


def _tokenize(word: str):
    low = word.lower()
    tokens = []  # (text, kind) kind in {"C", "VL", "VS"}
    i = 0
    while i < len(low):
        for pattern, kind in (
            (_DIGRAPH, "C"),
            (_CONSONANT, "C"),
            (_LONG_VOWEL, "VL"),
            (_VOWEL, "VS"),
        ):
            m = pattern.match(low, i)
            if m:
                tokens.append((word[i:m.end()], kind))
                i = m.end()
                break
        else:  # unknown symbol: treat as a consonant so nothing is dropped
            tokens.append((word[i], "C"))
            i += 1
    return tokens


def syllabify(word: str) -> List[Syllable]:
    """Split one Hausa word into syllables.

    >>> [s.text for s in syllabify("kwanduna")]
    ['kwan', 'du', 'na']
    >>> [s.weight for s in syllabify("kwanduna")]
    ['heavy', 'light', 'light']
    """
    clean = _CLEAN_RE.sub("", word)
    if not clean:
        return []
    tokens = _tokenize(clean)
    n = len(tokens)
    out: List[Syllable] = []
    j = 0
    while j < n:
        onset = nucleus = coda = ""
        long_vowel = False
        if tokens[j][1] == "C":
            onset = tokens[j][0]
            j += 1
        if j < n and tokens[j][1] in ("VS", "VL"):
            nucleus = tokens[j][0]
            long_vowel = tokens[j][1] == "VL"
            j += 1
        # A consonant closes this syllable if it ends the word or precedes another consonant.
        if j < n and tokens[j][1] == "C" and (j == n - 1 or tokens[j + 1][1] == "C"):
            coda = tokens[j][0]
            j += 1
        text = onset + nucleus + coda
        if text:
            out.append(Syllable(text, "heavy" if (long_vowel or coda) else "light"))
    return out
