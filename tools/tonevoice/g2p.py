"""Hausa orthography -> spellings that eSpeak's Swahili voice reads close to Hausa,
plus the bookkeeping needed to find each syllable's vowel in the synthesized audio.

eSpeak-ng has no Hausa voice. Swahili is the closest it ships (five pure vowels,
no stress contrasts we care about, similar consonants), so each syllable is respelled
the way Swahili spells the same sounds. Hooked consonants lose their hook
(b, d, k, y), c becomes ch, and a doubled vowel stands for a long vowel.
"""
import unicodedata

VOWELS = "aeiou"
_MAP = {"ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y", "c": "ch", "'": "", "’": "", "ʼ": ""}
GLOTTAL = ("'", "’", "ʼ")


def respell(syllable: str, long_vowel: bool = False) -> str:
    out = []
    for ch in syllable.lower():
        r = _MAP.get(ch, ch)
        if r in VOWELS and long_vowel:
            r = r * 2
        out.append(r)
    return "".join(out)


def has_glottal_onset(syllable: str) -> bool:
    return syllable[:1] in GLOTTAL


def vowel_letters(syllable: str) -> int:
    return sum(1 for ch in syllable.lower() if ch in VOWELS)


def has_coda(syllable: str) -> bool:
    s = syllable.lower().rstrip("'’ʼ")
    return bool(s) and s[-1] not in VOWELS


def marked_vowel_lengths(marked: str) -> list:
    """One boolean per vowel letter of a tone-marked word: does it carry a macron?"""
    nfd = unicodedata.normalize("NFD", marked)
    out = []
    i = 0
    while i < len(nfd):
        ch = nfd[i]
        if ch.lower() in VOWELS:
            j = i + 1
            marks = []
            while j < len(nfd) and unicodedata.combining(nfd[j]):
                marks.append(nfd[j])
                j += 1
            out.append("̄" in marks)
            i = j
        else:
            i += 1
    return out


def syllable_lengths(marked: str, sylls: list) -> list:
    """True for each syllable whose vowel is marked long in `marked`."""
    flags = marked_vowel_lengths(marked)
    counts = [vowel_letters(s) for s in sylls]
    if sum(counts) != len(flags):
        raise ValueError(f"vowel count mismatch for {marked!r} vs {sylls!r}: {sum(counts)} vs {len(flags)}")
    out, i = [], 0
    for c in counts:
        out.append(any(flags[i:i + c]))
        i += c
    return out
