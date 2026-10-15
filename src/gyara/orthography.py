"""Hausa orthography normalization.

Restores the hooked consonants (ɓ ɗ ƙ ƴ) from the ASCII apostrophe
spellings people actually type (b' d' k' y' 'y), preserving case, and folds
foreign digit glyphs to ASCII.
"""

import re

HOOKED_LOWER = {"b": "ɓ", "d": "ɗ", "k": "ƙ", "y": "ƴ"}
HOOKED_UPPER = {"B": "Ɓ", "D": "Ɗ", "K": "Ƙ", "Y": "Ƴ"}
HOOKED_LETTERS = "ɓɗƙƴƁƊƘƳ"

# ASCII apostrophe plus the typographic variants phones and word processors insert.
_APOS = "'’ʼ"

# Eastern Arabic-Indic (U+0660..) and Persian (U+06F0..) digits -> ASCII.
_FOREIGN_DIGITS = {
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
}
_FOREIGN_DIGIT_RE = re.compile("[" + "".join(_FOREIGN_DIGITS) + "]")

_LETTER = r"[^\W\d_]"  # any Unicode letter, including the hooked ones

# A hook is only restored when a letter follows the apostrophe: Hausa words do
# not end in ɓ ɗ ƙ ƴ, so "k'" at the end of a word is a closing quote.
#   b' d' k' y'  followed by a letter      ->  ɓ ɗ ƙ ƴ   (also mid-word: gab'a)
#   'y at the START of a word              ->  ƴ         ('yanci)
# 'y inside a word is a glottal stop plus y (ya'ya, "children") and is left alone,
# as is a quote-wrapped word like 'yaro' (see QUOTED_RE; `audit` warns about those).
_HOOK_RE = re.compile(
    rf"([bdkyBDKY])[{_APOS}](?={_LETTER})"
    rf"|(?<!{_LETTER})[{_APOS}]([yY])(?={_LETTER})(?!{_LETTER}*[{_APOS}](?!{_LETTER}))"
)
# ts' -> ts: the ejective is written plain "ts" in standard orthography.
_TS_RE = re.compile(rf"([tT][sS])[{_APOS}](?={_LETTER})")
# A word wrapped in single quotes that starts like a hooked word: 'yaro'
QUOTED_RE = re.compile(rf"(?<!{_LETTER})[{_APOS}][yY]{_LETTER}*[{_APOS}](?!{_LETTER})")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic / Persian digit glyphs to ASCII 0-9."""
    if not text:
        return text
    return _FOREIGN_DIGIT_RE.sub(lambda m: _FOREIGN_DIGITS[m.group(0)], text)


def _hook(match: "re.Match[str]") -> str:
    letter = match.group(1) or match.group(2)
    return HOOKED_UPPER[letter] if letter.isupper() else HOOKED_LOWER[letter.lower()]


def normalize(text: str, *, digits: bool = True) -> str:
    """Normalize ASCII-apostrophe Hausa spellings to standard Unicode.

    >>> normalize("k'asa da b'aki")
    'ƙasa da ɓaki'
    >>> normalize("'Yanci")
    'Ƴanci'
    """
    if not text:
        return text
    if digits:
        text = normalize_digits(text)
    text = _TS_RE.sub(lambda m: m.group(1), text)
    return _HOOK_RE.sub(_hook, text)


def count_hooked(text: str) -> int:
    """Number of proper hooked letters (ɓ ɗ ƙ ƴ, any case) in `text`."""
    return sum(text.count(c) for c in HOOKED_LETTERS)


def count_quoted_lookalikes(text: str) -> int:
    """Number of quote-wrapped words like 'yaro' that `normalize` deliberately leaves alone."""
    return len(QUOTED_RE.findall(text))


def count_ascii_hook_candidates(text: str) -> int:
    """Number of apostrophe spellings (b' d' k' y' 'y) that `normalize` would fix."""
    return len(_HOOK_RE.findall(text))
