"""Text preparation for Hausa speech synthesis.

A TTS voice trained on Hausa usually has a small symbol inventory (letters,
punctuation, ɓ ɗ ƙ ƴ). Anything outside it is silently dropped, which loses
meaning ("₦500" read as just "ɗari biyar"), glues words together
("maka/miki" -> "makamiki") or, worse, hits control symbols ("$" can be the
end-of-sequence token). `prepare_for_speech` rewrites text so it can be
spoken as written.
"""

import re

from .numbers import spell_out_numbers
from .orthography import HOOKED_LETTERS, normalize

# Currency-first matches Hausa word order: "naira dubu goma" = ten thousand naira.
CURRENCY_WORDS = {"₦": " naira ", "$": " dala ", "€": " yuro ", "£": " fam "}

# Short ASCII parentheticals in a Hausa text are usually English glosses
# ("ruwa (water)"). Keep them on screen, drop them from speech.
_PAREN_RE = re.compile(r"\s*\(([^()]{1,40})\)")
_ASCII_GLOSS_RE = re.compile(r"[A-Za-z][A-Za-z /'-]*$")
_HAUSA_ASIDES = {
    "misali",
    "wato",
    "watau",
    "kamar",
    "kuma",
    "gani",
    "duba",
    "watakila",
    "wajen",
    "ma",
    "dss",
    "sauransu",
    "haka",
    "shi",
    "ita",
}


def strip_english_glosses(text: str) -> str:
    """Remove short English-gloss parentheticals; keep Hausa asides.

    >>> strip_english_glosses("ruwa (water) yana da amfani")
    'ruwa yana da amfani'
    """

    def repl(m: "re.Match[str]") -> str:
        inner = m.group(1).strip()
        words = inner.split()
        if any(c in inner for c in HOOKED_LETTERS):
            return m.group(0)
        if len(words) > 3:
            return m.group(0)
        if any(w.lower().strip(".,") in _HAUSA_ASIDES for w in words):
            return m.group(0)
        return "" if _ASCII_GLOSS_RE.match(inner) else m.group(0)

    return _PAREN_RE.sub(repl, text)


def prepare_for_speech(text: str, *, strip_glosses: bool = True) -> str:
    """Full normalization pipeline for a TTS front end.

    Order matters: hooked letters, glosses, currency, percent, numbers,
    separators, markup, whitespace.

    >>> prepare_for_speech("Farashin ₦500 ne, kashi 50%")
    'Farashin naira ɗari biyar ne, kashi hamsin cikin ɗari'
    """
    text = normalize(text)
    if strip_glosses:
        text = strip_english_glosses(text)
    for symbol, word in CURRENCY_WORDS.items():
        text = text.replace(symbol, word)
    text = re.sub(r"(?<=\d)\s*%", " cikin ɗari", text)
    text = spell_out_numbers(text)
    text = re.sub(r"(?<=\w)/(?=\w)", " ko ", text)  # maka/miki -> maka ko miki
    text = text.replace("@", " a ")
    text = re.sub(r"[*#_^~|<>\[\]{}()\"“”„]", " ", text)  # markup -> space, never deleted
    return re.sub(r"\s+", " ", text).strip()
