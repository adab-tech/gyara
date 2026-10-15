"""Standard Hausa cardinal numbers and number spelling for speech."""

import re

_UNITS = ["", "ɗaya", "biyu", "uku", "huɗu", "biyar", "shida", "bakwai", "takwas", "tara"]
_TENS = {
    1: "goma", 2: "ashirin", 3: "talatin", 4: "arba'in", 5: "hamsin",
    6: "sittin", 7: "saba'in", 8: "tamanin", 9: "casa'in",
}


def _below_100(n: int) -> str:
    if n < 10:
        return _UNITS[n]
    if n < 20:
        r = n - 10
        return "goma" if r == 0 else f"goma sha {_UNITS[r]}"
    t, r = divmod(n, 10)
    return _TENS[t] if r == 0 else f"{_TENS[t]} da {_UNITS[r]}"


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    hundreds = "ɗari" if h == 1 else f"ɗari {_UNITS[h]}"
    return hundreds if r == 0 else f"{hundreds} da {_below_100(r)}"


def _digits(digits: str) -> str:
    return " ".join(_UNITS[int(d)] if d != "0" else "sifili" for d in digits)


def hausa_cardinal(n: int) -> str:
    """Non-negative integer -> Hausa words. >= 10^9 is read digit by digit.

    >>> hausa_cardinal(2026)
    'dubu biyu da ashirin da shida'
    """
    if n < 0:
        raise ValueError("hausa_cardinal expects a non-negative integer")
    if n == 0:
        return "sifili"
    if n >= 1_000_000_000:
        return _digits(str(n))
    parts = []
    if n >= 1_000_000:
        m, n = divmod(n, 1_000_000)
        parts.append("miliyan" if m == 1 else f"miliyan {_below_1000(m)}")
    if n >= 1000:
        th, n = divmod(n, 1000)
        parts.append("dubu" if th == 1 else f"dubu {_below_1000(th)}")
    if n > 0:
        parts.append(_below_1000(n))
    return " da ".join(parts)


def _token(token: str) -> str:
    if "." in token:
        whole, _, frac = token.partition(".")
        if not whole:
            words = "sifili"
        elif whole.startswith("0") and len(whole) > 1:
            words = _digits(whole)
        else:
            words = hausa_cardinal(int(whole))
        return f"{words} digo {_digits(frac)}"
    # Leading zero (phone numbers, codes) is read digit by digit.
    if token.startswith("0") and len(token) > 1:
        return _digits(token)
    return hausa_cardinal(int(token))


def spell_out_numbers(text: str) -> str:
    """Replace digit runs in `text` with Hausa words (thousands commas folded).

    >>> spell_out_numbers("Ina da 3.5 da 1,000")
    'Ina da uku digo biyar da dubu'
    """
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(
        r"(?<![A-Za-z\d])\d+(?:\.\d+)?(?![A-Za-z\d])",
        lambda m: _token(m.group(0)),
        text,
    )
