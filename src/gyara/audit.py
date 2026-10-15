"""Text audit and dataset statistics for Hausa corpora."""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List

from .orthography import (
    _FOREIGN_DIGIT_RE,
    count_ascii_hook_candidates,
    count_hooked,
)

_WORD_RE = re.compile(r"[\wƁƊƘƳɓɗƙƴ]+(?:['’ʼ][\wƁƊƘƳɓɗƙƴ]+)*")


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class AuditReport:
    words: int
    hooked_letters: int
    ascii_hook_spellings: int
    foreign_digits: int
    ascii_digit_runs: int
    issues: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def audit_text(text: str) -> AuditReport:
    """Check `text` for orthography problems `gyara fix` can repair."""
    ascii_hooks = count_ascii_hook_candidates(text)
    foreign = len(_FOREIGN_DIGIT_RE.findall(text))
    issues = []
    if ascii_hooks:
        issues.append(f"{ascii_hooks} ASCII-apostrophe hooked-letter spelling(s) (b' d' k' y' 'y)")
    if foreign:
        issues.append(f"{foreign} Arabic-Indic/Persian digit glyph(s); Hausa uses 0-9")
    return AuditReport(
        words=len(words(text)),
        hooked_letters=count_hooked(text),
        ascii_hook_spellings=ascii_hooks,
        foreign_digits=foreign,
        ascii_digit_runs=len(re.findall(r"\d+", text)),
        issues=issues,
    )


def dataset_stats(lines: Iterable[str]) -> Dict[str, Any]:
    """Summary statistics for a corpus with one sentence per line."""
    total = empty = with_digits = ascii_hooks = hooked = 0
    lengths: List[int] = []
    tokens = 0
    types = set()
    seen = set()
    duplicates = 0
    for raw in lines:
        total += 1
        line = raw.strip()
        if not line:
            empty += 1
            continue
        if line in seen:
            duplicates += 1
        seen.add(line)
        ws = words(line)
        lengths.append(len(ws))
        tokens += len(ws)
        types.update(ws)
        ascii_hooks += count_ascii_hook_candidates(line)
        hooked += count_hooked(line)
        if re.search(r"\d", line):
            with_digits += 1
    return {
        "lines": total,
        "empty_lines": empty,
        "duplicate_lines": duplicates,
        "lines_with_digits": with_digits,
        "tokens": tokens,
        "types": len(types),
        "type_token_ratio": round(len(types) / tokens, 4) if tokens else 0.0,
        "words_per_line": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        },
        "hooked_letters": hooked,
        "ascii_hook_spellings": ascii_hooks,
    }
