"""shakara: developer toolkit for Hausa text."""

from .audit import AuditReport, audit_text, dataset_stats
from .numbers import hausa_cardinal, spell_out_numbers
from .orthography import normalize, normalize_digits
from .syllables import Syllable, syllabify
from .tone import annotate_tones

__version__ = "0.1.0"

__all__ = [
    "AuditReport",
    "Syllable",
    "annotate_tones",
    "audit_text",
    "dataset_stats",
    "hausa_cardinal",
    "normalize",
    "normalize_digits",
    "spell_out_numbers",
    "syllabify",
]
