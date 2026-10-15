# shakara

Developer toolkit for **Hausa text** — extracted from the production code behind
[Murya](https://murya.ng) so anyone building Hausa NLP, TTS, or datasets can reuse it.
Pure Python, zero dependencies, Python 3.9+.

> `shakara` is a working name.

## What it does

| Module | Function |
|---|---|
| `orthography` | Restore hooked letters ɓ ɗ ƙ ƴ from `b' d' k' y' 'y` (case-preserving, handles ’ and ʼ); fold Arabic-Indic digits to 0-9 |
| `numbers` | Standard Hausa cardinals (`2026` → *dubu biyu da ashirin da shida*), decimals, phone numbers/codes digit-by-digit |
| `syllables` | Syllabify with light/heavy weight |
| `tone` | Heuristic tone marks: curated lexicon + plural-suffix melodies (unknown words are left unmarked unless `guess=True`) |
| `audit` | Orthography audit and corpus statistics (type/token ratio, duplicates, hook coverage) |

## Install

```bash
pip install -e '.[dev]'
```

## Python

```python
from shakara import normalize, spell_out_numbers, syllabify, annotate_tones, audit_text

normalize("K'asar Hausa da b'aki")      # 'Ƙasar Hausa da ɓaki'
spell_out_numbers("Shekara 2026")       # 'Shekara dubu biyu da ashirin da shida'
[s.text for s in syllabify("kwanduna")] # ['kwan', 'du', 'na']
annotate_tones("motoci")                # 'mótòcí'
audit_text("k'asa").ok                  # False
```

## CLI

```bash
echo "k'asa b'aki" | shakara fix        # ƙasa ɓaki
shakara audit corpus.txt                # exit 1 if problems -> usable in CI
shakara stats corpus.txt --json         # one sentence per line
shakara syllables kwanduna al\'ada
shakara tone < text.txt
shakara spell < text.txt
```

## Honest limits

- Hausa spelling does not mark tone, so `tone` is a **heuristic**, not a tagger.
- `ts'` is folded to `ts` (standard orthography writes the ejective without a hook).
- Apostrophe fixing is syntactic: a quoted `'yaro'` will be read as `'y…`. Audit before bulk-fixing.

## Develop

```bash
pytest
```
