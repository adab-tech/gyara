# gyara

Developer toolkit for **Hausa text** — extracted from the production code behind
[Murya](https://murya.ng) so anyone building Hausa NLP, TTS, or datasets can reuse it.
Pure Python, zero dependencies, Python 3.9+.

> Gyara (Hausa: "to fix, to correct").

## What it does

| Module | Function |
|---|---|
| `orthography` | Restore hooked letters ɓ ɗ ƙ ƴ from `b' d' k' y' 'y` (case-preserving, handles ’ and ʼ); fold Arabic-Indic digits to 0-9 |
| `numbers` | Standard Hausa cardinals (`2026` → *dubu biyu da ashirin da shida*), decimals, phone numbers/codes digit-by-digit |
| `speech` | TTS front end: ₦ $ € £ and % to words, `a/b` to *a ko b*, drop English glosses, strip markup |
| `syllables` | Syllabify with light/heavy weight |
| `tone` | Heuristic tone marks: curated lexicon + plural-suffix melodies (unknown words are left unmarked unless `guess=True`) |
| `audit` | Orthography audit and corpus statistics (type/token ratio, duplicates, hook coverage) |

## Install

```bash
pip install -e '.[dev]'
```

## Python

```python
from gyara import normalize, spell_out_numbers, syllabify, annotate_tones, audit_text

normalize("K'asar Hausa da b'aki")      # 'Ƙasar Hausa da ɓaki'
spell_out_numbers("Shekara 2026")       # 'Shekara dubu biyu da ashirin da shida'
[s.text for s in syllabify("kwanduna")] # ['kwan', 'du', 'na']
annotate_tones("motoci")                # 'mótòcí'
audit_text("k'asa").ok                  # False
```

## CLI

```bash
echo "k'asa b'aki" | gyara fix        # ƙasa ɓaki
gyara audit corpus.txt                # exit 1 if problems -> usable in CI
gyara stats corpus.txt --json         # one sentence per line
gyara syllables kwanduna al\'ada
gyara tone < text.txt
gyara spell < text.txt
gyara speech < reply.txt            # ready for a TTS voice
```

## Honest limits

- Hausa spelling does not mark tone, so `tone` is a **heuristic**, not a tagger.
- `ts'` is folded to `ts` (standard orthography writes the ejective without a hook).
- Apostrophe fixing is conservative: a hook is restored only before a letter, `'y` only at the start of a word, and quote-wrapped words like `'yaro'` are left alone (`audit` warns). `ya'ya` (children) keeps its glottal stop.

## Develop

```bash
pytest
```
