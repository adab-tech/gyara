"""Build a tone-marked Hausa lexicon from Newman (1977) and Wiktionary, Newman first.

    python scripts/build_tone_lexicon.py \
        --newman path/to/ha_en_pairs_merged.jsonl \
        --wiktionary path/to/kaikki-Hausa.jsonl \
        [--corpus path/to/articles.jsonl]            # Hausa text, for frequency-weighted review samples
        --out-dir build/tone

Source policy (set by the project owner): Newman's tone marks are the authority. Where both
sources have a word, Newman's pattern is used and Wiktionary's differing pattern is kept only as
`alt_wiktionary`. A word found only in Wiktionary is kept but labelled `wiktionary_only` (lower
confidence: it agrees with Newman on only about two thirds of shared words).

Confidence tiers on every record:
    both             Newman and Wiktionary give the same pattern
    newman           Newman only, or Newman chosen over a differing Wiktionary pattern
    newman_bare      a Newman headword with NO tone mark, read as all-high (unmarked = high).
                     This holds for core words (wani H-H, shi H, yi H, sun H, kuma H-H) but is not
                     yet confirmed for loanwords, so it stays out of the trusted runtime mapping
    wiktionary_only  the word is not in Newman at all

Outputs (in --out-dir):
    lexicon.jsonl             merged records, one per (word, form kind, tone pattern, sense)
    lexicon-newman.jsonl      only the records that come from Newman
    lexicon-wiktionary.jsonl  only the records that come from Wiktionary
    runtime-trusted.json      word -> melody for unambiguous words in the `both` and `newman` tiers
    runtime-with-bare.json    the same plus `newman_bare` (inferred all-high) words
    runtime-all.json          the same plus `wiktionary_only` words
    report.json               counts per tier, agreement, coverage of the corpus
    review.md / review.csv    a frequency-weighted sample for a native speaker (--sample N)

Notation (the same in both sources):
    grave L low, circumflex F falling, acute H high (rare), NO mark = high.
    Wiktionary also uses a macron for vowel LENGTH only. A tilde on r (r̃), cedilla, dot-below and
    breve are consonant/vowel-quality marks and are ignored.
A Newman headword with no tone mark at all is read as all-high, in its own tier (see above).
Caution: where Wiktionary also marks such a word it is all-high only 38% of the time, so either
the convention does not hold for every entry or Murya's page-image transcription dropped marks.

Wiktionary is CC-BY-SA 4.0 (credit Wiktionary contributors); Newman & Newman (1977) is used with
the permission of Prof. Paul Newman as reported by the project owner. Settle the licence of any
published data before release; the per-source output files exist so the two can be separated.
"""

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from gyara.syllables import syllabify
from gyara.tone import LEXICON as GYARA_LEXICON

GRAVE, ACUTE, CIRCUMFLEX, MACRON = "̀", "́", "̂", "̄"
TONE_MARKS = {GRAVE: "L", ACUTE: "H", CIRCUMFLEX: "F"}
VOWELS = "aeiouAEIOU"
# Letters a Latin-script Hausa word may contain (hooked letters included).
LATIN_OK = set("abcdefghijklmnopqrstuvwxyz'’ʼɓɗƙƴ")

NEWMAN, WIKTIONARY = "newman_1977", "wiktionary_ha_ccbysa"
TRUSTED_TIERS = ("both", "newman")
BARE_TIERS = TRUSTED_TIERS + ("newman_bare",)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def plain(s: str) -> str:
    """Lowercase, marks removed (hooked letters are precomposed and survive)."""
    d = unicodedata.normalize("NFD", s)
    return nfc("".join(c for c in d if unicodedata.category(c) != "Mn")).lower()


def parse_marked(form: str) -> Optional[Tuple[List[str], bool]]:
    """Per-syllable tones of a marked form, or None if it cannot be parsed.

    Returns (tones, transcribed) where tones use "H" / "L" / "F". `transcribed` says the form
    carries at least one tone mark OR length mark, i.e. someone deliberately wrote down its
    pronunciation. A bare spelling (no marks at all) is not transcribed: every H in it would
    be a guess.
    """
    base = plain(form)
    if not base or any(c not in LATIN_OK for c in base):
        return None
    # (base char, set of tone marks) per character, from the decomposed form
    clusters: List[Tuple[str, set]] = []
    transcribed = False
    for c in unicodedata.normalize("NFD", form.lower()):
        if unicodedata.category(c) == "Mn":
            if c == MACRON:
                transcribed = True  # length only: does not change the tone
            if clusters and c in TONE_MARKS:
                clusters[-1][1].add(TONE_MARKS[c])
                transcribed = True
        else:
            clusters.append((c, set()))
    chars = "".join(c for c, _ in clusters)
    if nfc(chars) != base:
        return None
    sylls = syllabify(base)
    if "".join(s.text for s in sylls) != base:
        return None
    tones: List[str] = []
    pos = 0
    for s in sylls:
        marks: set = set()
        has_vowel = False
        for _ in s.text:
            ch, ms = clusters[pos]
            pos += 1
            if ch in VOWELS:
                has_vowel = True
                marks |= ms
        if not has_vowel:
            return None
        if len(marks) > 1:
            return None  # conflicting marks inside one syllable
        tones.append(next(iter(marks)) if marks else "H")
    return tones, transcribed


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def first_gloss(entry: dict) -> str:
    for sense in entry.get("senses") or []:
        for g in sense.get("glosses") or []:
            return g[:80]
    return ""


# ---------------------------------------------------------------------------------------------
# Wiktionary
# ---------------------------------------------------------------------------------------------
def build_from_wiktionary(entries: Iterable[dict]) -> Tuple[List[dict], Counter]:
    """One record per (word, form kind, tone pattern, gloss)."""
    stats: Counter = Counter()
    seen = set()
    out: List[dict] = []
    for e in entries:
        stats["entries"] += 1
        head = e.get("word", "")
        for f in e.get("forms") or []:
            form = (f.get("form") or "").strip()
            tags = f.get("tags") or []
            if not form or " " in form or "-" in form or "romanization" in tags:
                stats["skipped_multiword_or_romanization"] += 1
                continue
            parsed = parse_marked(form)
            if parsed is None:
                stats["skipped_unparseable"] += 1
                continue
            tones, explicit = parsed
            kind = (
                "canonical"
                if "canonical" in tags
                else ("+".join(t for t in tags if t != "canonical") or "other")
            )
            if kind == "canonical" and plain(form) != plain(head):
                stats["skipped_form_differs_from_headword"] += 1
                continue
            key = plain(form)
            ident = (key, kind, tuple(tones), e.get("pos"), first_gloss(e))
            if ident in seen:
                continue
            seen.add(ident)
            out.append(
                {
                    "word": key,
                    "tones": tones,
                    "marked": nfc(form.lower()),
                    "kind": kind,
                    "pos": e.get("pos"),
                    "lemma": head,
                    "gloss": first_gloss(e),
                    "explicit": explicit,
                    "source": WIKTIONARY,
                }
            )
    stats["records"] = len(out)
    return out, stats


# ---------------------------------------------------------------------------------------------
# Newman
# ---------------------------------------------------------------------------------------------
_FORM_RE = re.compile(r"\((pl|vn)\.\s*([^)]*)\)")


def forms_from_context(context: str) -> List[Tuple[str, str]]:
    """Plural and verbal-noun forms written in a Newman `context`, e.g. 'n.m. (pl. zùgàzùgai)'."""
    out: List[Tuple[str, str]] = []
    for kind, body in _FORM_RE.findall(context or ""):
        for tok in re.split(r"[,;]|\bor\b", body):
            tok = tok.strip()
            if tok:
                out.append(("plural" if kind == "pl" else "verbal-noun", tok))
    return out


def build_from_newman(rows: Iterable[dict]) -> Tuple[List[dict], Counter]:
    """Single-word headwords plus the plurals / verbal nouns they list.

    A form with a tone mark gives an `explicit` record. A form with none gives an `inferred`
    all-high record (unmarked = high), which the merge puts in the `newman_bare` tier.
    """
    stats: Counter = Counter()
    out: List[dict] = []
    seen = set()

    def add(word_form: str, kind: str, row: dict, lemma: str) -> None:
        parsed = parse_marked(word_form)
        if parsed is None:
            stats["skipped_unparseable"] += 1
            return
        tones, transcribed = parsed
        key = plain(word_form)
        pos = re.sub(r"\s*\(.*", "", row.get("context") or "").strip()
        ident = (key, kind, tuple(tones), pos, row.get("source_en"))
        if ident in seen:
            return
        seen.add(ident)
        out.append(
            {
                "word": key,
                "tones": tones,
                "marked": nfc(word_form.lower()),
                "kind": kind,
                "pos": pos,
                "lemma": lemma,
                "gloss": (row.get("source_en") or "")[:80],
                "explicit": transcribed,
                "inferred": not transcribed,
                "source": NEWMAN,
            }
        )
        stats[f"records_{kind}" if transcribed else f"records_{kind}_bare_inferred_high"] += 1

    for row in rows:
        stats["rows"] += 1
        head = (row.get("target_ha") or "").strip()
        if head and " " not in head and "-" not in head:
            add(head, "canonical", row, head)
        else:
            stats["skipped_multiword_headword"] += 1
        for kind, tok in forms_from_context(row.get("context") or ""):
            if " " in tok or "-" in tok:
                continue
            add(tok, kind, row, head)
    stats["records"] = len(out)
    return out, stats


# ---------------------------------------------------------------------------------------------
# Merge: Newman first
# ---------------------------------------------------------------------------------------------
def merge(newman: Iterable[dict], wiktionary: Iterable[dict]) -> List[dict]:
    """Newman's pattern wins; a Wiktionary word absent from Newman is kept as lower confidence."""
    n_by: Dict[str, List[dict]] = defaultdict(list)
    w_by: Dict[str, List[dict]] = defaultdict(list)
    for r in newman:
        n_by[r["word"]].append(r)
    for r in wiktionary:
        if r["explicit"]:
            w_by[r["word"]].append(r)
    out: List[dict] = []
    for word in sorted(set(n_by) | set(w_by)):
        if word in n_by:
            w_patterns = {tuple(r["tones"]) for r in w_by.get(word, [])}
            for r in n_by[word]:
                rec = dict(r)
                rec["sources"] = [NEWMAN]
                if tuple(r["tones"]) in w_patterns:
                    rec["confidence"] = "both"
                    rec["sources"].append(WIKTIONARY)
                else:
                    rec["confidence"] = "newman_bare" if r.get("inferred") else "newman"
                    if w_patterns:
                        rec["alt_wiktionary"] = sorted([list(p) for p in w_patterns])
                out.append(rec)
        else:
            for r in w_by[word]:
                rec = dict(r)
                rec["sources"] = [WIKTIONARY]
                rec["confidence"] = "wiktionary_only"
                out.append(rec)
    return out


def runtime_mapping(records: Iterable[dict], tiers: Iterable[str]) -> Dict[str, List[str]]:
    """word -> melody, only for words whose records in `tiers` all agree (ambiguous words are
    left out: precision over recall)."""
    wanted = set(tiers)
    by: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    for r in records:
        if r["confidence"] in wanted:
            by[r["word"]].add(tuple(r["tones"]))
    return {w: list(next(iter(p))) for w, p in sorted(by.items()) if len(p) == 1}


# ---------------------------------------------------------------------------------------------
# Corpus frequency
# ---------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def corpus_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("content") or ""
            text = re.sub(r"https?://\S+|\[\[.*?\]\]|\{\{.*?\}\}", " ", text)
            counts.update(plain(w) for w in _TOKEN_RE.findall(text))
    return counts


def coverage(mapping: Dict[str, List[str]], counts: Counter) -> float:
    total = sum(counts.values())
    return sum(c for w, c in counts.items() if w in mapping) / total if total else 0.0


# ---------------------------------------------------------------------------------------------
# Review sample
# ---------------------------------------------------------------------------------------------
def fmt(tones: Iterable[str]) -> str:
    return "-".join(tones)


def review_sample(records: List[dict], counts: Counter, n: int, seed: int) -> List[dict]:
    """A frequency-weighted sample covering the cases a native speaker most needs to judge."""
    rng = random.Random(seed)
    usable = [
        r
        for r in records
        if r["kind"] == "canonical"
        and " " not in r["marked"]
        and not (r.get("gloss") or "")
        .lower()
        .startswith(("alternative form", "alternative spelling"))
        and r.get("pos") != "name"
    ]
    by_word: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    for r in records:
        by_word[r["word"]].add(tuple(r["tones"]))

    def freq(r: dict) -> int:
        return counts.get(r["word"], 0)

    def top(rows: List[dict], k: int) -> List[dict]:
        return sorted(rows, key=lambda r: (-freq(r), r["word"]))[:k]

    pools: Dict[str, List[dict]] = {
        "frequent word (Newman or both)": top(
            [r for r in usable if r["confidence"] in TRUSTED_TIERS], 500
        ),
        "Wiktionary only": top([r for r in usable if r["confidence"] == "wiktionary_only"], 400),
        "Newman differs from Wiktionary": top([r for r in usable if r.get("alt_wiktionary")], 300)
        or [r for r in usable if r.get("alt_wiktionary")],
        "bare Newman entry (read as all-high)": [
            r for r in usable if r["confidence"] == "newman_bare"
        ],
        "falling tone (Newman)": [r for r in usable if "F" in r["tones"] and r["source"] == NEWMAN],
        "plural or verbal noun (Newman)": [
            r for r in records if r["source"] == NEWMAN and r["kind"] in ("plural", "verbal-noun")
        ],
    }
    quota = {
        "frequent word (Newman or both)": 12,
        "Wiktionary only": 10,
        "Newman differs from Wiktionary": 12,
        "bare Newman entry (read as all-high)": 6,
        "falling tone (Newman)": 5,
        "plural or verbal noun (Newman)": 5,
    }
    assert set(quota) == set(pools)  # every stratum needs a quota
    picked: List[dict] = []
    used: Set[str] = set()
    for name, pool in pools.items():
        pool = [r for r in pool if r["word"] not in used]
        # weight towards frequent words but keep some variety
        rng.shuffle(pool)
        pool.sort(key=lambda r: -min(freq(r), 50) + rng.random() * 25)
        for r in pool[: quota[name]]:
            used.add(r["word"])
            picked.append({**r, "why": name})
    rest = [r for r in usable if r["word"] not in used]
    rng.shuffle(rest)
    for r in rest:
        if len(picked) >= n:
            break
        used.add(r["word"])
        picked.append({**r, "why": "random"})
    for r in picked:
        r["freq"] = freq(r)
        r["n_patterns"] = len(by_word[r["word"]])
    picked = picked[:n]
    picked.sort(key=lambda r: (-r["freq"], r["word"]))
    return picked


def write_review(sample: List[dict], out_dir: Path) -> None:
    rows = []
    for i, r in enumerate(sample, 1):
        alt = " / ".join(fmt(p) for p in r.get("alt_wiktionary", []))
        rows.append(
            {
                "n": i,
                "word": r["word"],
                "meaning": r["gloss"],
                "pos": r["pos"],
                "source": r["confidence"],
                "proposed": r["marked"],
                "proposed_tones": fmt(r["tones"]),
                "wiktionary_differs": alt,
                "corpus_count": r["freq"],
                "why_in_sample": r["why"],
                "correct_tones": "",
                "notes": "",
            }
        )
    with (out_dir / "review.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Hausa tone lexicon: review sheet (Newman first)",
        "",
        f"{len(rows)} words, weighted towards frequent words and towards the cases that need a native "
        "speaker. `source` is where the proposed pattern comes from: `both` (Newman and Wiktionary "
        "agree), `newman`, `newman_bare` (Newman has no tone mark, so it is read as all-high) or "
        "`wiktionary_only` (the weakest tier).",
        "",
        "| # | word | meaning | proposed | → tones | source | Wiktionary differs | count |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | **{r['word']}** | {r['meaning']} ({r['pos']}) | {r['proposed']} | "
            f"`{r['proposed_tones']}` | {r['source']} | `{r['wiktionary_differs'] or '—'}` | {r['corpus_count']} |"
        )
    (out_dir / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--wiktionary", required=True, type=Path)
    ap.add_argument(
        "--newman", type=Path, help="Newman (1977) pairs; the primary source when given"
    )
    ap.add_argument("--corpus", type=Path, help="Hausa text (jsonl with a text/content field)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wik, wik_stats = build_from_wiktionary(load_jsonl(args.wiktionary))
    new, new_stats = build_from_newman(load_jsonl(args.newman)) if args.newman else ([], Counter())
    merged = merge(new, wik)
    sort_key = lambda r: (r["word"], r["kind"], r["tones"])  # noqa: E731
    write_jsonl(args.out_dir / "lexicon.jsonl", sorted(merged, key=sort_key))
    write_jsonl(args.out_dir / "lexicon-newman.jsonl", sorted(new, key=sort_key))
    write_jsonl(args.out_dir / "lexicon-wiktionary.jsonl", sorted(wik, key=sort_key))

    trusted = runtime_mapping(merged, TRUSTED_TIERS)
    with_bare = runtime_mapping(merged, BARE_TIERS)
    everything = runtime_mapping(merged, BARE_TIERS + ("wiktionary_only",))
    (args.out_dir / "runtime-trusted.json").write_text(
        json.dumps(trusted, ensure_ascii=False, indent=0) + "\n", encoding="utf-8"
    )
    (args.out_dir / "runtime-with-bare.json").write_text(
        json.dumps(with_bare, ensure_ascii=False, indent=0) + "\n", encoding="utf-8"
    )
    (args.out_dir / "runtime-all.json").write_text(
        json.dumps(everything, ensure_ascii=False, indent=0) + "\n", encoding="utf-8"
    )

    tiers = Counter(r["confidence"] for r in merged)
    words_by_tier = {t: len({r["word"] for r in merged if r["confidence"] == t}) for t in tiers}
    report: Dict[str, object] = {
        "wiktionary": dict(wik_stats),
        "newman": dict(new_stats),
        "records_by_tier": dict(tiers),
        "distinct_words_by_tier": words_by_tier,
        "distinct_words_total": len({r["word"] for r in merged}),
        "newman_wins_over_a_differing_wiktionary_pattern": sum(
            1 for r in merged if r.get("alt_wiktionary")
        ),
        "runtime_words": {
            "trusted": len(trusted),
            "with_bare": len(with_bare),
            "with_wiktionary_only": len(everything),
        },
    }
    old = {w: tuple(t) for w, t in GYARA_LEXICON.items()}
    comparable = [w for w in old if w in trusted]
    report["old_101_word_list_vs_trusted"] = {
        "words_in_both": len(comparable),
        "agree": sum(1 for w in comparable if old[w] == tuple(trusted[w])),
    }
    counts: Counter = Counter()
    if args.corpus:
        counts = corpus_counts(args.corpus)
        report["corpus_tokens"] = sum(counts.values())
        report["corpus_coverage"] = {
            "trusted": round(coverage(trusted, counts), 4),
            "with_bare": round(coverage(with_bare, counts), 4),
            "with_wiktionary_only": round(coverage(everything, counts), 4),
        }
    if args.sample and merged:
        write_review(review_sample(merged, counts, args.sample, args.seed), args.out_dir)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
