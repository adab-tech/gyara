"""Build a tone-marked Hausa lexicon from the Wiktionary (Kaikki) extract.

    python scripts/build_tone_lexicon.py \
        --wiktionary path/to/kaikki-Hausa.jsonl \
        [--newman path/to/ha_en_pairs_merged.jsonl]   # validation only, never written to the output
        --out-dir build/tone

Outputs (in --out-dir):
    lexicon.jsonl        one line per (word, tone pattern), Wiktionary only
    report.json          counts, ambiguity, agreement with other sources
    review.md/.csv       a stratified sample for a native speaker to check (--sample N)

Notation (Wiktionary / Newman style, assumed -- see the review sheet):
    grave     L  low tone
    circumflex F  falling tone (HL on one syllable)
    acute     H  high tone (almost never written)
    no mark   H  high tone (inferred: high is the unmarked tone)
    macron    vowel LENGTH only; does not change the tone
    tilde on r (r̃), cedilla, dot-below, breve: consonant/vowel-quality marks, ignored

Wiktionary text is CC-BY-SA 4.0: any lexicon built from it carries that licence and needs
attribution to Wiktionary contributors (see ATTRIBUTION in the output folder).
"""

import argparse
import csv
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from gyara.syllables import syllabify
from gyara.tone import LEXICON as GYARA_LEXICON

GRAVE, ACUTE, CIRCUMFLEX, MACRON = "\u0300", "\u0301", "\u0302", "\u0304"
TONE_MARKS = {GRAVE: "L", ACUTE: "H", CIRCUMFLEX: "F"}
VOWELS = "aeiouAEIOU"
# Letters a Latin-script Hausa word may contain (hooked letters included).
LATIN_OK = set("abcdefghijklmnopqrstuvwxyz'’ʼɓɗƙƴ")


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


def first_gloss(entry: dict) -> str:
    for sense in entry.get("senses") or []:
        for g in sense.get("glosses") or []:
            return g[:80]
    return ""


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_from_wiktionary(entries: Iterable[dict]) -> Tuple[List[dict], Counter]:
    """One record per (word, form kind, tone pattern)."""
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
                    "source": "wiktionary_ha_ccbysa",
                }
            )
    stats["records"] = len(out)
    return out, stats


def patterns_by_word(records: Iterable[dict], only_explicit: bool = False) -> Dict[str, set]:
    by: Dict[str, set] = defaultdict(set)
    for r in records:
        if only_explicit and not r["explicit"]:
            continue
        by[r["word"]].add(tuple(r["tones"]))
    return by


def newman_patterns(rows: Iterable[dict]) -> Dict[str, set]:
    by: Dict[str, set] = defaultdict(set)
    for r in rows:
        h = (r.get("target_ha") or "").strip().lower()
        if not h or " " in h or "-" in h:
            continue
        parsed = parse_marked(h)
        if parsed and parsed[1]:  # only forms with at least one explicit tone mark
            by[plain(h)].add(tuple(parsed[0]))
    return by


def fmt(tones: Iterable[str]) -> str:
    return "-".join(tones)


def stratified_sample(records: List[dict], newman: Dict[str, set], n: int, seed: int) -> List[dict]:
    """Pick n rows across situations a reviewer should see, reproducibly."""
    rng = random.Random(seed)
    by_word = patterns_by_word(records, only_explicit=True)
    gyara = {w: tuple(t) for w, t in GYARA_LEXICON.items()}
    canon = [
        r
        for r in records
        if r["kind"] == "canonical"
        and r["explicit"]
        and not r["gloss"].lower().startswith(("alternative form", "alternative spelling"))
        and r["pos"] in ("noun", "verb", "adj", "adv", "num", "intj", "prep", "conj", "pron")
    ]
    buckets: Dict[str, List[dict]] = {
        "agrees with Newman": [],
        "differs from Newman": [],
        "homograph (several tone patterns)": [],
        "differs from gyara's current list": [],
        "short word (1-2 syllables)": [],
        "verb or adjective": [],
        "no Newman entry": [],
    }
    for r in canon:
        w, pat = r["word"], tuple(r["tones"])
        nm = newman.get(w)
        if nm:
            buckets["agrees with Newman" if pat in nm else "differs from Newman"].append(r)
        else:
            buckets["no Newman entry"].append(r)
        if len(by_word[w]) > 1:
            buckets["homograph (several tone patterns)"].append(r)
        if w in gyara and len(gyara[w]) == len(pat) and gyara[w] != pat:
            buckets["differs from gyara's current list"].append(r)
        if len(pat) <= 2:
            buckets["short word (1-2 syllables)"].append(r)
        if r["pos"] in ("verb", "adj"):
            buckets["verb or adjective"].append(r)
    quota = {
        "agrees with Newman": 9,
        "differs from Newman": 12,
        "homograph (several tone patterns)": 7,
        "differs from gyara's current list": 8,
        "short word (1-2 syllables)": 5,
        "verb or adjective": 5,
        "no Newman entry": 4,
    }
    picked: List[dict] = []
    used = set()
    for name, rows in buckets.items():
        rows = [r for r in rows if (r["word"], r["marked"]) not in used]
        rng.shuffle(rows)
        for r in rows[: quota[name]]:
            used.add((r["word"], r["marked"]))
            picked.append({**r, "why": name})
    pool = [r for r in canon if (r["word"], r["marked"]) not in used]
    rng.shuffle(pool)
    for r in pool:
        if len(picked) >= n:
            break
        used.add((r["word"], r["marked"]))
        picked.append({**r, "why": "random"})
    picked = picked[:n]
    picked.sort(key=lambda r: r["word"])
    return picked


def write_review(sample: List[dict], newman: Dict[str, set], out_dir: Path) -> None:
    gyara = {w: tuple(t) for w, t in GYARA_LEXICON.items()}
    rows = []
    for i, r in enumerate(sample, 1):
        w = r["word"]
        nm = newman.get(w)
        rows.append(
            {
                "n": i,
                "word": w,
                "meaning": r["gloss"],
                "pos": r["pos"],
                "wiktionary": r["marked"],
                "wiktionary_tones": fmt(r["tones"]),
                "newman_tones": " / ".join(fmt(t) for t in sorted(nm)) if nm else "",
                "gyara_current": fmt(gyara[w]) if w in gyara else "",
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
        "# Hausa tone lexicon: review sheet",
        "",
        f"{len(rows)} words from the Wiktionary extract, chosen to cover easy and hard cases. "
        "Please tell me which tone patterns are **right**, which are **wrong**, and what the "
        "correct pattern is. Corrections become a native-speaker test set.",
        "",
        "**How to read it.** Tones are per syllable: `H` high, `L` low, `F` falling (high-low "
        "on one syllable). I assumed an *unmarked* vowel is high and a *macron* only marks "
        "vowel length. If either assumption is wrong, that is the most important thing to tell "
        "me.",
        "",
        "| # | word | meaning | Wiktionary | → tones | Newman → tones | gyara now | why shown |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | **{r['word']}** | {r['meaning']} ({r['pos']}) | {r['wiktionary']} | "
            f"`{r['wiktionary_tones']}` | `{r['newman_tones'] or '—'}` | "
            f"`{r['gyara_current'] or '—'}` | {r['why_in_sample']} |"
        )
    (out_dir / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--wiktionary", required=True, type=Path)
    ap.add_argument("--newman", type=Path, help="validation only; never copied to the output")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, stats = build_from_wiktionary(load_jsonl(args.wiktionary))
    with (args.out_dir / "lexicon.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(records, key=lambda r: (r["word"], r["kind"], r["tones"])):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    explicit = patterns_by_word(records, only_explicit=True)
    report: Dict[str, object] = {
        **stats,
        "distinct_words": len({r["word"] for r in records}),
        "distinct_words_with_explicit_marks": len(explicit),
        "homographs_with_several_patterns": sum(1 for v in explicit.values() if len(v) > 1),
        "records_by_kind": dict(Counter(r["kind"] for r in records).most_common()),
        "records_by_pos": dict(Counter(r["pos"] for r in records).most_common(8)),
    }
    gyara = {w: tuple(t) for w, t in GYARA_LEXICON.items()}
    comparable = [w for w in gyara if w in explicit]
    report["gyara_current_lexicon_overlap"] = {
        "words_in_both": len(comparable),
        "agree": sum(1 for w in comparable if gyara[w] in explicit[w]),
    }
    newman: Dict[str, set] = {}
    if args.newman:
        newman = newman_patterns(load_jsonl(args.newman))
        both = [w for w in explicit if w in newman]
        report["newman_validation"] = {
            "words_marked_in_both": len(both),
            "identical_tone_pattern": sum(1 for w in both if explicit[w] & newman[w]),
        }
    if args.sample:
        write_review(
            stratified_sample(records, newman, args.sample, args.seed), newman, args.out_dir
        )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
