"""Tests for scripts/build_tone_lexicon.py (tone-notation parser and the Newman-first lexicon)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_tone_lexicon.py"
_spec = importlib.util.spec_from_file_location("build_tone_lexicon", SCRIPT)
assert _spec is not None and _spec.loader is not None
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)


@pytest.mark.parametrize(
    "marked,tones,explicit",
    [
        ("àbàdâ", ["L", "L", "F"], True),  # grave = low, circumflex = falling
        ("mālàm", ["H", "L"], True),  # macron is length only: stays high
        ("rānā", ["H", "H"], True),  # only length marked: still a deliberate transcription
        ("bā̀ba", ["L", "H"], True),  # macron + grave = long low; unmarked = high
        ("dōkī̀", ["H", "L"], True),
        ("kâi", ["F"], True),  # diphthong ai is one syllable
        ("tauna", ["H", "H"], False),  # nothing marked: high is inferred
        ("bàr̃gō", ["L", "H"], True),  # tilde on r is a consonant, not tone
        ("kōr̃ḕ", ["H", "L"], True),
        ("ƙasà", ["H", "L"], True),  # hooked letters survive
        ("saba'in".replace("saba", "sàbà"), ["L", "L", "H"], True),  # glottal-stop syllable
        ("marāyá", ["H", "H", "H"], True),  # an explicit acute is high
    ],
)
def test_parse_marked(marked, tones, explicit):
    assert b.parse_marked(marked) == (tones, explicit)


@pytest.mark.parametrize("bad", ["", "Alh.", "کَفَا", "ts ts", "à1"])
def test_parse_marked_rejects(bad):
    assert b.parse_marked(bad) is None


def test_conflicting_marks_in_one_syllable_are_rejected():
    assert b.parse_marked("àí") is None  # low and high on the same diphthong


def _entry(word, forms, pos="noun", gloss="x"):
    return {"word": word, "pos": pos, "senses": [{"glosses": [gloss]}], "forms": forms}


def _row(head, gloss="g", context="n.m."):
    return {"target_ha": head, "source_en": gloss, "context": context, "provenance": "newman_1977"}


def test_wiktionary_collects_inflected_forms_and_skips_junk():
    entries = [
        _entry(
            "rana",
            [
                {"form": "rānā", "tags": ["canonical", "feminine"]},
                {"form": "rā̀nàikū", "tags": ["plural"]},
                {"form": "ran", "tags": ["possessed-form"]},
                {"form": "two words", "tags": ["plural"]},
            ],
        ),
        _entry("tuntu", [{"form": "تُنْتُو", "tags": ["canonical"]}]),
        _entry(
            "tsintsiya", [{"form": "tsintsinyā", "tags": ["canonical"]}]
        ),  # differs from headword
    ]
    records, stats = b.build_from_wiktionary(entries)
    by_form = {(r["word"], r["kind"]): r for r in records}
    canonical = by_form[("rana", "canonical")]
    assert canonical["tones"] == ["H", "H"] and canonical["explicit"] is True
    plural = [r for r in records if r["kind"] == "plural"][0]
    assert plural["word"] == "ranaiku" and plural["tones"] == ["L", "L", "H"]
    possessed = [r for r in records if r["kind"] == "possessed-form"][0]
    assert possessed["explicit"] is False
    assert stats["skipped_multiword_or_romanization"] == 1
    assert stats["skipped_unparseable"] == 1
    assert stats["skipped_form_differs_from_headword"] == 1


def test_forms_from_context():
    assert b.forms_from_context("n.m. (pl. zùgàzùgai)") == [("plural", "zùgàzùgai")]
    assert b.forms_from_context("v.t. (vn. shiri)") == [("verbal-noun", "shiri")]
    assert b.forms_from_context("n.f. (pl. jama'o'i, jama'u)") == [
        ("plural", "jama'o'i"),
        ("plural", "jama'u"),
    ]
    assert b.forms_from_context("n.m. (pl. ƙàyàyyakī or ƙayayyaki)") == [
        ("plural", "ƙàyàyyakī"),
        ("plural", "ƙayayyaki"),
    ]
    assert b.forms_from_context("n.m.") == []
    assert b.forms_from_context("") == []


def test_newman_marked_forms_are_explicit_and_bare_forms_are_inferred_all_high():
    rows = [
        _row("sacè", "steal", "v.t."),
        _row("ara", "borrow", "v.t. (vn. aro)"),  # bare headword and bare verbal noun
        _row("zugàzugi", "bellows", "n.m. (pl. zùgàzùgai)"),
        _row("dan ràhotò", "spy", "n.m. idiom"),  # multi-word headword
    ]
    records, stats = b.build_from_newman(rows)
    got = {(r["word"], r["kind"]): r for r in records}
    assert (
        got[("sace", "canonical")]["tones"] == ["H", "L"] and got[("sace", "canonical")]["explicit"]
    )
    assert got[("zugazugi", "canonical")]["tones"] == ["H", "L", "H", "H"]
    assert got[("zugazugai", "plural")]["tones"] == ["L", "L", "L", "H"]
    for key in [("ara", "canonical"), ("aro", "verbal-noun")]:
        r = got[key]
        assert r["tones"] == ["H", "H"] and r["inferred"] is True and r["explicit"] is False
    assert stats["records_canonical_bare_inferred_high"] == 1
    assert stats["records_verbal-noun_bare_inferred_high"] == 1
    assert stats["skipped_multiword_headword"] == 1
    assert all(r["source"] == "newman_1977" for r in records)


def _rec(word, tones, source, pos="noun", gloss="g", kind="canonical"):
    return {
        "word": word,
        "tones": tones,
        "marked": word,
        "kind": kind,
        "pos": pos,
        "lemma": word,
        "gloss": gloss,
        "explicit": True,
        "source": source,
    }


def test_merge_is_newman_first():
    n = [
        _rec("aa", ["L", "H"], b.NEWMAN),
        _rec("bb", ["H", "L"], b.NEWMAN),
        _rec("dd", ["H", "H"], b.NEWMAN),
    ]
    w = [
        _rec("aa", ["L", "H"], b.WIKTIONARY),  # agrees with Newman
        _rec("bb", ["L", "H"], b.WIKTIONARY),  # differs: Newman wins
        _rec("cc", ["F"], b.WIKTIONARY),  # only Wiktionary
    ]
    merged = {r["word"]: r for r in b.merge(n, w)}
    assert merged["aa"]["confidence"] == "both" and merged["aa"]["sources"] == [
        b.NEWMAN,
        b.WIKTIONARY,
    ]
    assert merged["bb"]["confidence"] == "newman" and merged["bb"]["tones"] == ["H", "L"]
    assert merged["bb"]["alt_wiktionary"] == [["L", "H"]]
    assert merged["cc"]["confidence"] == "wiktionary_only" and merged["cc"]["sources"] == [
        b.WIKTIONARY
    ]
    assert merged["dd"]["confidence"] == "newman" and "alt_wiktionary" not in merged["dd"]


def test_bare_newman_entries_get_their_own_tier_and_never_enter_the_trusted_mapping():
    bare = _rec("wani", ["H", "H"], b.NEWMAN)
    bare["explicit"], bare["inferred"] = False, True
    loan = _rec("alhaki", ["H", "H", "H"], b.NEWMAN)
    loan["explicit"], loan["inferred"] = False, True
    n = [bare, loan, _rec("doki", ["H", "L"], b.NEWMAN)]
    w = [_rec("wani", ["H", "H"], b.WIKTIONARY), _rec("alhaki", ["L", "H", "H"], b.WIKTIONARY)]
    merged = {r["word"]: r for r in b.merge(n, w)}
    assert merged["wani"]["confidence"] == "both"  # Wiktionary also says all-high
    assert merged["alhaki"]["confidence"] == "newman_bare"
    assert merged["alhaki"]["alt_wiktionary"] == [["L", "H", "H"]]
    records = list(merged.values())
    assert set(b.runtime_mapping(records, b.TRUSTED_TIERS)) == {"wani", "doki"}  # wani is 'both'
    assert set(b.runtime_mapping(records, b.BARE_TIERS)) == {"wani", "doki", "alhaki"}


def test_merge_ignores_untranscribed_wiktionary_forms():
    w = _rec("ee", ["H", "H"], b.WIKTIONARY)
    w["explicit"] = False
    assert b.merge([], [w]) == []


def test_runtime_mapping_drops_ambiguous_words_and_respects_tiers():
    n = [
        _rec("one", ["L", "H"], b.NEWMAN),
        _rec("two", ["L", "H"], b.NEWMAN, gloss="a"),
        _rec("two", ["H", "L"], b.NEWMAN, gloss="b"),  # homograph with different tones
    ]
    w = [_rec("three", ["H", "L"], b.WIKTIONARY)]
    merged = b.merge(n, w)
    trusted = b.runtime_mapping(merged, b.TRUSTED_TIERS)
    assert trusted == {"one": ["L", "H"]}
    assert b.runtime_mapping(merged, b.TRUSTED_TIERS + ("wiktionary_only",)) == {
        "one": ["L", "H"],
        "three": ["H", "L"],
    }


def test_main_writes_every_output(tmp_path):
    wik = tmp_path / "w.jsonl"
    wik.write_text(
        "\n".join(
            json.dumps(_entry(w, [{"form": f, "tags": ["canonical"]}]), ensure_ascii=False)
            for w, f in [("malam", "mālàm"), ("mutum", "mùtûm")]
        ),
        encoding="utf-8",
    )
    new = tmp_path / "n.jsonl"
    new.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in [_row("mùtum", "person"), _row("dokì", "horse"), _row("wani", "some")]
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"text": "malam mutum doki doki"}), encoding="utf-8")
    out = tmp_path / "out"
    assert (
        b.main(
            [
                "--wiktionary",
                str(wik),
                "--newman",
                str(new),
                "--corpus",
                str(corpus),
                "--out-dir",
                str(out),
                "--sample",
                "3",
            ]
        )
        == 0
    )
    for name in [
        "lexicon.jsonl",
        "lexicon-newman.jsonl",
        "lexicon-wiktionary.jsonl",
        "runtime-trusted.json",
        "runtime-all.json",
        "report.json",
        "review.md",
        "review.csv",
    ]:
        assert (out / name).exists(), name
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["records_by_tier"]["wiktionary_only"] == 1  # malam
    assert (
        report["newman_wins_over_a_differing_wiktionary_pattern"] == 1
    )  # mutum: Newman L-H vs Wiktionary L-F
    trusted = json.loads((out / "runtime-trusted.json").read_text(encoding="utf-8"))
    assert (
        trusted["mutum"] == ["L", "H"] and trusted["doki"] == ["H", "L"] and "malam" not in trusted
    )
    assert report["corpus_coverage"]["trusted"] == 0.75  # mutum, doki, doki out of 4 tokens
    assert report["corpus_coverage"]["with_wiktionary_only"] == 1.0
    # the Newman file carries no Wiktionary data and vice versa
    newman_lines = (out / "lexicon-newman.jsonl").read_text(encoding="utf-8").splitlines()
    wiktionary_lines = (out / "lexicon-wiktionary.jsonl").read_text(encoding="utf-8").splitlines()
    assert newman_lines and all(json.loads(x)["source"] == b.NEWMAN for x in newman_lines)
    assert wiktionary_lines and all(
        json.loads(x)["source"] == b.WIKTIONARY for x in wiktionary_lines
    )
