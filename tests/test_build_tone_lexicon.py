"""Tests for scripts/build_tone_lexicon.py (the Wiktionary tone-notation parser)."""

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


def test_build_collects_inflected_forms_and_skips_junk():
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


def test_main_writes_outputs(tmp_path):
    src = tmp_path / "w.jsonl"
    rows = [_entry(f"w{i}", [{"form": "mālàm", "tags": ["canonical"]}]) for i in range(1)]
    rows[0]["word"] = "malam"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    assert b.main(["--wiktionary", str(src), "--out-dir", str(out), "--sample", "1"]) == 0
    assert json.loads((out / "lexicon.jsonl").read_text(encoding="utf-8"))["tones"] == ["H", "L"]
    assert (out / "report.json").exists() and (out / "review.md").exists()
