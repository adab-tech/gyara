"""Render patterns with the JavaScript engine and check them by the spacing of the glottal pulses.

For every base in voice.json, renders the patterns you ask for (the ones in words.json, or every
H/L/F pattern with --all) through node_render.js, then measures the pitch of each syllable from the
output audio, period by period, and compares it with the intended contour.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import parselmouth  # noqa: E402
import soundfile as sf  # noqa: E402
from parselmouth.praat import call  # noqa: E402

import g2p  # noqa: E402

HALF = 3.25  # half the high-low distance in semitones (tonevoice.RANGE_ST / 2)


def jobs_from_words(words, base_keys):
    jobs, seen = [], set()
    for w in words:
        longs = g2p.syllable_lengths(w["marked"], w["sylls"])
        key = "".join(g2p.respell(s, lg) for s, lg in zip(w["sylls"], longs))
        pats = [w["tones"]] + [n["tones"] for n in w.get("newman", [])] + ([w["old"]] if w.get("old") else [])
        pats += [o["tones"] for o in w.get("others", [])]
        for p in pats:
            if key in base_keys and (key, "".join(p)) not in seen:
                seen.add((key, "".join(p)))
                jobs.append(dict(key=key, tones=list(p)))
    return jobs


def jobs_all(base_dir):
    jobs = []
    for f in sorted(os.listdir(base_dir)):
        if f.endswith(".json"):
            b = json.load(open(os.path.join(base_dir, f), encoding="utf-8"))
            jobs += [dict(key=b["key"], tones=list(c)) for c in itertools.product("HLF", repeat=len(b["nuclei"]))]
    return jobs


def check(jobs, base_dir, outdir):
    bad, n_syll, worst = [], 0, 0.0
    for j in jobs:
        b = json.load(open(os.path.join(base_dir, f"{j['key']}.json"), encoding="utf-8"))
        y, sr = sf.read(os.path.join(outdir, f"{j['key']}_{''.join(j['tones'])}.wav"))
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 0.7005:
            bad.append((j["key"], "".join(j["tones"]), "bad samples"))
            continue
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = call(snd, "To Pitch (cc)", 0.0025, 70, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 300)
        pp = call([snd, pitch], "To PointProcess (cc)")
        n = call(pp, "Get number of points")
        tt = np.array([call(pp, "Get time from index", i + 1) for i in range(n)])
        per = np.diff(tt)
        mid_t = (tt[:-1] + tt[1:]) / 2
        ok = per < 0.02
        st = 12 * np.log2((1 / per) / b["f0"])
        for k, ((s, e), tone) in enumerate(zip(b["nuclei"], j["tones"])):
            n_syll += 1
            sel = ok & (mid_t >= s) & (mid_t <= e)
            if sel.sum() < 4:
                bad.append((j["key"], "".join(j["tones"]), f"syll {k}: only {int(sel.sum())} periods in {e - s:.3f}s"))
                continue
            v = st[sel]
            start, end, mid = float(np.median(v[:3])), float(np.median(v[-3:])), float(np.median(v))
            if tone == "F":
                fine = start > 1.5 and end < -1.5
                err = max(abs(start - HALF), abs(end + HALF)) if fine else 99
            else:
                tgt = HALF if tone == "H" else -HALF
                fine = abs(mid - tgt) < 1.0 and abs(start - end) < 2.0
                err = abs(mid - tgt)
            if err < 99:
                worst = max(worst, err)
            if not fine:
                bad.append((j["key"], "".join(j["tones"]), f"syll {k} {tone}: {start:+.2f} {mid:+.2f} {end:+.2f} st"))
    return n_syll, bad, worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.path.join(HERE, "base"))
    ap.add_argument("--voice", default=os.path.join(HERE, "voice.json"))
    ap.add_argument("--words", help="words.json; check the patterns it lists")
    ap.add_argument("--all", action="store_true", help="check every H/L/F pattern of every base")
    a = ap.parse_args()
    keys = {f[:-5] for f in os.listdir(a.base) if f.endswith(".json")}
    if a.all:
        jobs = jobs_all(a.base)
    elif a.words:
        jobs = jobs_from_words(json.load(open(a.words, encoding="utf-8")), keys)
    else:
        ap.error("give --words or --all")
    with tempfile.TemporaryDirectory() as tmp:
        jp = os.path.join(tmp, "jobs.json")
        json.dump(jobs, open(jp, "w"))
        subprocess.run(["node", os.path.join(HERE, "node_render.js"), jp, a.voice, os.path.join(tmp, "out")], check=True)
        n_syll, bad, worst = check(jobs, a.base, os.path.join(tmp, "out"))
    print(f"{len(jobs)} renders, {n_syll} syllables checked; {len(bad)} failures; worst deviation among passing syllables {worst:.2f} st")
    for r in bad[:40]:
        print("  FAIL", *r)
    sys.exit(1 if bad else 0)
