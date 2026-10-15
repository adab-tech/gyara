"""Synthesize a flat (monotone) base recording for every distinct review word and find its syllables.

Output per base:  base/<key>.wav (16 kHz mono) and base/<key>.json with
  sr, n, text, sylls, nuclei (seconds, one [start,end] per syllable, tone-bearing span),
  vowels (seconds, raw vowel spans), marks (glottal pulse positions, samples), events.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import parselmouth  # noqa: E402
import soundfile as sf  # noqa: E402
from parselmouth.praat import call  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import g2p  # noqa: E402
from espeak import EV_PHONEME, FLAT_VOICE, PITCH, RANGE, RATE, Espeak, prepare_data_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SR_OUT = 16000
VOICE = FLAT_VOICE
PARAMS = dict(rate=105, pitch=66, rng=0)
GAP_MS = 45          # glottal stop gap
SONORANT = {"m", "n", "N", "l", "R", "w", "j", "r", "n^"}
PAUSES = {"_", "_:", "_|"}
PAD_BEFORE, PAD_AFTER = 0.07, 0.09


def is_vowel(name):
    return name is not None and name.lstrip("',") in tuple(g2p.VOWELS)


def synth_word(e, text):
    e.set_voice(VOICE)
    e.set_param(RATE, PARAMS["rate"]); e.set_param(PITCH, PARAMS["pitch"]); e.set_param(RANGE, PARAMS["rng"])
    x, ev = e.synth(text)
    ph = [(name, ms / 1000.0) for (typ, ms, _tp, _ln, name) in ev if typ == EV_PHONEME]
    return x, ph


def analyse(word):
    sylls = word["sylls"]
    longs = g2p.syllable_lengths(word["marked"], sylls)
    parts = [g2p.respell(s, l) for s, l in zip(sylls, longs)]
    return "".join(parts), parts, longs


def build(e, key, text, sylls, parts, longs):
    x, ph = synth_word(e, text)
    sr = e.sr
    # events -> intervals
    total = len(x) / sr
    starts = [t for _, t in ph]
    names = [n for n, _ in ph]
    ends = starts[1:] + [total]
    # drop pure pause/closure events from consideration but keep their time slots
    vow_idx = [i for i, n in enumerate(names) if is_vowel(n)]
    need = [g2p.vowel_letters(p) for p in parts]
    if sum(need) != len(vow_idx):
        raise RuntimeError(f"{key}: expected {sum(need)} vowel events, got {len(vow_idx)}: {list(zip(names, starts))}")
    # assign vowel events to syllables in order
    groups, k = [], 0
    for c in need:
        groups.append(vow_idx[k:k + c]); k += c
    # sanity: vowel events of one syllable must be adjacent events
    for g in groups:
        if any(b - a != 1 for a, b in zip(g, g[1:])):
            raise RuntimeError(f"{key}: vowel events of one syllable are not adjacent: {g} {names}")
    # tone-bearing spans (seconds): vowel(s), extended over an immediately following sonorant coda
    nuclei, vowels = [], []
    for si, g in enumerate(groups):
        s, t = starts[g[0]], ends[g[-1]]
        vowels.append([s, t])
        j = g[-1] + 1
        if g2p.has_coda(sylls[si]) and j < len(names) and names[j] in SONORANT:
            t = ends[j]
        nuclei.append([s, t])
    # glottal gaps before syllables that start with a glottal stop
    gaps = []
    for si, syl in enumerate(sylls):
        if g2p.has_glottal_onset(syl) and si > 0:
            gaps.append(starts[groups[si][0]])
    x = x.astype(np.float64) / 32768.0
    shift = 0.0
    gap_n = int(round(GAP_MS / 1000 * sr))
    fade = int(0.006 * sr)
    # insert silence, shifting every later time
    def shift_times(t0, d):
        """Everything that starts at or after the cut moves by d; an interval that ends exactly at the cut stays."""
        nonlocal nuclei, vowels, starts, ends
        fs = lambda t: t + d if t >= t0 - 1e-9 else t
        fe = lambda t: t + d if t > t0 + 1e-9 else t
        nuclei = [[fs(a), fe(b)] for a, b in nuclei]
        vowels = [[fs(a), fe(b)] for a, b in vowels]
        starts = [fs(a) for a in starts]
        ends = [fe(a) for a in ends]
    for tg in gaps:
        n0 = int(round((tg + shift) * sr))
        # cut at the nearest zero crossing (rising) within 6 ms
        w = x[max(0, n0 - int(0.006 * sr)): n0 + int(0.006 * sr)]
        zc = np.where((w[:-1] < 0) & (w[1:] >= 0))[0]
        if len(zc):
            n0 = max(0, n0 - int(0.006 * sr)) + int(zc[np.argmin(np.abs(zc - len(w) // 2))])
        a, b = x[:n0].copy(), x[n0:].copy()
        a[-fade:] *= np.linspace(1, 0, fade)
        b[:fade] *= np.linspace(0, 1, fade)
        x = np.concatenate([a, np.zeros(gap_n), b])
        shift_times(tg + shift, gap_n / sr)
        shift += gap_n / sr
    # trim to speech with padding
    speech_start = starts[0]
    last = max(i for i, n in enumerate(names) if n not in PAUSES)
    speech_end = ends[last]
    t0 = max(0.0, speech_start - PAD_BEFORE)
    t1 = min(len(x) / sr, speech_end + PAD_AFTER)
    n0, n1 = int(t0 * sr), int(t1 * sr)
    x = x[n0:n1]
    nuclei = [[a - t0, b - t0] for a, b in nuclei]
    vowels = [[a - t0, b - t0] for a, b in vowels]
    # fades
    f = int(0.008 * sr)
    x[:f] *= np.linspace(0, 1, f); x[-f:] *= np.linspace(1, 0, f)
    # resample + normalize
    x = resample_poly(x, SR_OUT, sr)
    peak = np.max(np.abs(x))
    x = x * (0.7 / peak)
    return x, nuclei, vowels, list(zip(names, starts, ends))


def pitch_marks(x, sr, f_lo=70, f_hi=220):
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = call(snd, "To Pitch (cc)", 0.005, f_lo, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f_hi)
    pp = call([snd, pitch], "To PointProcess (cc)")
    n = call(pp, "Get number of points")
    times = [call(pp, "Get time from index", i + 1) for i in range(n)]
    return np.array(times), pitch


def repair_marks(x, marks_s, sr):
    """Fill the occasional dropped pulse (taps, flutter) so every voiced run has regular marks.
    Returns (marks in samples, typical period in samples, run-break gap in samples)."""
    m = np.round(np.asarray(marks_s) * sr)
    d = np.diff(m)
    P0 = float(np.median(d[d < 0.013 * sr]))
    brk = 2.6 * P0
    out = [m[0]]
    filled = 0
    for a, b in zip(m[:-1], m[1:]):
        g = b - a
        if 1.4 * P0 < g <= brk:
            k = max(2, int(round(g / P0)))
            for j in range(1, k):
                pos = a + j * g / k
                lo, hi = int(pos - 0.0015 * sr), int(pos + 0.0015 * sr) + 1
                lo = max(lo, 0)
                out.append(float(lo + int(np.argmax(x[lo:hi]))))
                filled += 1
        out.append(b)
    return np.array(out), P0, brk, filled


def refine_nuclei(nuclei, marks, sr, P0):
    """The vowel's event span can include a silent stop closure (before k, t); keep only the voiced stretch,
    so a tone contour is spread over sound the listener actually hears."""
    mk = np.asarray(marks) / sr
    half = P0 / sr / 2
    out = []
    for s, e in nuclei:
        inside = mk[(mk >= s - 0.005) & (mk <= e + 0.005)]
        if len(inside) < 3:
            out.append([s, e]); continue
        out.append([max(s, inside[0] - half), min(e, inside[-1] + half)])
    return out


def main(words_path, out_dir, data_dir):
    os.makedirs(out_dir, exist_ok=True)
    ws = json.load(open(words_path, encoding="utf-8"))
    e = Espeak(prepare_data_dir(data_dir))
    bases = {}
    for w in ws:
        text, parts, longs = analyse(w)
        if text in bases:
            bases[text]["ids"].append(w["id"]); continue
        x, nuclei, vowels, evs = build(e, w["word"], text, w["sylls"], parts, longs)
        sf.write(os.path.join(out_dir, f"{text}.wav"), x.astype(np.float32), SR_OUT, subtype="PCM_16")
        marks_s, pitch = pitch_marks(x, SR_OUT)
        marks, P0, brk, filled = repair_marks(x, marks_s, SR_OUT)
        nuclei = refine_nuclei(nuclei, marks, SR_OUT, P0)
        info = dict(key=text, word=w["word"], ids=[w["id"]], sr=SR_OUT, n=int(len(x)), sylls=w["sylls"], parts=parts,
                    longs=longs, nuclei=nuclei, vowels=vowels, marks=[int(v) for v in marks], f0=round(SR_OUT / P0, 2),
                    run_gap=round(brk, 2), filled=filled,
                    events=[[n, round(a, 4), round(b, 4)] for n, a, b in evs])
        json.dump(info, open(os.path.join(out_dir, f"{text}.json"), "w"), ensure_ascii=False)
        bases[text] = info
        d = np.diff(marks) / SR_OUT * 1000
        inrun = d[d < brk / SR_OUT * 1000]
        print(f"{text:14s} dur={len(x)/SR_OUT:.2f}s marks={len(marks):3d} f0={SR_OUT / P0:.1f} filled={filled} in-run period ms: min={inrun.min():.1f} max={inrun.max():.1f}")
    print(len(bases), "bases")
    return bases


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("words", help="words.json (see README for the fields each word needs)")
    ap.add_argument("--out", default=os.path.join(HERE, "base"), help="folder for base/<key>.wav and .json")
    ap.add_argument("--data", default=os.path.join(HERE, ".espeak-data"), help="eSpeak data copy with the custom voice")
    a = ap.parse_args()
    main(a.words, a.out, a.data)
