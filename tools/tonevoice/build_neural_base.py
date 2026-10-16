"""Words -> base recordings from a Piper (VITS) voice, in the same format build_base.py writes.

    python build_neural_base.py words.json --model model.onnx --speaker F1 --out base-f1

Each word is spoken as plain text (no tone marks). A lone word makes the duration predictor
erratic (a 0.4 s closure before an initial k, a vowel of 11 ms), so the word is first timed inside
a carrier sentence and then spoken alone with those durations, slowed by --rate. Because the
durations are known, so is where each vowel is: a syllable's tone-bearing stretch is its vowel
(and a sonorant coda) as the voice timed it, trimmed to the part that is audible and voiced. The tone engine
re-times the pulses, so whatever pitch the voice chose is replaced by the pattern.

The defaults are Murya's own (backend/services/vits_engine.py and fly.toml): noise scale 0.667 and
pace 1.25. The voice's own pitch drifts down across a word, and the further it has to be moved, the
rougher the result, so --candidates takes are drawn (the noise makes each one different) and the one
whose pitch strays least inside the vowels, with the clearest voice and the cleanest pulses, is kept.

Words are `{word, sylls, marked?, key?}`. A macron in `marked` makes that syllable's vowel long (the
voice reads plain text and cannot be told otherwise, so its duration is set directly). Entries with
the same `key` (default: `word`) share one recording; give entries that differ in vowel length
different keys.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from parselmouth import Sound  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import g2p  # noqa: E402
from marks import pitch_marks, refine_nuclei, repair_marks  # noqa: E402
from nuclei import audible, detect  # noqa: E402
from piper_base import PiperVoice, char_spans, long_vowel_positions, plan_frames, syllable_nuclei  # noqa: E402

PAD_BEFORE, PAD_AFTER = 0.07, 0.09


def trim(x, sr, floor_db=-40.0):
    """Cut leading and trailing silence, keep a short pad. Returns (x, seconds removed at the start)."""
    env = np.abs(x)
    thr = env.max() * 10 ** (floor_db / 20)
    idx = np.where(env > thr)[0]
    a = max(0, int(idx[0] - PAD_BEFORE * sr))
    b = min(len(x), int(idx[-1] + PAD_AFTER * sr))
    return x[a:b], a / sr


def wander(marks, sr, nuclei):
    """How far the recording's own pitch strays from its median inside the nuclei, RMS in semitones.
    The tone engine has to move the pitch by this much on top of the tone itself, so a flat take is best."""
    m = np.asarray(marks, float)
    per = np.diff(m)
    mid = (m[:-1] + m[1:]) / 2 / sr
    inside = np.zeros(len(per), bool)
    for s_, e_ in nuclei:
        inside |= (mid >= s_) & (mid <= e_)
    sel = inside & (per < 0.013 * sr)
    if sel.sum() < 4:
        return 99.0
    st = -12 * np.log2(per[sel])
    return float(np.sqrt(np.mean((st - np.median(st)) ** 2)))


def glitches(marks, sr):
    """Share of neighbouring pulse periods in a voiced run that differ by more than ~15 percent."""
    per = np.diff(np.asarray(marks, float))
    ok = per < 0.013 * sr
    r = per[1:] / per[:-1]
    both = ok[1:] & ok[:-1]
    return float(np.mean((r[both] < 0.87) | (r[both] > 1.15))) if both.any() else 1.0


def clarity(x, sr, f0):
    """Median harmonics-to-noise ratio (dB) of the voiced frames."""
    h = Sound(x, sampling_frequency=sr).to_harmonicity_cc(time_step=0.005, minimum_pitch=max(50.0, 0.5 * f0))
    v = h.values[0]
    v = v[v > -100]
    return float(np.median(v)) if len(v) else 0.0


def take(voice, word, speaker, frames, noise_scale, target_sr=None):
    """One recording of the word with the given durations. Returns (x, info)."""
    sylls = word["sylls"]
    text = "".join(sylls)
    audio, chars = voice.synth_timed(text, frames, speaker=speaker, noise_scale=noise_scale, noise_w=0.0)
    sr = voice.sr
    spans = char_spans(np.array(frames), chars, sr)
    nuclei = syllable_nuclei(sylls, spans, voice.id_map)
    x, cut = trim(audio.astype(np.float64), sr)
    if nuclei is not None:
        nuclei = [[a - cut, b - cut] for a, b in nuclei]
    if target_sr and target_sr != sr:
        x = resample_poly(x, target_sr, sr)
        sr = target_sr
    x = x * (0.7 / np.abs(x).max())
    snd = Sound(x, sampling_frequency=sr)
    p = snd.to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=500)
    f = p.selected_array["frequency"]
    f = f[f > 0]
    med = float(np.median(f)) if len(f) else 120.0
    placed = nuclei is not None
    if placed:
        nuclei = audible(nuclei, x, sr)
    else:  # the spelling and the character spans disagree: find the vowels in the audio instead
        nuclei = detect(x, sr, len(sylls))
    marks_s, _ = pitch_marks(x, sr, f_lo=max(50.0, 0.6 * med), f_hi=min(600.0, 1.8 * med))
    marks, P0, brk, filled = repair_marks(x, marks_s, sr)
    nuclei = refine_nuclei(nuclei, marks, sr, P0)
    return x, dict(
        key=word.get("key", text), word=word["word"], ids=[word["id"]], sr=sr, n=int(len(x)), sylls=sylls,
        nuclei=nuclei, marks=[int(v) for v in marks], f0=round(sr / P0, 2), run_gap=round(brk, 2), filled=filled,
        placed=placed,
    )


def score(x, info):
    """Lower is better: a take whose own pitch stays put, with a clear voice and regular pulses."""
    w = wander(info["marks"], info["sr"], info["nuclei"])
    c = clarity(x, info["sr"], info["f0"])
    g = glitches(info["marks"], info["sr"])
    return w + 0.25 * max(0.0, 18.0 - c) + 20.0 * g, dict(wander=round(w, 2), clarity=round(c, 1), glitches=round(g, 3))


def build(voice, word, speaker, rate=1.25, noise_scale=0.667, carrier="ya ce {} ne", target_sr=None, candidates=16):
    sylls = word["sylls"]
    text = "".join(sylls)
    longs = word.get("longs")
    if longs is None and word.get("marked"):
        longs = g2p.syllable_lengths(word["marked"], sylls)   # a macron in the marked spelling = a long vowel
    long_at = long_vowel_positions(sylls, longs, voice.id_map) if longs else ()
    _, frames = plan_frames(voice, text, speaker, carrier=carrier, rate=rate, long_at=long_at)
    best = None
    for _ in range(max(1, candidates)):
        x, info = take(voice, word, speaker, frames, noise_scale, target_sr)
        sc, detail = score(x, info)
        if best is None or sc < best[0]:
            best = (sc, x, info, detail)
    sc, x, info, detail = best
    info["take"] = dict(of=max(1, candidates), score=round(sc, 2), **detail)
    return x, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", help="default: <model>.json")
    ap.add_argument("--speaker", default="F1")
    ap.add_argument("--out", default=os.path.join(HERE, "base-neural"))
    ap.add_argument("--key-prefix", default="", help="prepended to every base key (e.g. f1-)")
    ap.add_argument("--rate", type=float, default=1.25, help="slow the in-sentence durations by this factor (default 1.25, Murya's pace)")
    ap.add_argument("--noise-scale", type=float, default=0.667, help="the voice's own variation (default 0.667, Murya's)")
    ap.add_argument("--candidates", type=int, default=16, help="takes to draw per word; the steadiest is kept (default 16)")
    ap.add_argument("--sr", type=int, default=0, help="resample to this rate (default: the voice's own)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    voice = PiperVoice(a.model, a.config)
    seen = {}
    for w in json.load(open(a.words, encoding="utf-8")):
        key = a.key_prefix + (w.get("key") or w.get("voice") or w["word"])
        if key in seen:
            seen[key]["ids"].append(w["id"])
            continue
        x, info = build(voice, dict(w, key=key), a.speaker, a.rate, a.noise_scale, target_sr=a.sr or None, candidates=a.candidates)
        sf.write(os.path.join(a.out, f"{key}.wav"), x.astype(np.float32), info["sr"], subtype="PCM_16")
        json.dump(info, open(os.path.join(a.out, f"{key}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        seen[key] = info
        print(f"{key:16s} {len(x) / info['sr']:.2f}s f0={info['f0']:.0f} marks={len(info['marks'])} placed={info['placed']} take={info['take']} "
              f"nuclei={[[round(p, 2), round(q, 2)] for p, q in info['nuclei']]}")
    print(len(seen), "bases in", a.out)


if __name__ == "__main__":
    main()
