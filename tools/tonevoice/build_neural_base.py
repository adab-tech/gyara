"""Words -> base recordings from a Piper (VITS) voice, in the same format build_base.py writes.

    python build_neural_base.py words.json --model model.onnx --speaker M1 --out base-m1

Each word is spoken as plain text (no tone marks). The duration predictor says roughly where each
vowel is; the sonority peaks of the audio place its voiced stretch exactly (nuclei.py). The page's
tone engine then re-times the pulses, so whatever pitch the voice chose is replaced by the pattern.
"""
import argparse
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from parselmouth import Sound  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import g2p  # noqa: E402
from marks import pitch_marks, refine_nuclei, repair_marks  # noqa: E402
from nuclei import detect  # noqa: E402
from piper_base import PiperVoice, char_spans  # noqa: E402

PAD_BEFORE, PAD_AFTER = 0.07, 0.09
VOWELS = set(g2p.VOWELS)


def syllable_hints(sylls, spans, id_map):
    """Centre of each syllable's vowel(s), in seconds, from the character spans (None if they don't line up)."""
    per_syll = [[c for c in unicodedata.normalize("NFD", s).casefold() if c in id_map] for s in sylls]
    if sum(len(c) for c in per_syll) != len(spans):
        return None
    hints, k = [], 0
    for chars in per_syll:
        mine = spans[k:k + len(chars)]
        k += len(chars)
        v = [(a, b) for (c, a, b) in mine if c in VOWELS]
        if not v:
            return None
        hints.append((v[0][0] + v[-1][1]) / 2)
    return hints


def trim(x, sr, floor_db=-40.0):
    """Cut leading and trailing silence, keep a short pad. Returns (x, seconds removed at the start)."""
    env = np.abs(x)
    thr = env.max() * 10 ** (floor_db / 20)
    idx = np.where(env > thr)[0]
    a = max(0, int(idx[0] - PAD_BEFORE * sr))
    b = min(len(x), int(idx[-1] + PAD_AFTER * sr))
    return x[a:b], a / sr


def build(voice, word, speaker, params, target_sr=None):
    sylls = word["sylls"]
    text = "".join(sylls)
    audio, frames, chars = voice.synth(text, speaker=speaker, **params)
    sr = voice.sr
    hints = None
    if frames is not None:
        spans = char_spans(frames, chars, sr)
        hints = syllable_hints(sylls, spans, voice.id_map)
    x, cut = trim(audio.astype(np.float64), sr)
    if hints is not None:
        hints = [h - cut for h in hints]
    if target_sr and target_sr != sr:
        x = resample_poly(x, target_sr, sr)
        sr = target_sr
    x = x * (0.7 / np.abs(x).max())
    snd = Sound(x, sampling_frequency=sr)
    p = snd.to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=500)
    f = p.selected_array["frequency"]
    f = f[f > 0]
    med = float(np.median(f)) if len(f) else 120.0
    nuclei = detect(x, sr, len(sylls), hints=hints)
    marks_s, _ = pitch_marks(x, sr, f_lo=max(50.0, 0.6 * med), f_hi=min(600.0, 1.8 * med))
    marks, P0, brk, filled = repair_marks(x, marks_s, sr)
    nuclei = refine_nuclei(nuclei, marks, sr, P0)
    return x, dict(
        key=word.get("key", text), word=word["word"], ids=[word["id"]], sr=sr, n=int(len(x)), sylls=sylls,
        nuclei=nuclei, marks=[int(v) for v in marks], f0=round(sr / P0, 2), run_gap=round(brk, 2), filled=filled,
        hinted=hints is not None,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", help="default: <model>.json")
    ap.add_argument("--speaker", default="M1")
    ap.add_argument("--out", default=os.path.join(HERE, "base-neural"))
    ap.add_argument("--noise-scale", type=float, default=0.33, help="lower = steadier voice (default 0.33)")
    ap.add_argument("--noise-w", type=float, default=0.4, help="lower = steadier rhythm (default 0.4)")
    ap.add_argument("--length-scale", type=float, default=1.15, help="above 1 is slower (default 1.15)")
    ap.add_argument("--sr", type=int, default=0, help="resample to this rate (default: the voice's own)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    voice = PiperVoice(a.model, a.config)
    params = dict(noise_scale=a.noise_scale, noise_w=a.noise_w, length_scale=a.length_scale)
    seen = {}
    for w in json.load(open(a.words, encoding="utf-8")):
        key = w["word"]
        if key in seen:
            seen[key]["ids"].append(w["id"])
            continue
        x, info = build(voice, dict(w, key=key), a.speaker, params, a.sr or None)
        sf.write(os.path.join(a.out, f"{key}.wav"), x.astype(np.float32), info["sr"], subtype="PCM_16")
        json.dump(info, open(os.path.join(a.out, f"{key}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        seen[key] = info
        print(f"{key:14s} {len(x) / info['sr']:.2f}s f0={info['f0']:.0f} marks={len(info['marks'])} hinted={info['hinted']} "
              f"nuclei={[[round(p, 2), round(q, 2)] for p, q in info['nuclei']]}")
    print(len(seen), "bases in", a.out)


if __name__ == "__main__":
    main()
