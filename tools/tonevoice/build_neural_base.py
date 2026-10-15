"""Words -> base recordings from a Piper (VITS) voice, in the same format build_base.py writes.

    python build_neural_base.py words.json --model model.onnx --speaker F1 --out base-f1

Each word is spoken as plain text (no tone marks). A lone word makes the duration predictor
erratic (a 0.4 s closure before an initial k, a vowel of 11 ms), so the word is first timed inside
a carrier sentence and then spoken alone with those durations, slowed by --rate. Because the
durations are known, so is where each vowel is: a syllable's tone-bearing stretch is its vowel
(and a sonorant coda) as the voice timed it, trimmed to the part that is audible and voiced. The tone engine
re-times the pulses, so whatever pitch the voice chose is replaced by the pattern.
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

from marks import pitch_marks, refine_nuclei, repair_marks  # noqa: E402
from nuclei import audible, detect  # noqa: E402
from piper_base import PiperVoice, char_spans, plan_frames, syllable_nuclei  # noqa: E402

PAD_BEFORE, PAD_AFTER = 0.07, 0.09


def trim(x, sr, floor_db=-40.0):
    """Cut leading and trailing silence, keep a short pad. Returns (x, seconds removed at the start)."""
    env = np.abs(x)
    thr = env.max() * 10 ** (floor_db / 20)
    idx = np.where(env > thr)[0]
    a = max(0, int(idx[0] - PAD_BEFORE * sr))
    b = min(len(x), int(idx[-1] + PAD_AFTER * sr))
    return x[a:b], a / sr


def build(voice, word, speaker, rate=1.8, noise_scale=0.3, carrier="ya ce {} ne", target_sr=None):
    sylls = word["sylls"]
    text = "".join(sylls)
    _, frames = plan_frames(voice, text, speaker, carrier=carrier, rate=rate)
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", help="default: <model>.json")
    ap.add_argument("--speaker", default="F1")
    ap.add_argument("--out", default=os.path.join(HERE, "base-neural"))
    ap.add_argument("--key-prefix", default="", help="prepended to every base key (e.g. f1-)")
    ap.add_argument("--rate", type=float, default=1.8, help="slow the in-sentence durations by this factor (default 1.8)")
    ap.add_argument("--noise-scale", type=float, default=0.3, help="lower = steadier voice (default 0.3)")
    ap.add_argument("--sr", type=int, default=0, help="resample to this rate (default: the voice's own)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    voice = PiperVoice(a.model, a.config)
    seen = {}
    for w in json.load(open(a.words, encoding="utf-8")):
        key = a.key_prefix + w["word"]
        if key in seen:
            seen[key]["ids"].append(w["id"])
            continue
        x, info = build(voice, dict(w, key=key), a.speaker, a.rate, a.noise_scale, target_sr=a.sr or None)
        sf.write(os.path.join(a.out, f"{key}.wav"), x.astype(np.float32), info["sr"], subtype="PCM_16")
        json.dump(info, open(os.path.join(a.out, f"{key}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        seen[key] = info
        print(f"{key:16s} {len(x) / info['sr']:.2f}s f0={info['f0']:.0f} marks={len(info['marks'])} placed={info['placed']} "
              f"nuclei={[[round(p, 2), round(q, 2)] for p, q in info['nuclei']]}")
    print(len(seen), "bases in", a.out)


if __name__ == "__main__":
    main()
