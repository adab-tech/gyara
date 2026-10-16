"""Glottal pulses ("pitch marks") of a recording, and the helpers that clean them up."""
import numpy as np
import parselmouth
from parselmouth.praat import call


def pitch_marks(x, sr, f_lo=70, f_hi=220):
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = call(snd, "To Pitch (cc)", 0.005, f_lo, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f_hi)
    pp = call([snd, pitch], "To PointProcess (cc)")
    n = call(pp, "Get number of points")
    times = [call(pp, "Get time from index", i + 1) for i in range(n)]
    return np.array(times), pitch


def repair_marks(x, marks_s, sr):
    """Fill the occasional dropped pulse (taps, flutter) so every voiced run has regular marks.

    A dropped pulse leaves a gap about twice the *local* period, so the gap is judged against the
    intervals just before and after it, not against one period for the whole word: a voice whose
    pitch falls by half an octave inside a word would otherwise get a bogus extra mark in every
    period of its low stretch (a double pulse, heard as buzz).
    Returns (marks in samples, typical period in samples, run-break gap in samples)."""
    m = np.round(np.asarray(marks_s) * sr)
    d = np.diff(m)
    valid = d < 0.013 * sr
    P0 = float(np.median(d[valid]))
    brk = 2.6 * P0
    out = [m[0]]
    filled = 0
    for i, (a, b) in enumerate(zip(m[:-1], m[1:])):
        g = b - a
        near = [d[j] for j in range(max(0, i - 4), min(len(d), i + 5)) if j != i and valid[j] and d[j] < 1.5 * (d[j - 1] if j > 0 and valid[j - 1] else d[j])]
        loc = float(np.median(near)) if len(near) >= 2 else P0
        if 1.6 * loc < g <= 2.6 * loc:
            k = max(2, int(round(g / loc)))
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
