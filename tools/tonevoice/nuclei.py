"""Find the vowel stretch of each syllable in a recording of one spoken word, from the audio alone.

For a base voice that gives no timings (a neural voice, a human recording). The syllable count is
known from the spelling, so the job is to pick that many sonority peaks, in order, and to take the
voiced stretch around each one.
"""
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks

STEP = 0.005          # seconds between frames
DROP_DB = 12.0        # a nucleus reaches down this far from its peak (and no further than the valley)
MIN_GAP_S = 0.07      # nearest two nuclei can be
EDGE_DB = 60.0        # padding level so a vowel at the very start or end still counts as a peak


def _frames(x, sr, f_lo=60, f_hi=400):
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=sr)
    band = call(snd, "Filter (pass Hann band)", 250, 3200, 100)
    inten = call(band, "To Intensity", 100, STEP, "yes")
    n = call(inten, "Get number of frames")
    t = np.array([call(inten, "Get time from frame number", i + 1) for i in range(n)])
    db = np.array([call(inten, "Get value in frame", i + 1) for i in range(n)])
    pitch = call(snd, "To Pitch (ac)", STEP, f_lo, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, f_hi)
    f0 = np.array([call(pitch, "Get value at time", float(tt), "Hertz", "linear") for tt in t])
    return t, np.nan_to_num(db, nan=-100.0), np.nan_to_num(f0, nan=0.0) > 0


def detect(x, sr, n, hints=None, tol=0.08):
    """Return [[start, end], ...] in seconds, one per syllable, in time order.

    `hints` (seconds, one per syllable) are rough expected centres, for example from a voice's
    duration predictor. Each syllable then takes the sonority peak nearest its hint, within `tol`.
    Without hints the n most prominent peaks are taken.
    """
    t, db, voiced = _frames(x, sr)
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    s = np.convolve(np.pad(db, 2, mode="edge"), k / k.sum(), mode="valid")
    s = np.where(voiced, s, s - 30.0)  # an unvoiced frame cannot be a nucleus
    # pad one low frame each side so peaks at the edges are found; indices shift by one
    padded = np.concatenate([[s.min() - 1], s, [s.min() - 1]])
    dist = max(1, int(MIN_GAP_S / STEP))
    pk, props = find_peaks(padded, prominence=0.05, distance=dist)
    cand = [(int(p) - 1, float(pr)) for p, pr in zip(pk, props["prominences"]) if 0 <= int(p) - 1 < len(s) and voiced[int(p) - 1]]
    if hints is not None:
        assert len(hints) == n
        keep, used = [], set()
        for h in hints:
            hf = int(round((h - t[0]) / STEP))
            near = [(abs(p - hf), -pr, p) for p, pr in cand if abs(p - hf) <= tol / STEP and p not in used]
            if near:
                p = min(near)[2]
            else:  # no sonority peak near the hint: loudest voiced frame there, else the hint itself
                lo, hi = max(0, hf - int(tol / STEP)), min(len(s), hf + int(tol / STEP) + 1)
                p = lo + int(np.argmax(s[lo:hi])) if hi > lo and voiced[lo:hi].any() else min(max(hf, 0), len(s) - 1)
            used.add(p)
            keep.append(p)
        keep = sorted(keep)
    else:
        cand.sort(key=lambda c: -c[1])
        keep = sorted(p for p, _ in cand[:n])
    while len(keep) < n:  # vowels run together: split the widest nucleus at its shallowest dip
        spans = _extents(s, voiced, keep) if keep else [(0, len(s) - 1)]
        i = int(np.argmax([b - a for a, b in spans]))
        a, b = spans[i]
        if b - a < 4:
            keep = sorted(set(keep) | {min(len(s) - 1, (keep[-1] if keep else 0) + int(0.1 / STEP))})
            continue
        cut = a + 1 + int(np.argmin(s[a + 1:b]))
        keep = sorted(set(keep) - {keep[i] if keep else -1} | {(a + cut) // 2, (cut + b) // 2})
    spans = _extents(s, voiced, keep)
    return [[float(t[a] - STEP / 2), float(t[b] + STEP / 2)] for a, b in spans]


def audible(nuclei, x, sr, drop_db=20.0, win=0.012, hop=0.002):
    """Cut each [start, end] down to the stretch around its loudest point that stays within drop_db of it.

    The last vowel of a word fades out over tens of milliseconds; a tone contour spread over the
    fade completes where nobody hears it (a fall on the last syllable would sound level)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(max(3, int(win * sr)))
    env = np.sqrt(np.convolve(x * x, w / w.sum(), mode="same"))
    step = max(1, int(hop * sr))
    db = 20 * np.log10(env[::step] + 1e-9)
    out = []
    for a, b in nuclei:
        lo, hi = max(0, int(np.ceil(a * sr / step))), min(len(db) - 1, int(b * sr / step))
        if hi - lo < 2:
            out.append([a, b])
            continue
        p = lo + int(np.argmax(db[lo:hi + 1]))
        i = p
        while i - 1 >= lo and db[i - 1] >= db[p] - drop_db:
            i -= 1
        j = p
        while j + 1 <= hi and db[j + 1] >= db[p] - drop_db:
            j += 1
        out.append([max(a, i * step / sr), min(b, (j + 1) * step / sr)])
    return out


def _extents(s, voiced, pk):
    """Frame range of each nucleus: down DROP_DB from the peak, no further than the valley to the
    neighbouring peak, and inside the voiced run."""
    spans = []
    for i, p in enumerate(pk):
        lo_lim = 0 if i == 0 else pk[i - 1] + int(np.argmin(s[pk[i - 1]:p + 1]))
        hi_lim = len(s) - 1 if i == len(pk) - 1 else p + int(np.argmin(s[p:pk[i + 1] + 1]))
        a = p
        while a - 1 >= lo_lim and voiced[a - 1] and s[a - 1] >= s[p] - DROP_DB:
            a -= 1
        b = p
        while b + 1 <= hi_lim and voiced[b + 1] and s[b + 1] >= s[p] - DROP_DB:
            b += 1
        spans.append((a, b))
    return spans
