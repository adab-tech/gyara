"""Reference implementation of the tone voice: impose a Hausa tone melody (H/L/F per syllable)
on a flat base recording with TD-PSOLA.  The browser version (tonevoice.js) is a line-for-line port.
"""
import math
import numpy as np

RANGE_ST = 6.5          # distance between a high and a low tone, in semitones
EASE_MS = 14.0          # how far inside a syllable's span the target is already reached
UNV_HOP_MS = 5.0        # hop for unvoiced stretches (copied unchanged)
RUN_GAP_MS = 24.0       # marks further apart than this start a new voiced run
FILL = 0.02             # weight of the untouched signal in the final normalisation


def base_f0(marks, sr):
    d = np.diff(np.asarray(marks, dtype=float))
    d = d[d < 0.012 * sr]
    return sr / float(np.median(d))


def contour_points(nuclei, tones, range_st=RANGE_ST):
    """(time_s, semitones_from_center) control points."""
    pts = []
    hi, lo = range_st / 2.0, -range_st / 2.0
    for (s, e), t in zip(nuclei, tones):
        a = min(EASE_MS / 1000.0, 0.15 * (e - s))
        t0, t1 = s + a, max(s + a + 1e-3, e - a)
        v0, v1 = {"H": (hi, hi), "L": (lo, lo), "F": (hi, lo)}[t]
        pts.append((t0, v0))
        pts.append((t1, v1))
    return pts


def contour_st(pts, t):
    """Semitone offset at time t (seconds): hold at the ends, half-cosine between points."""
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    # find segment
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (ta, va), (tb, vb) = pts[lo], pts[hi]
    if tb - ta <= 1e-9:
        return vb
    u = (t - ta) / (tb - ta)
    return va + (vb - va) * 0.5 * (1.0 - math.cos(math.pi * u))


def voiced_runs(marks, sr):
    m = np.asarray(marks, dtype=float)
    if len(m) == 0:
        return []
    runs, start = [], 0
    lim = RUN_GAP_MS / 1000.0 * sr
    for i in range(1, len(m)):
        if m[i] - m[i - 1] > lim:
            runs.append((start, i - 1)); start = i
    runs.append((start, len(m) - 1))
    return runs


def _rnd(v):
    return int(math.floor(v + 0.5))   # round half up, like JS Math.round


def psola(x, marks, f0_of_t, sr):
    """x: float array; marks: sample positions of glottal pulses; f0_of_t: seconds -> Hz."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = np.asarray(marks, dtype=float)
    runs = voiced_runs(marks, sr)
    P_med = float(np.median(np.diff(m)[np.diff(m) < 0.012 * sr])) if len(m) > 2 else 0.008 * sr
    # local analysis period per mark
    Pa = np.empty(len(m))
    for i in range(len(m)):
        nb = []
        if i > 0 and m[i] - m[i - 1] <= RUN_GAP_MS / 1000.0 * sr:
            nb.append(m[i] - m[i - 1])
        if i + 1 < len(m) and m[i + 1] - m[i] <= RUN_GAP_MS / 1000.0 * sr:
            nb.append(m[i + 1] - m[i])
        Pa[i] = np.mean(nb) if nb else P_med
    run_lo = [m[a] - Pa[a] / 2 for a, b in runs]
    run_hi = [m[b] + Pa[b] / 2 for a, b in runs]

    def run_at(t):
        for k, (lo, hi) in enumerate(zip(run_lo, run_hi)):
            if lo <= t <= hi:
                return k
        return -1

    y = np.zeros(n + 4096)
    ws = np.zeros(n + 4096)
    H = _rnd(UNV_HOP_MS / 1000.0 * sr)
    t = 0.0
    guard = 0
    while t < n and guard < 200000:
        guard += 1
        k = run_at(t)
        if k >= 0:
            a, b = runs[k]
            # nearest analysis mark inside the run
            j = a + int(np.argmin(np.abs(m[a:b + 1] - t)))
            P = Pa[j]
            c = m[j]
            L = _rnd(P)
            lo_i = _rnd(c) - L
            hi_i = _rnd(c) + L + 1
            seg_lo, seg_hi = max(lo_i, 0), min(hi_i, n)
            idx = np.arange(seg_lo, seg_hi)
            win = 0.5 * (1.0 + np.cos(math.pi * (idx - _rnd(c)) / (L + 1)))
            dst = _rnd(t) - L + (seg_lo - lo_i)
            if dst >= 0:
                y[dst:dst + len(idx)] += x[seg_lo:seg_hi] * win
                ws[dst:dst + len(idx)] += win
            f0 = f0_of_t(t / sr)
            t += sr / f0
        else:
            c = _rnd(t)
            lo_i, hi_i = c - H, c + H + 1
            seg_lo, seg_hi = max(lo_i, 0), min(hi_i, n)
            idx = np.arange(seg_lo, seg_hi)
            win = 0.5 * (1.0 + np.cos(math.pi * (idx - c) / (H + 1)))
            dst = seg_lo
            y[dst:dst + len(idx)] += x[seg_lo:seg_hi] * win
            ws[dst:dst + len(idx)] += win
            t += H
    y = y[:n]; ws = ws[:n]
    out = (y + FILL * x) / (ws + FILL)
    return out


def render(x, marks, nuclei, tones, sr, range_st=RANGE_ST, center_hz=None):
    c = center_hz or base_f0(marks, sr)
    pts = contour_points(nuclei, tones, range_st)
    f0 = lambda t: c * 2.0 ** (contour_st(pts, t) / 12.0)
    return psola(x, marks, f0, sr), f0
