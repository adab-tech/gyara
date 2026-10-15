"""Pack the base recordings into the compact form the review page embeds: voice.json {key: {...}}."""
import argparse
import base64
import glob
import json
import os
from statistics import median

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))


def level(x, target_db, peak_max=0.95):
    """Scale int16 samples so the speech (samples above 2% of the peak) has RMS target_db (dBFS),
    but never so that the peak passes peak_max."""
    f = x.astype(np.float64) / 32768.0
    peak = np.abs(f).max()
    active = f[np.abs(f) > 0.02 * peak]
    gain = 10 ** (target_db / 20) / np.sqrt(np.mean(active ** 2))
    gain = min(gain, peak_max / peak)
    return np.clip(np.round(f * gain * 32768.0), -32768, 32767).astype(np.int16)


def export(base_dir, out_path, center="own", level_db=None):
    """center "own": each base's tone contour is centred on its own median pitch.
    center "voice": every base is centred on one pitch (the median of the bases' medians), so the
    words of a voice stay at one level instead of following whatever pitch each recording drifted to.
    level_db: give every base the same loudness (RMS of its speech, in dBFS)."""
    out = {}
    bases = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(base_dir, "*.json")))]
    common = float(median(b["f0"] for b in bases)) if bases else 0.0
    for b in bases:
        x, sr = sf.read(os.path.join(base_dir, f"{b['key']}.wav"), dtype="int16")
        assert sr == b["sr"] and len(x) == b["n"]
        if level_db is not None:
            x = level(x, level_db)
        marks = b["marks"]
        deltas = [marks[0]] + [marks[i] - marks[i - 1] for i in range(1, len(marks))]
        out[b["key"]] = dict(
            sr=sr,
            n=len(x),
            f0=round(common, 2) if center == "voice" else b["f0"],
            runGap=b["run_gap"],
            pcm=base64.b64encode(x.astype("<i2").tobytes()).decode("ascii"),
            marks=deltas,
            nuclei=[[round(a, 4), round(c, 4)] for a, c in b["nuclei"]],
        )
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    return len(out), os.path.getsize(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.path.join(HERE, "base"))
    ap.add_argument("--out", default=os.path.join(HERE, "voice.json"))
    ap.add_argument("--center", choices=["own", "voice"], default="own",
                    help="centre each word's contour on its own median pitch, or all on one (default own)")
    ap.add_argument("--level", type=float, default=None, metavar="DBFS",
                    help="give every word the same loudness: RMS of its speech in dBFS (e.g. -20)")
    a = ap.parse_args()
    n, size = export(a.base, a.out, a.center, a.level)
    print(f"{n} bases, {size // 1024} KB -> {a.out}")
