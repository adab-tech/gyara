"""Pack the base recordings into the compact form the review page embeds: voice.json {key: {...}}."""
import argparse
import base64
import glob
import json
import os

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))


def export(base_dir, out_path):
    out = {}
    for p in sorted(glob.glob(os.path.join(base_dir, "*.json"))):
        b = json.load(open(p, encoding="utf-8"))
        x, sr = sf.read(os.path.join(base_dir, f"{b['key']}.wav"), dtype="int16")
        assert sr == b["sr"] and len(x) == b["n"]
        marks = b["marks"]
        deltas = [marks[0]] + [marks[i] - marks[i - 1] for i in range(1, len(marks))]
        out[b["key"]] = dict(
            sr=sr,
            n=len(x),
            f0=b["f0"],
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
    a = ap.parse_args()
    n, size = export(a.base, a.out)
    print(f"{n} bases, {size // 1024} KB -> {a.out}")
