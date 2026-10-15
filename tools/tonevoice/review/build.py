"""Assemble the review page: template + words.json + voice.json + the tone engine -> index.html.

index.html is what gets published. index.test.html is the same page with a charset declaration, for
the local browser test (the publishing step adds the declaration itself).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import g2p  # noqa: E402


def build(words_path, voice_path, out_dir):
    template = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    engine = open(os.path.join(os.path.dirname(HERE), "tonevoice.js"), encoding="utf-8").read()
    words = json.load(open(words_path, encoding="utf-8"))
    voice_text = open(voice_path, encoding="utf-8").read().strip()
    voice = json.loads(voice_text)
    for w in words:
        longs = g2p.syllable_lengths(w["marked"], w["sylls"])
        w["voice"] = "".join(g2p.respell(s, lg) for s, lg in zip(w["sylls"], longs))
        if w["voice"] not in voice:
            raise SystemExit(f"no base recording {w['voice']!r} for {w['word']!r}; run build_base.py and export_voice.py on these words")
    blobs = [json.dumps(words, ensure_ascii=False, separators=(",", ":")), voice_text, engine]
    for blob in blobs:
        assert "</script" not in blob.lower()
    for ph in ("/*__ENGINE__*/", "/*__WORDS__*/[]", "/*__VOICE__*/{}"):
        assert template.count(ph) == 1, ph
    page = template.replace("/*__ENGINE__*/", engine).replace("/*__WORDS__*/[]", blobs[0]).replace("/*__VOICE__*/{}", voice_text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8").write(page)
    open(os.path.join(out_dir, "index.test.html"), "w", encoding="utf-8").write('<!doctype html><meta charset="utf-8">' + page)
    return len(page)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("words")
    ap.add_argument("--voice", default=os.path.join(os.path.dirname(HERE), "voice.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    print(f"index.html: {build(a.words, a.voice, a.out) // 1024} KB in {a.out}")
