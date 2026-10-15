"""export_voice: one pitch for all the words of a voice, and one loudness."""
import json
import os
import sys

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export_voice import export, level  # noqa: E402

SR = 16000


def write_base(folder, key, f0, amp):
    x = amp * np.sin(2 * np.pi * f0 * np.arange(SR // 2) / SR)
    sf.write(str(folder / f"{key}.wav"), x, SR, subtype="PCM_16")
    marks = [int(i * SR / f0) for i in range(1, 60)]
    json.dump(dict(key=key, sr=SR, n=len(x), f0=f0, run_gap=30.0, marks=marks, nuclei=[[0.05, 0.4]]),
              open(folder / f"{key}.json", "w"))


def speech_rms_db(x):
    f = x.astype(np.float64) / 32768
    return 20 * np.log10(np.sqrt(np.mean(f[np.abs(f) > 0.02 * np.abs(f).max()] ** 2)))


@pytest.fixture()
def bases(tmp_path):
    folder = tmp_path / "base"
    folder.mkdir()
    for key, f0, amp in (("a", 180.0, 0.5), ("b", 200.0, 0.2), ("c", 240.0, 0.7)):
        write_base(folder, key, f0, amp)
    return tmp_path


def test_each_word_keeps_its_own_pitch_by_default(bases):
    export(str(bases / "base"), str(bases / "own.json"))
    v = json.load(open(bases / "own.json"))
    assert [v[k]["f0"] for k in "abc"] == [180.0, 200.0, 240.0]


def test_center_voice_puts_every_word_on_the_median_pitch(bases):
    export(str(bases / "base"), str(bases / "voice.json"), center="voice")
    v = json.load(open(bases / "voice.json"))
    assert {v[k]["f0"] for k in "abc"} == {200.0}


def test_level_gives_every_word_the_same_loudness(bases):
    export(str(bases / "base"), str(bases / "level.json"), level_db=-16)
    import base64
    v = json.load(open(bases / "level.json"))
    for k in "abc":
        x = np.frombuffer(base64.b64decode(v[k]["pcm"]), dtype="<i2")
        assert speech_rms_db(x) == pytest.approx(-16, abs=0.3)


def test_level_never_pushes_the_peak_past_the_cap():
    t = np.arange(SR) / SR
    x = (0.05 * np.sin(2 * np.pi * 200 * t) * 32768).astype(np.int16)
    x[100] = 20000                      # one spike: a -16 dB speech level would clip it
    assert np.abs(level(x, -16)).max() <= 0.95 * 32768 + 1
