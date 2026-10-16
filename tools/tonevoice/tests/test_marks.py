"""repair_marks must judge a gap against the local period: a voice whose pitch falls by an octave inside a word
has long periods that are not dropped pulses."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("parselmouth")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marks import repair_marks  # noqa: E402

SR = 22050


def falling_train(f_start=340.0, f_end=170.0, seconds=0.6):
    """Pulse positions (seconds) of a voice whose pitch falls smoothly, and a waveform with a peak at each."""
    t, out = 0.01, []
    while t < seconds:
        out.append(t)
        f = f_start + (f_end - f_start) * (t / seconds)
        t += 1.0 / f
    x = np.zeros(int(seconds * SR) + 200)
    for p in out:
        i = int(round(p * SR))
        x[i - 20:i + 21] += np.hanning(41)
    return np.array(out), x


def test_a_falling_pitch_gets_no_extra_marks():
    marks, x = falling_train()
    fixed, P0, brk, filled = repair_marks(x, marks, SR)
    assert filled == 0
    assert len(fixed) == len(marks)


def test_a_dropped_pulse_is_still_filled():
    marks, x = falling_train()
    dropped = np.delete(marks, 40)
    fixed, P0, brk, filled = repair_marks(x, dropped, SR)
    assert filled == 1
    assert len(fixed) == len(marks)
    assert np.abs(fixed / SR - marks).max() < 0.0006
