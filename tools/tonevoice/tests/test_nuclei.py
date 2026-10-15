"""nuclei.audible on synthetic vowels: a steady one, one that fades, one that starts in silence."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("parselmouth")
pytest.importorskip("scipy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuclei import audible  # noqa: E402

SR = 16000


def tone(seconds, amp):
    t = np.arange(int(seconds * SR)) / SR
    return t, amp(t) * np.sin(2 * np.pi * 200 * t)


def test_a_steady_vowel_is_left_alone():
    _, x = tone(0.4, lambda t: 0.5 + 0 * t)
    (a, b), = audible([[0.1, 0.3]], x, SR)
    assert a == pytest.approx(0.1, abs=0.005) and b == pytest.approx(0.3, abs=0.005)


def test_a_fading_tail_is_cut_where_it_drops_out_of_earshot():
    # holds for 0.1 s, then fades at 100 dB/s: 20 dB down at 0.3 s
    _, x = tone(0.6, lambda t: np.where(t < 0.1, 1.0, 10 ** (-(t - 0.1) * 100 / 20)))
    (a, b), = audible([[0.0, 0.5]], x, SR, drop_db=20)
    assert a == 0.0 and 0.27 < b < 0.34


def test_a_quiet_start_is_skipped():
    _, x = tone(0.5, lambda t: np.where(t < 0.1, 0.001, 1.0))
    (a, b), = audible([[0.0, 0.5]], x, SR)
    assert 0.085 < a < 0.12 and b == 0.5
