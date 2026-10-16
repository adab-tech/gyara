"""The Piper wrapper against a stand-in ONNX graph with the same inputs and the same Ceil node.

The stand-in is not a voice: each input id gets ceil(exp(0.02 * id + 1.5) * length_scale) frames and
the audio is a sine of the right length. It checks the plumbing that matters: ids built the way
Murya's backend builds them, the Ceil output found and recognised, and character spans that add up.
"""
import json
import os
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx import TensorProto as T  # noqa: E402
from onnx import helper as h  # noqa: E402

from piper_base import HOP, PiperVoice, char_spans, long_vowel_positions, plan_frames, syllable_nuclei  # noqa: E402

SYMBOLS = "_^$ !',-.:;?abcdefghijklmnorstuwyzɓɗƙƴpqvx"
CONFIG = {
    "audio": {"sample_rate": 22050},
    "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
    "phoneme_id_map": {s: [i] for i, s in enumerate(SYMBOLS)},
    "speaker_id_map": {"F2": 0, "M1": 6},
}


def fake_model(path):
    f = lambda name, vals, dt=T.FLOAT: h.make_tensor(name, dt, [len(vals)], vals)  # noqa: E731
    nodes = [
        h.make_node("Cast", ["input"], ["xf"], to=T.FLOAT),
        h.make_node("Mul", ["xf", "k_slope"], ["xs"]),
        h.make_node("Add", ["xs", "k_bias"], ["logw"]),
        h.make_node("Exp", ["logw"], ["wexp"]),
        h.make_node("Gather", ["scales", "k_one"], ["len_scale"], axis=0),
        h.make_node("Mul", ["wexp", "len_scale"], ["w"]),
        h.make_node("Ceil", ["w"], ["w_ceil"]),
        h.make_node("ReduceSum", ["w_ceil"], ["total"], keepdims=0),
        h.make_node("Mul", ["total", "k_hop"], ["nsamp"]),
        h.make_node("Cast", ["nsamp"], ["nsamp_i"], to=T.INT64),
        h.make_node("Reshape", ["nsamp_i", "k_shape1"], ["shape"]),
        h.make_node("ConstantOfShape", ["shape"], ["ones"], value=h.make_tensor("v", T.FLOAT, [1], [1.0])),
        h.make_node("CumSum", ["ones", "k_axis0"], ["t"]),
        h.make_node("Mul", ["t", "k_freq"], ["phase"]),
        h.make_node("Sin", ["phase"], ["sine"]),
        h.make_node("Reshape", ["sine", "k_out_shape"], ["output"]),
    ]
    inits = [
        f("k_slope", [0.02]), f("k_bias", [1.5]), f("k_hop", [float(HOP)]), f("k_freq", [0.05]),
        h.make_tensor("k_one", T.INT64, [1], [1]), h.make_tensor("k_axis0", T.INT64, [], [0]),
        h.make_tensor("k_shape1", T.INT64, [1], [1]), h.make_tensor("k_out_shape", T.INT64, [3], [1, 1, -1]),
    ]
    graph = h.make_graph(
        nodes, "fake_vits",
        [h.make_tensor_value_info("input", T.INT64, [1, None]), h.make_tensor_value_info("input_lengths", T.INT64, [1]),
         h.make_tensor_value_info("scales", T.FLOAT, [3]), h.make_tensor_value_info("sid", T.INT64, [1])],
        [h.make_tensor_value_info("output", T.FLOAT, [1, 1, None])], inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.fixture()
def voice(tmp_path):
    fake_model(tmp_path / "model.onnx")
    json.dump(CONFIG, open(tmp_path / "model.onnx.json", "w", encoding="utf-8"), ensure_ascii=False)
    return PiperVoice(tmp_path / "model.onnx")


def test_ids_follow_the_backend_recipe(voice):
    ids, chars = voice.ids("Mùtûm ’")
    m = CONFIG["phoneme_id_map"]
    # NFD drops the tone marks (they are not in the map) and keeps the vowel; the curly quote becomes '
    expect = [m["^"][0], m["_"][0]]
    for ch in "mutum '":
        expect += [m[ch][0], m["_"][0]]
    expect += [m["$"][0]]
    assert ids == expect
    assert "".join(chars) == "mutum '"
    assert len(chars) == len(ids)


def test_hooked_letters_survive(voice):
    ids, chars = voice.ids("ɗari ƙasa")
    assert "ɗ" in chars and "ƙ" in chars


def test_durations_are_recovered_and_add_up(voice):
    audio, frames, chars = voice.synth("ruwa", speaker="M1")
    assert frames is not None and len(frames) == len(chars)
    assert abs(int(frames.sum()) * HOP - len(audio)) <= HOP
    spans = char_spans(frames, chars, voice.sr)
    assert [c for c, _, _ in spans] == list("ruwa")
    assert all(b > a for _, a, b in spans)
    assert spans[-1][2] <= len(audio) / voice.sr + 1e-6


def test_length_scale_slows_every_character(voice):
    _, f1, _ = voice.synth("kare", length_scale=1.0)
    _, f2, _ = voice.synth("kare", length_scale=2.0)
    assert f2.sum() > 1.8 * f1.sum()


def test_synth_timed_speaks_exactly_the_durations_it_is_given(voice):
    ids, chars = voice.ids("ruwa")
    frames = [3, 1] + [6, 2] * 4 + [3]
    assert len(frames) == len(ids)
    audio, chars2 = voice.synth_timed("ruwa", frames, speaker="M1")
    assert chars2 == chars
    assert len(audio) == sum(frames) * HOP
    spans = char_spans(np.array(frames), chars, voice.sr)
    assert spans[0][1] == pytest.approx((3 + 1) * HOP / voice.sr)   # r starts after bos and its pad


def test_synth_timed_rejects_the_wrong_number_of_durations(voice):
    with pytest.raises(ValueError):
        voice.synth_timed("ruwa", [1, 2, 3])


def test_plan_frames_are_whole_frames_and_vowels_get_room_for_a_tone(voice):
    word, frames = plan_frames(voice, "ruwa", "M1", rate=1.8)
    ids, chars = voice.ids("ruwa")
    assert word == "ruwa" and len(frames) == len(ids)
    assert all(isinstance(f, int) and f >= 1 for f in frames)
    vowels = [i for i, c in enumerate(chars) if c in set("aeiou")]
    assert all(frames[i] >= 7 for i in vowels)
    assert frames[vowels[-1]] >= 11           # the vowel that ends the word is held longer


def test_syllable_nuclei_are_the_vowel_and_a_sonorant_coda(voice):
    # san-nu: the first nucleus is the a and the n that closes the syllable, the second is the u alone
    spans = [("s", 0.00, 0.10), ("a", 0.10, 0.20), ("n", 0.20, 0.30), ("n", 0.30, 0.38), ("u", 0.38, 0.50)]
    assert syllable_nuclei(["san", "nu"], spans, voice.id_map) == [[0.10, 0.30], [0.38, 0.50]]
    # an obstruent coda is not part of it, and an onset never is
    spans = [("b", 0.0, 0.1), ("a", 0.1, 0.2), ("k", 0.2, 0.3), ("y", 0.3, 0.4), ("a", 0.4, 0.5)]
    assert syllable_nuclei(["bak", "ya"], spans, voice.id_map) == [[0.1, 0.2], [0.4, 0.5]]


def test_syllable_nuclei_give_up_when_spelling_and_spans_disagree(voice):
    spans = [("k", 0.0, 0.1), ("a", 0.1, 0.2)]
    assert syllable_nuclei(["ka", "re"], spans, voice.id_map) is None


def test_long_vowels_are_held_longer(voice):
    # ru-wa with the final a long, and with the first u long
    pos = long_vowel_positions(["ru", "wa"], [False, True], voice.id_map)
    assert pos == {3}                                  # r u w a: the a
    _, frames = plan_frames(voice, "ruwa", "M1", rate=1.8, long_at=pos)
    ids, chars = voice.ids("ruwa")
    vowels = [i for i, c in enumerate(chars) if c in set("aeiou")]
    assert frames[vowels[-1]] >= 17
    pos = long_vowel_positions(["ru", "wa"], [True, False], voice.id_map)
    _, frames = plan_frames(voice, "ruwa", "M1", rate=1.8, long_at=pos)
    assert frames[vowels[0]] >= 13 and frames[vowels[-1]] < 17


def test_a_diphthong_is_not_lengthened_by_a_macron(voice):
    assert long_vowel_positions(["kai", "ya"], [True, False], voice.id_map) == set()
    assert long_vowel_positions(["ɗa", "ƙi"], [True, True], voice.id_map) == {1, 3}
