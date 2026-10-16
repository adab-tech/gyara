"""Run a Piper (VITS) ONNX voice on one word, and recover where each character landed in the audio.

The inference path matches Murya's backend (backend/services/vits_engine.py): the text is
NFD-normalised and casefolded, each character is looked up in `phoneme_id_map`, a pad id goes after
every character, and the sequence is wrapped in bos/eos. Tone marks are not in the map, so they are
dropped: the voice reads the word with whatever tone its training data gave it.

Where each character landed comes from the duration predictor. The exported graph does not return
it, so `PiperVoice` adds the output of the `Ceil` node (VITS: w_ceil, frames per input id) to the
graph. A frame is `hop` samples (256 for Piper).
"""
import json
import unicodedata

import numpy as np
import onnx
import onnxruntime as ort

HOP = 256
_FIXES = {"’": "'", "‘": "'", "`": "'", "–": "-", "—": "-", "\xa0": " "}


def _norm(text):
    """NFD, casefolded, typographic quotes and dashes made plain: the text as the voice sees it."""
    text = unicodedata.normalize("NFD", text).casefold()
    for a, b in _FIXES.items():
        text = text.replace(a, b)
    return text


class PiperVoice:
    def __init__(self, model_path, config_path=None, providers=("CPUExecutionProvider",)):
        cfg_path = config_path or str(model_path) + ".json"
        self.config = json.load(open(cfg_path, encoding="utf-8"))
        self.id_map = self.config["phoneme_id_map"]
        self.speakers = self.config.get("speaker_id_map", {})
        self.sr = self.config.get("audio", {}).get("sample_rate", 22050)
        defaults = self.config.get("inference", {})
        self.defaults = dict(
            noise_scale=defaults.get("noise_scale", 0.667),
            length_scale=defaults.get("length_scale", 1.0),
            noise_w=defaults.get("noise_w", 0.8),
        )
        self.model_path = str(model_path)
        self.providers = list(providers)
        self._timed = None
        model = onnx.load(str(model_path))
        self._ceil_outputs = [n.output[0] for n in model.graph.node if n.op_type == "Ceil"]
        extra = [onnx.helper.make_empty_tensor_value_info(name) for name in self._ceil_outputs]
        model.graph.output.extend(extra)
        self.session = ort.InferenceSession(model.SerializeToString(), providers=list(providers))
        self.input_names = [i.name for i in self.session.get_inputs()]
        self._ceil_index = None

    def ids(self, text):
        """(ids, chars): the id sequence and, for each id, the character it stands for ('' for pad/bos/eos)."""
        text = _norm(text)
        pad, bos, eos = self.id_map.get("_", [0]), self.id_map.get("^", [1]), self.id_map.get("$", [2])
        ids, chars = list(bos) + list(pad), [""] * (len(bos) + len(pad))
        for ch in text:
            mapped = self.id_map.get(ch)
            if mapped is None:
                continue
            ids += list(mapped) + list(pad)
            chars += [ch] * len(mapped) + [""] * len(pad)
        ids += list(eos)
        chars += [""] * len(eos)
        return ids, chars

    def synth(self, text, speaker=None, noise_scale=None, length_scale=None, noise_w=None):
        """Return (audio float32 at self.sr, frames per id or None, chars per id)."""
        ids, chars = self.ids(text)
        p = dict(self.defaults)
        for k, v in (("noise_scale", noise_scale), ("length_scale", length_scale), ("noise_w", noise_w)):
            if v is not None:
                p[k] = v
        feeds = {}
        if "input" in self.input_names:
            feeds["input"] = np.array([ids], dtype=np.int64)
        if "input_lengths" in self.input_names:
            feeds["input_lengths"] = np.array([len(ids)], dtype=np.int64)
        if "scales" in self.input_names:
            feeds["scales"] = np.array([p["noise_scale"], p["length_scale"], p["noise_w"]], dtype=np.float32)
        if "sid" in self.input_names:
            sid = self.speakers.get(speaker, speaker if isinstance(speaker, int) else 0)
            feeds["sid"] = np.array([sid], dtype=np.int64)
        outs = self.session.run(None, feeds)
        audio = np.asarray(outs[0]).squeeze().astype(np.float32)
        frames = self._pick_durations(outs[1:], len(ids), len(audio))
        return audio, frames, chars

    def _timed_session(self):
        """A copy of the model that takes the duration of every input id as an extra input
        (`w_override`, frames, float, shape [1, 1, ids]) in place of its own predictor."""
        if self._timed is None:
            model = onnx.load(self.model_path)
            ceils = [n for n in model.graph.node if n.op_type == "Ceil"]
            if len(ceils) != 1:
                raise RuntimeError(f"expected one Ceil node, found {len(ceils)}")
            ceils[0].input[0] = "w_override"
            model.graph.input.append(onnx.helper.make_tensor_value_info("w_override", onnx.TensorProto.FLOAT, [1, 1, None]))
            self._timed = ort.InferenceSession(model.SerializeToString(), providers=self.providers)
        return self._timed

    def synth_timed(self, text, frames, speaker=None, noise_scale=None, noise_w=None):
        """Speak `text` with exactly `frames[i]` frames for input id i. Returns (audio, chars)."""
        ids, chars = self.ids(text)
        if len(frames) != len(ids):
            raise ValueError(f"{len(frames)} durations for {len(ids)} ids")
        p = dict(self.defaults)
        if noise_scale is not None:
            p["noise_scale"] = noise_scale
        if noise_w is not None:
            p["noise_w"] = noise_w
        sid = self.speakers.get(speaker, speaker if isinstance(speaker, int) else 0)
        feeds = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([p["noise_scale"], 1.0, p["noise_w"]], dtype=np.float32),
            "sid": np.array([sid], dtype=np.int64),
            "w_override": np.asarray(frames, dtype=np.float32).reshape(1, 1, -1),
        }
        out = self._timed_session().run(None, feeds)
        return np.asarray(out[0]).squeeze().astype(np.float32), chars

    def _pick_durations(self, extra, n_ids, n_samples):
        """The Ceil output that is one non-negative count per input id and adds up to the audio length."""
        if self._ceil_index is not None:
            return np.asarray(extra[self._ceil_index]).reshape(-1).astype(int)
        for i, t in enumerate(extra):
            a = np.asarray(t)
            if a.size == n_ids and a.min() >= 0 and abs(a.sum() * HOP - n_samples) <= 2 * HOP:
                self._ceil_index = i
                return a.reshape(-1).astype(int)
        return None


def char_spans(frames, chars, sr, hop=HOP):
    """Seconds spanned by each character: [(char, start, end), ...].

    The pad id after a character carries part of that character's sound (transitions), so it is
    counted with the character: a span runs from the character's own frames to the next character."""
    edges = np.concatenate([[0], np.cumsum(frames)]) * hop / sr
    idx = [i for i, c in enumerate(chars) if c]
    out = []
    for k, i in enumerate(idx):
        end = edges[idx[k + 1]] if k + 1 < len(idx) else edges[i + 2] if i + 2 < len(edges) else edges[i + 1]
        out.append((chars[i], float(edges[i]), float(end)))
    return out


def plan_frames(voice, word, speaker, carrier="ya ce {} ne", rate=1.0, lead=2, tail=3,
                noise_scale=0.2, min_char=2, min_vowel=7, min_final=11, long_at=(), min_long=13, min_long_final=17):
    """Per-id frame counts (whole frames, as the predictor itself gives) for the isolated `word`, taken
    from how the voice times it inside `carrier`.

    The predictor's own timing for a lone word is erratic (a 0.4 s closure before an initial k, a
    vowel of 11 ms); inside a sentence it is natural. Each character keeps its in-sentence frames
    (and the pad after it), scaled by `rate`, with a short lead-in and tail around the word. In a
    sentence the vowels are clipped short, too short to carry a tone, so a vowel gets at least
    `min_vowel` frames and a vowel that ends the word `min_final` (spoken alone it is held longer).
    The voice reads plain text, so it has no way to be told a vowel is long: `long_at` (character
    positions in the word, see `long_vowel_positions`) gets `min_long`, or `min_long_final` at the end.
    Returns (text, frames)."""
    prefix = carrier.split("{}")[0]
    text = carrier.format(word)
    _, chars = voice.ids(text)
    audio, frames, _ = voice.synth(text, speaker=speaker, noise_scale=noise_scale, noise_w=0.0, length_scale=1.0)
    if frames is None:
        raise RuntimeError("the voice did not return durations")
    first = len([c for c in voice.ids(prefix)[1] if c])
    n = len([c for c in voice.ids(word)[1] if c])
    out = [lead, 1]                                   # bos, its pad
    for j in range(n):
        i = 2 + 2 * (first + j)                       # id index of this character in the carrier text
        vowel = chars[i] in VOWEL_CHARS
        last = j == n - 1
        floor = (min_final if last else min_vowel) if vowel else min_char
        if vowel and j in long_at:
            floor = min_long_final if last else min_long
        c = max(floor, int(round(frames[i] * rate)))
        p = max(1, int(round(frames[i + 1] * rate)))
        out += [c, p]
    out.append(tail)                                  # eos
    return word, [int(x) for x in out]


VOWEL_CHARS = set("aeiou")
SONORANT_CODAS = set("mnlrwy")


def long_vowel_positions(sylls, longs, id_map):
    """Positions (among the word's characters as the voice reads them) of the vowel of every syllable
    marked long. `longs` has one boolean per syllable. A syllable with two vowel letters (ai, au) is
    left alone: a macron marks one vowel, and those are not short anyway."""
    pos, k = set(), 0
    for syl, long in zip(sylls, longs):
        chars = [c for c in _norm(syl) if c in id_map]
        vowels = [k + i for i, c in enumerate(chars) if c in VOWEL_CHARS]
        if long and len(vowels) == 1:
            pos.add(vowels[0])
        k += len(chars)
    return pos


def syllable_nuclei(sylls, spans, id_map):
    """The tone-bearing stretch of each syllable, in seconds, from the character spans.

    A syllable's nucleus runs from the start of its first vowel to the end of its last vowel, and on
    over an immediately following sonorant coda (m n l r w y), the same rule build_base.py applies to
    eSpeak's events. Returns None when the spans do not line up with the syllables."""
    per_syll = [[c for c in _norm(s) if c in id_map] for s in sylls]
    if sum(len(c) for c in per_syll) != len(spans):
        return None
    out, k = [], 0
    for chars in per_syll:
        mine = spans[k:k + len(chars)]
        k += len(chars)
        v = [i for i, (c, _, _) in enumerate(mine) if c in VOWEL_CHARS]
        if not v:
            return None
        a, b = mine[v[0]][1], mine[v[-1]][2]
        j = v[-1] + 1
        if j < len(mine) and mine[j][0] in SONORANT_CODAS:
            b = mine[j][2]
        out.append([a, b])
    return out
