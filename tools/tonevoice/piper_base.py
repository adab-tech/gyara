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
        model = onnx.load(str(model_path))
        self._ceil_outputs = [n.output[0] for n in model.graph.node if n.op_type == "Ceil"]
        extra = [onnx.helper.make_empty_tensor_value_info(name) for name in self._ceil_outputs]
        model.graph.output.extend(extra)
        self.session = ort.InferenceSession(model.SerializeToString(), providers=list(providers))
        self.input_names = [i.name for i in self.session.get_inputs()]
        self._ceil_index = None

    def ids(self, text):
        """(ids, chars): the id sequence and, for each id, the character it stands for ('' for pad/bos/eos)."""
        text = unicodedata.normalize("NFD", text).casefold()
        for a, b in _FIXES.items():
            text = text.replace(a, b)
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
    """Seconds spanned by each character: [(char, start, end), ...] (pad ids are folded into no character)."""
    edges = np.concatenate([[0], np.cumsum(frames)]) * hop / sr
    return [(c, float(edges[i]), float(edges[i + 1])) for i, c in enumerate(chars) if c]
