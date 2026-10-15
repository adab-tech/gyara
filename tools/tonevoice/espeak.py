"""Minimal ctypes binding to libespeak-ng (the one shipped in the `espeakng-loader` wheel).

Synthesizes text and returns the audio together with per-phoneme timing events.
`prepare_data_dir` copies eSpeak's data and adds the custom `hx_flat1` voice (Swahili with the
stress contrast flattened), which is what the review voice is built from.
"""
import ctypes as C
import os
import shutil

import espeakng_loader
import numpy as np

AUDIO_OUTPUT_SYNCHRONOUS = 2
INITIALIZE_PHONEME_EVENTS = 0x0001
CHARS_UTF8 = 1
ENDPAUSE = 0x1000

RATE, VOLUME, PITCH, RANGE = 1, 2, 3, 4
EV_PHONEME = 7

# Swahili is the closest voice eSpeak ships to Hausa. Equal stress lengths and amplitudes stop it
# lengthening the penultimate vowel, which Hausa does not do.
FLAT_VOICE = "hx_flat1"
FLAT_VOICE_FILE = (
    "name hx_flat1\n"
    "language sw\n"
    "stressLength 150 150 150 150 0 0 150 150\n"
    "stressAmp 18 18 18 18 18 18 18 18\n"
)


def prepare_data_dir(dest):
    """Copy the wheel's eSpeak data to `dest` (once) and add the custom voice. Returns `dest`."""
    dest = os.path.abspath(dest)
    if not os.path.isdir(dest):
        shutil.copytree(espeakng_loader.get_data_path(), dest)
    voices = os.path.join(dest, "voices")
    os.makedirs(voices, exist_ok=True)
    with open(os.path.join(voices, FLAT_VOICE), "w", encoding="utf-8") as f:
        f.write(FLAT_VOICE_FILE)
    return dest


class _IdUnion(C.Union):
    _fields_ = [("number", C.c_int), ("name", C.c_char_p), ("string", C.c_char * 8)]


class _Event(C.Structure):
    _fields_ = [
        ("type", C.c_int),
        ("unique_identifier", C.c_uint),
        ("text_position", C.c_int),
        ("length", C.c_int),
        ("audio_position", C.c_int),
        ("sample", C.c_int),
        ("user_data", C.c_void_p),
        ("id", _IdUnion),
    ]


_CB = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_short), C.c_int, C.POINTER(_Event))


class Espeak:
    def __init__(self, data_dir=None):
        self.lib = C.CDLL(espeakng_loader.get_library_path())
        data = (data_dir or espeakng_loader.get_data_path()).encode()
        self.lib.espeak_Initialize.restype = C.c_int
        self.sr = self.lib.espeak_Initialize(
            AUDIO_OUTPUT_SYNCHRONOUS, 0, data, INITIALIZE_PHONEME_EVENTS
        )
        if self.sr <= 0:
            raise RuntimeError(f"espeak_Initialize failed: {self.sr}")
        self.lib.espeak_SetVoiceByName.argtypes = [C.c_char_p]
        self.lib.espeak_SetParameter.argtypes = [C.c_int, C.c_int, C.c_int]
        self.lib.espeak_Synth.argtypes = [
            C.c_char_p,
            C.c_size_t,
            C.c_uint,
            C.c_int,
            C.c_uint,
            C.c_uint,
            C.POINTER(C.c_uint),
            C.c_void_p,
        ]
        self._chunks = []
        self._events = []
        self._cb = _CB(self._callback)
        self.lib.espeak_SetSynthCallback(self._cb)

    def _callback(self, wav, n, events):
        if wav and n > 0:
            self._chunks.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
        i = 0
        while events[i].type != 0:
            e = events[i]
            name = None
            if e.type == EV_PHONEME:
                name = bytes(e.id.string).split(b"\0", 1)[0].decode("utf-8", "replace")
            self._events.append((e.type, e.audio_position, e.text_position, e.length, name))
            i += 1
        return 0

    def set_voice(self, name):
        r = self.lib.espeak_SetVoiceByName(name.encode())
        if r != 0:
            raise RuntimeError(f"voice {name!r}: error {r}")

    def set_param(self, p, v):
        self.lib.espeak_SetParameter(p, int(v), 0)

    def synth(self, text):
        """Return (int16 samples, events) for `text`; events are (type, ms, text_pos, len, name)."""
        self._chunks, self._events = [], []
        raw = text.encode("utf-8")
        r = self.lib.espeak_Synth(raw, len(raw) + 1, 0, 0, 0, CHARS_UTF8 | ENDPAUSE, None, None)
        if r != 0:
            raise RuntimeError(f"espeak_Synth error {r}")
        self.lib.espeak_Synchronize()
        x = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)
        return x, list(self._events)
