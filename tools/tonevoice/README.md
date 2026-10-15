# tonevoice

Makes Hausa words audible with their tones, so a native speaker can judge a tone pattern by ear.
Experimental. Lives outside the `gyara` package: nothing here is imported by it, and CI does not
build or lint this folder.

## What it does

Hausa tone is pitch, and ordinary TTS voices (Murya's included) read text without tone marks.
So the pitch is set afterwards:

1. `build_base.py` synthesizes each word as a monotone recording and finds each syllable's voiced
   vowel stretch and the glottal pulses (the "pitch marks").
2. `tonevoice.js` / `tonevoice.py` take one recording, a pattern such as `L L H` or `L F`, and
   re-time its pulses with TD-PSOLA. H is a high plateau, L a low plateau, F a glide from high to
   low inside the syllable. High and low are 6.5 semitones apart, centred on the recording's own pitch.
3. `review/` is a page that plays any pattern of any word in the browser. The JavaScript engine
   needs only the flat recording, its pulse positions and the syllable spans, so any base voice works.

The Python and JavaScript engines are the same algorithm. The Python one is the reference.

## Run it

```
pip install -r requirements.txt            # plus node 18+ for verify.py and the page
python build_base.py example-words.json    # base/<key>.wav and .json
python export_voice.py                     # voice.json (what the page embeds)
python verify.py --words example-words.json    # or --all for every H/L/F pattern
python review/build.py example-words.json  # review/out/index.html
```

`words.json` is a list of `{id, n, word, gloss, pos, marked, tones, sylls, newman, old, others}`;
see `example-words.json`. `marked` is the tone-marked spelling (a macron makes the vowel long in the
recording), `sylls` the syllables, `tones` one of H/L/F per syllable, `newman` a list of
`{marked, tones, gloss}`. The 50-word data used for the first review is not in the repo.

`verify.py` measures pitch from the rendered audio period by period and fails if any syllable is
off its target. On the first 50 words: 86 renders, 205 syllables, worst deviation 0.1 semitone.
Across every H/L/F pattern of the 49 recordings, 2,196 syllables, the only misses were three
renders of a fall on the first syllable of `barka`, where the weak `r` gives the tracker too few
pulses to measure.

## Limits

- The base voice is eSpeak NG with Swahili settings (`espeak.py`). It is a formant synthesizer, so
  it sounds robotic, and it only approximates ɓ ɗ ƙ ƴ, c, j and r. The tones are right; the voice
  is not good to listen to for long. Reviewers said as much.
- Better base voice: Murya's own Piper model (`adab-tech/murya-piper-hausa-tts`, CC-BY-NC-SA).
  `build_neural_base.py` is written for it and not yet run on the real model. It speaks each word as
  plain text, takes rough syllable positions from the duration predictor (`piper_base.py` adds the
  `Ceil` node's output to the ONNX graph), and places each vowel's voiced stretch from the audio
  (`nuclei.py`). The finder was tested on eSpeak recordings with known timings: with hints 40 ms off,
  97 to 100 percent of syllables land in the right place (88 to 98 percent cover at least half of the
  true vowel); without hints 91 to 99 percent. The Piper wrapper is tested against a stand-in ONNX
  graph only.
- Pitch is set by rule: flat plateaus, no downdrift, no consonant effects on pitch.
- Nothing here was checked by ear. Pitch is measured; naturalness is not.

## Files

| File | Purpose |
|---|---|
| `espeak.py` | ctypes binding to libespeak-ng, plus the flat Swahili voice |
| `g2p.py` | Hausa spelling to Swahili-style spelling, vowel lengths from the marked form |
| `build_base.py` | words to monotone eSpeak recordings, syllable spans, pitch marks |
| `build_neural_base.py` | the same from a Piper voice, spans from its durations and the audio |
| `piper_base.py` | runs a Piper ONNX voice as Murya's backend does and recovers per-character timing |
| `nuclei.py` | finds each syllable's voiced vowel stretch in a recording |
| `marks.py` | glottal pulse marks and their clean-up |
| `export_voice.py` | recordings to the compact `voice.json` |
| `tonevoice.py`, `tonevoice.js` | the TD-PSOLA tone engine (reference and browser) |
| `node_render.js` | renders a list of patterns with the JS engine |
| `verify.py` | measures rendered pitch against the intended pattern |
| `review/` | the review page template, its builder and a browser test |
