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

With Murya's Piper voice instead of eSpeak (needs `onnx`, `onnxruntime`; the model is a download, not in the repo):

```
python build_neural_base.py words.json --model model.onnx --speaker F1 --out base-f1
python export_voice.py --base base-f1 --out voice.json --center voice --level -16
python verify.py --voice voice.json --all
```

`--center voice` puts every word of the voice on one pitch (the median of the recordings'), so words
do not wander with whatever pitch the voice chose for each. `--level` gives every word the same
loudness, and the page keeps it (`ToneVoice.render(base, tones, {keepLevel: true})`) instead of
peak-normalising each clip. A VITS voice draws random noise, so rebuilding the recordings gives
slightly different audio each time; the page embeds the recordings, not the model.

`words.json` is a list of `{id, n, word, gloss, pos, marked, tones, sylls, newman, old, others}`;
see `example-words.json`. `marked` is the tone-marked spelling (a macron makes the vowel long in the
recording), `sylls` the syllables, `tones` one of H/L/F per syllable, `newman` a list of
`{marked, tones, gloss}`. The 50-word data used for the first review is not in the repo.

`verify.py` reads the `voice.json` the page embeds, measures pitch from the rendered audio period
by period and fails if any syllable is off its target. On the first 50 words: 86 renders, 205 syllables, worst deviation 0.1 semitone.
Across every H/L/F pattern of the 49 recordings, 2,196 syllables, the only misses were three
renders of a fall on the first syllable of `barka`, where the weak `r` gives the tracker too few
pulses to measure.

## Limits

- The base voice is eSpeak NG with Swahili settings (`espeak.py`). It is a formant synthesizer, so
  it sounds robotic, and it only approximates ɓ ɗ ƙ ƴ, c, j and r. The tones are right; the voice
  is not good to listen to for long. Reviewers said as much.
- Better base voice: Murya's own Piper model (`adab-tech/murya-piper-hausa-tts`, CC-BY-NC-SA), run
  as Murya's backend runs it, with no tone marks, at its settings (noise 0.667, pace 1.25). A lone
  word makes its duration predictor erratic (a 0.4 s closure before an initial k, a vowel of 11 ms),
  so `build_neural_base.py` first times the word inside a carrier sentence (`ya ce <word> ne`), then
  speaks it alone with those durations, slowed by `--rate`, with a floor on vowel length and longer
  vowels where the marked spelling has a macron. `piper_base.py` does this by exposing the `Ceil`
  node (frames per input id) and by swapping its input for one of its own, which is how the durations
  are imposed. Because they are known, so is where each vowel is: a syllable's nucleus is its vowel
  (and a sonorant coda) as timed, cut to the part within 20 dB of its peak (`nuclei.audible`; the
  last vowel fades out over tens of milliseconds, and a fall spread over the fade would land where
  nobody hears it) and to the voiced stretch. `nuclei.detect` finds vowels from the audio alone and
  is the fallback when the spelling and the timing disagree. The voice's own pitch drifts down across
  a word, and the further the engine has to move it the rougher the result, so `--candidates` takes
  are drawn (the noise makes each one different) and the one whose own pitch strays least inside the
  vowels, with the clearest voice and the most regular pulses, is kept.
  Checked on four female speakers (F1 to F4) and the 49 recordings of the first review: every H/L/F
  pattern of every word, 801 renders and 2,196 syllables per voice, no failures in three voices and one
  in the fourth (a syllable whose quiet fade the tracker mis-measured; the pulses are on target to
  the last audible period), worst deviation 1.1 semitones. The harmonicity of the renders stays
  within a few dB of the recordings and no clicks appear. The Piper wrapper's unit tests use a
  stand-in ONNX graph.
- Pitch marks: `repair_marks` fills dropped pulses by comparing each gap with the local period. It
  once used one period for the whole word, which is fine for a monotone voice and wrong for one
  whose pitch falls by half an octave inside a word: every long period in the low stretch got a
  bogus extra mark (up to half of the periods in some words), a double pulse heard as buzz.
- Pitch is set by rule: flat plateaus, no downdrift, no consonant effects on pitch.
- Nothing here was checked by ear. Pitch is measured; naturalness is not.

## Files

| File | Purpose |
|---|---|
| `espeak.py` | ctypes binding to libespeak-ng, plus the flat Swahili voice |
| `g2p.py` | Hausa spelling to Swahili-style spelling, vowel lengths from the marked form |
| `build_base.py` | words to monotone eSpeak recordings, syllable spans, pitch marks |
| `build_neural_base.py` | the same from a Piper voice: carrier-timed durations, nuclei from the timing |
| `piper_base.py` | runs a Piper ONNX voice as Murya's backend does, recovers per-character timing, speaks with chosen durations |
| `nuclei.py` | finds each syllable's voiced vowel stretch in a recording; cuts a span to its audible part |
| `marks.py` | glottal pulse marks and their clean-up |
| `export_voice.py` | recordings to the compact `voice.json`, optionally on one pitch and one loudness |
| `tonevoice.py`, `tonevoice.js` | the TD-PSOLA tone engine (reference and browser) |
| `node_render.js` | renders a list of patterns with the JS engine |
| `verify.py` | measures rendered pitch (from `voice.json`) against the intended pattern |
| `tests/` | pytest: Piper wrapper on a stand-in graph, nuclei, export |
| `review/` | the review page template, its builder and a browser test |
