// Dependency-free check of the BUILT package against the golden vectors, so CI can run it on
// every supported Node version (the test runner itself needs Node >= 22.12).
//   npm run build && node test/smoke.mjs
import { readFileSync } from "node:fs";
import * as g from "../dist/index.js";

const V = JSON.parse(readFileSync(new URL("../../tests/vectors.json", import.meta.url), "utf-8"));
const checks = {
  normalize: (t) => g.normalize(t),
  normalize_keep_digits: (t) => g.normalize(t, { digits: false }),
  cardinal: (n) => g.hausaCardinal(n),
  spell: (t) => g.spellOutNumbers(t),
  glosses: (t) => g.stripEnglishGlosses(t),
  speech: (t) => g.prepareForSpeech(t),
  speech_keep_glosses: (t) => g.prepareForSpeech(t, { stripGlosses: false }),
  syllables: (w) => g.syllabify(w).map((s) => [s.text, s.weight]),
  tone: (t) => g.annotateTones(t),
  tone_guess: (t) => g.annotateTones(t, { guess: true }),
};

let total = 0;
const bad = [];
for (const [name, fn] of Object.entries(checks)) {
  for (const c of V[name]) {
    total++;
    const got = JSON.stringify(fn(c.in));
    if (got !== JSON.stringify(c.out)) bad.push({ name, input: c.in, want: c.out, got: fn(c.in) });
  }
}
if (bad.length) {
  console.error(`${bad.length} of ${total} vectors differ from Python; first:`, bad[0]);
  process.exit(1);
}
console.log(`node ${process.versions.node}: ${total} vectors OK`);
