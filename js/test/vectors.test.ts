import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  annotateTones,
  auditText,
  datasetStats,
  hausaCardinal,
  normalize,
  prepareForSpeech,
  spellOutNumbers,
  stripEnglishGlosses,
  syllabify,
} from "../src/index.js";
import { LEXICON } from "../src/lexicon.js";

// Golden vectors generated from the Python implementation (scripts/gen_vectors.py).
const V = JSON.parse(
  readFileSync(new URL("../../tests/vectors.json", import.meta.url), "utf-8"),
) as Record<string, any>;

type Case<I, O> = { in: I; out: O };

function each<I, O>(name: string, fn: (input: I) => O): void {
  describe(`vectors: ${name}`, () => {
    const cases = V[name] as Case<I, O>[];
    it(`has cases`, () => expect(cases.length).toBeGreaterThan(0));
    it(`matches Python on all ${cases.length} cases`, () => {
      const failures = cases
        .map((c) => ({ input: c.in, want: c.out, got: fn(c.in) }))
        .filter((r) => JSON.stringify(r.got) !== JSON.stringify(r.want));
      expect(failures.slice(0, 5)).toEqual([]);
    });
  });
}

each("normalize", (t: string) => normalize(t));
each("normalize_keep_digits", (t: string) => normalize(t, { digits: false }));
each("cardinal", (n: number) => hausaCardinal(n));
each("spell", (t: string) => spellOutNumbers(t));
each("glosses", (t: string) => stripEnglishGlosses(t));
each("speech", (t: string) => prepareForSpeech(t));
each("speech_keep_glosses", (t: string) => prepareForSpeech(t, { stripGlosses: false }));
each("syllables", (w: string) => syllabify(w).map((s) => [s.text, s.weight]));
each("tone", (t: string) => annotateTones(t));
each("tone_guess", (t: string) => annotateTones(t, { guess: true }));

describe("vectors: audit", () => {
  it("matches Python", () => {
    const failures = (V.audit as Case<string, any>[])
      .map(({ in: input, out: want }) => {
        const r = auditText(input);
        const got = {
          words: r.words,
          hooked_letters: r.hookedLetters,
          ascii_hook_spellings: r.asciiHookSpellings,
          foreign_digits: r.foreignDigits,
          ascii_digit_runs: r.asciiDigitRuns,
          issues: r.issues,
          warnings: r.warnings,
          ok: r.ok,
        };
        return { input, want, got };
      })
      .filter((r) => JSON.stringify(r.got) !== JSON.stringify(r.want));
    expect(failures.slice(0, 5)).toEqual([]);
  });
});

describe("vectors: stats", () => {
  it("matches Python", () => {
    for (const { in: lines, out: want } of V.stats as Case<string[], any>[]) {
      const g = datasetStats(lines);
      expect(g.lines).toBe(want.lines);
      expect(g.emptyLines).toBe(want.empty_lines);
      expect(g.duplicateLines).toBe(want.duplicate_lines);
      expect(g.linesWithDigits).toBe(want.lines_with_digits);
      expect(g.tokens).toBe(want.tokens);
      expect(g.types).toBe(want.types);
      expect(g.typeTokenRatio).toBeCloseTo(want.type_token_ratio, 4);
      expect(g.wordsPerLine.min).toBe(want.words_per_line.min);
      expect(g.wordsPerLine.max).toBe(want.words_per_line.max);
      expect(g.wordsPerLine.mean).toBeCloseTo(want.words_per_line.mean, 2);
      expect(g.hookedLetters).toBe(want.hooked_letters);
      expect(g.asciiHookSpellings).toBe(want.ascii_hook_spellings);
    }
  });
});

describe("vectors: lexicon", () => {
  it("is identical to the Python lexicon", () => {
    expect(Object.fromEntries(LEXICON)).toEqual(V.lexicon);
  });
});
