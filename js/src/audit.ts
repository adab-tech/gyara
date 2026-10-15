/** Text audit and dataset statistics for Hausa corpora. Mirrors gyara/audit.py. */
import {
  FOREIGN_DIGIT_RE,
  countAsciiHookCandidates,
  countHooked,
  countQuotedLookalikes,
} from "./orthography.js";

const WORD_RE = /[\p{L}\p{N}_ƁƊƘƳɓɗƙƴ]+(?:['’ʼ][\p{L}\p{N}_ƁƊƘƳɓɗƙƴ]+)*/gu;
const DIGIT_RUN_RE = /\p{Nd}+/gu;

export function words(text: string): string[] {
  return text.toLowerCase().match(WORD_RE) ?? [];
}

export interface AuditReport {
  words: number;
  hookedLetters: number;
  asciiHookSpellings: number;
  foreignDigits: number;
  asciiDigitRuns: number;
  issues: string[];
  warnings: string[];
  ok: boolean;
}

/** Check `text` for orthography problems `normalize` can repair. */
export function auditText(text: string): AuditReport {
  const asciiHooks = countAsciiHookCandidates(text);
  const foreign = (text.match(FOREIGN_DIGIT_RE) ?? []).length;
  const issues: string[] = [];
  if (asciiHooks) {
    issues.push(`${asciiHooks} ASCII-apostrophe hooked-letter spelling(s) (b' d' k' y' 'y)`);
  }
  if (foreign) {
    issues.push(`${foreign} Arabic-Indic/Persian digit glyph(s); Hausa uses 0-9`);
  }
  const warnings: string[] = [];
  const quoted = countQuotedLookalikes(text);
  if (quoted) {
    warnings.push(
      `${quoted} quote-wrapped word(s) like 'yaro' left as-is; ` +
        "check by hand whether an opening 'y was meant to be ƴ",
    );
  }
  return {
    words: words(text).length,
    hookedLetters: countHooked(text),
    asciiHookSpellings: asciiHooks,
    foreignDigits: foreign,
    asciiDigitRuns: (text.match(DIGIT_RUN_RE) ?? []).length,
    issues,
    warnings,
    ok: issues.length === 0,
  };
}

export interface DatasetStats {
  lines: number;
  emptyLines: number;
  duplicateLines: number;
  linesWithDigits: number;
  tokens: number;
  types: number;
  typeTokenRatio: number;
  wordsPerLine: { min: number; max: number; mean: number };
  hookedLetters: number;
  asciiHookSpellings: number;
}

const round = (x: number, places: number): number => {
  const f = 10 ** places;
  return Math.round(x * f) / f;
};

/** Summary statistics for a corpus with one sentence per line. */
export function datasetStats(lines: Iterable<string>): DatasetStats {
  let total = 0;
  let empty = 0;
  let withDigits = 0;
  let asciiHooks = 0;
  let hooked = 0;
  let tokens = 0;
  let duplicates = 0;
  const lengths: number[] = [];
  const types = new Set<string>();
  const seen = new Set<string>();
  for (const raw of lines) {
    total++;
    const line = raw.trim();
    if (!line) {
      empty++;
      continue;
    }
    if (seen.has(line)) duplicates++;
    seen.add(line);
    const ws = words(line);
    lengths.push(ws.length);
    tokens += ws.length;
    for (const w of ws) types.add(w);
    asciiHooks += countAsciiHookCandidates(line);
    hooked += countHooked(line);
    if (/\p{Nd}/u.test(line)) withDigits++;
  }
  return {
    lines: total,
    emptyLines: empty,
    duplicateLines: duplicates,
    linesWithDigits: withDigits,
    tokens,
    types: types.size,
    typeTokenRatio: tokens ? round(types.size / tokens, 4) : 0,
    wordsPerLine: {
      min: lengths.length ? Math.min(...lengths) : 0,
      max: lengths.length ? Math.max(...lengths) : 0,
      mean: lengths.length ? round(lengths.reduce((a, b) => a + b, 0) / lengths.length, 2) : 0,
    },
    hookedLetters: hooked,
    asciiHookSpellings: asciiHooks,
  };
}
