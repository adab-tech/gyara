/**
 * Heuristic tone annotation for Hausa. Mirrors gyara/tone.py.
 *
 * Standard Hausa spelling does not mark tone, so this is a heuristic: a curated lexicon of common
 * words, a few plural-suffix melodies, and an optional low-confidence guess for everything else.
 * Unknown words are left unmarked unless `guess` is set. Melodies are aligned to syllables
 * right-to-left; the leftmost tone spreads over any extra syllables.
 */
import { LEXICON } from "./lexicon.js";
import { syllabify } from "./syllables.js";

export type Tone = "H" | "L";

const ACUTE = "́"; // high
const GRAVE = "̀"; // low

const PLURAL_SUFFIXES: Array<[readonly string[], readonly Tone[]]> = [
  [["una"], ["H", "L", "L"]],
  [["oshi", "oshe"], ["H", "L", "H"]],
  [["aye", "ayi"], ["L", "H", "L"]],
  [["oci", "otsi"], ["H", "L", "H"]],
];
const AI_PLURAL = /[bdfghjklmnrstwyzɓɗƙƴ']ai$/u;
const LETTERS = "[\\p{L}\\p{N}_ƁƊƘƳɓɗƙƴ'’ʼ]";
const NOT_LETTERS = "[^\\p{L}\\p{N}_ƁƊƘƳɓɗƙƴ'’ʼ]";
const WORD_RE = new RegExp(`^(${NOT_LETTERS}*)(${LETTERS}+)(${NOT_LETTERS}*)$`, "u");

/** Tone melody for a word, or null if unknown. */
export function melodyFor(word: string, options: { guess?: boolean } = {}): readonly Tone[] | null {
  const low = word.toLowerCase();
  const known = LEXICON.get(low);
  if (known) return known;
  for (const [suffixes, melody] of PLURAL_SUFFIXES) {
    if (suffixes.some((s) => low.endsWith(s))) return melody;
  }
  if (AI_PLURAL.test(low)) return ["L", "H", "H"];
  if (options.guess) {
    const sylls = syllabify(low);
    const last = sylls[sylls.length - 1];
    if (last) return last.weight === "heavy" ? ["L", "H"] : ["H", "L"];
  }
  return null;
}

function align(melody: readonly Tone[], n: number): Tone[] {
  if (n <= melody.length) return melody.slice(melody.length - n);
  return [...Array<Tone>(n - melody.length).fill(melody[0] as Tone), ...melody];
}

function mark(syllable: string, tone: Tone): string {
  const m = /[aeiouAEIOU]/.exec(syllable);
  if (!m) return syllable;
  const end = m.index + m[0].length;
  return syllable.slice(0, end) + (tone === "H" ? ACUTE : GRAVE) + syllable.slice(end);
}

/** Mark known words with acute (H) / grave (L) accents on each syllable. */
export function annotateTones(text: string, options: { guess?: boolean } = {}): string {
  const out: string[] = [];
  for (const tok of text.split(/\s+/).filter(Boolean)) {
    const m = WORD_RE.exec(tok);
    if (!m) {
      out.push(tok);
      continue;
    }
    const [, left = "", word = "", right = ""] = m;
    const sylls = syllabify(word);
    const melody = melodyFor(word, options);
    if (sylls.length === 0 || melody === null) {
      out.push(tok);
      continue;
    }
    const tones = align(melody, sylls.length);
    const marked = sylls.map((s, i) => mark(s.text, tones[i] as Tone)).join("");
    out.push(left + marked.normalize("NFC") + right);
  }
  return out.join(" ");
}
