/**
 * Text preparation for Hausa speech synthesis. Mirrors gyara/speech.py.
 *
 * A TTS voice trained on Hausa usually has a small symbol inventory (letters, punctuation,
 * ɓ ɗ ƙ ƴ). Anything outside it is silently dropped, which loses meaning ("₦500" read as just
 * "ɗari biyar"), glues words together ("maka/miki" -> "makamiki") or hits control symbols.
 */
import { HOOKED_LETTERS, normalize } from "./orthography.js";
import { spellOutNumbers } from "./numbers.js";

/** Currency-first matches Hausa word order: "naira dubu goma" = ten thousand naira. */
export const CURRENCY_WORDS: ReadonlyArray<readonly [string, string]> = [
  ["₦", " naira "],
  ["$", " dala "],
  ["€", " yuro "],
  ["£", " fam "],
];

const HAUSA_ASIDES = new Set([
  "misali", "wato", "watau", "kamar", "kuma", "gani", "duba", "watakila",
  "wajen", "ma", "dss", "sauransu", "haka", "shi", "ita",
]);
const PAREN_RE = /\s*\(([^()]{1,40})\)/g;
const ASCII_GLOSS_RE = /^[A-Za-z][A-Za-z /'-]*$/;

/** Remove short English-gloss parentheticals; keep Hausa asides. */
export function stripEnglishGlosses(text: string): string {
  return text.replace(PAREN_RE, (whole: string, group: string) => {
    const inner = group.trim();
    const words = inner.split(/\s+/).filter(Boolean);
    if ([...inner].some((c) => HOOKED_LETTERS.includes(c))) return whole;
    if (words.length > 3) return whole;
    if (words.some((w) => HAUSA_ASIDES.has(w.toLowerCase().replace(/^[.,]+|[.,]+$/g, "")))) {
      return whole;
    }
    return ASCII_GLOSS_RE.test(inner) ? "" : whole;
  });
}

/**
 * Full normalization pipeline for a TTS front end. Order matters: hooked letters, glosses,
 * currency, percent, numbers, separators, markup, whitespace.
 */
export function prepareForSpeech(text: string, options: { stripGlosses?: boolean } = {}): string {
  let out = normalize(text);
  if (options.stripGlosses !== false) out = stripEnglishGlosses(out);
  for (const [symbol, word] of CURRENCY_WORDS) out = out.split(symbol).join(word);
  out = out.replace(/(?<=\d)\s*%/g, " cikin ɗari");
  out = spellOutNumbers(out);
  out = out.replace(/(?<=[\p{L}\p{N}_])\/(?=[\p{L}\p{N}_])/gu, " ko "); // maka/miki -> maka ko miki
  out = out.split("@").join(" a ");
  out = out.replace(/[*#_^~|<>[\]{}()"“”„]/g, " "); // markup -> space, never deleted
  return out.replace(/\s+/g, " ").trim();
}
