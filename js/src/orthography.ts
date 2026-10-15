/**
 * Hausa orthography normalization.
 *
 * Restores the hooked consonants (ɓ ɗ ƙ ƴ) from the ASCII apostrophe spellings people
 * actually type (b' d' k' y' 'y), preserving case, and folds foreign digit glyphs to ASCII.
 * Mirrors gyara/orthography.py; the shared vectors in tests/vectors.json keep them in sync.
 */

export const HOOKED_LETTERS = "ɓɗƙƴƁƊƘƳ";

const HOOKED_LOWER: Record<string, string> = { b: "ɓ", d: "ɗ", k: "ƙ", y: "ƴ" };
const HOOKED_UPPER: Record<string, string> = { B: "Ɓ", D: "Ɗ", K: "Ƙ", Y: "Ƴ" };

// ASCII apostrophe plus the typographic variants phones and word processors insert.
const APOS = "'’ʼ";
const LETTER = "\\p{L}";

// A hook is only restored when a letter follows the apostrophe: Hausa words do not end in
// ɓ ɗ ƙ ƴ, so "k'" at the end of a word is a closing quote. 'y is a hook only at the START of a
// word (inside a word it is a glottal stop plus y: ya'ya), and quote-wrapped words like 'yaro'
// are left alone.
const hookSource =
  `([bdkyBDKY])[${APOS}](?=${LETTER})` +
  `|(?<!${LETTER})[${APOS}]([yY])(?=${LETTER})(?!${LETTER}*[${APOS}](?!${LETTER}))`;
const tsSource = `([tT][sS])[${APOS}](?=${LETTER})`;
const quotedSource = `(?<!${LETTER})[${APOS}][yY]${LETTER}*[${APOS}](?!${LETTER})`;

const foreignDigits = new Map<string, string>();
for (let i = 0; i < 10; i++) {
  foreignDigits.set(String.fromCharCode(0x0660 + i), String(i)); // Arabic-Indic
  foreignDigits.set(String.fromCharCode(0x06f0 + i), String(i)); // Persian
}
export const FOREIGN_DIGIT_RE = /[٠-٩۰-۹]/g;

/** Convert Arabic-Indic / Persian digit glyphs to ASCII 0-9. */
export function normalizeDigits(text: string): string {
  if (!text) return text;
  return text.replace(FOREIGN_DIGIT_RE, (c) => foreignDigits.get(c) ?? c);
}

/**
 * Normalize ASCII-apostrophe Hausa spellings to standard Unicode.
 *
 * normalize("k'asa da b'aki") === "ƙasa da ɓaki"
 */
export function normalize(text: string, options: { digits?: boolean } = {}): string {
  if (!text) return text;
  let out = options.digits === false ? text : normalizeDigits(text);
  out = out.replace(new RegExp(tsSource, "gu"), "$1");
  return out.replace(new RegExp(hookSource, "gu"), (_m, a?: string, b?: string) => {
    const letter = (a ?? b) as string;
    return letter === letter.toUpperCase()
      ? (HOOKED_UPPER[letter] as string)
      : (HOOKED_LOWER[letter.toLowerCase()] as string);
  });
}

/** Number of proper hooked letters (ɓ ɗ ƙ ƴ, any case) in `text`. */
export function countHooked(text: string): number {
  let n = 0;
  for (const ch of text) if (HOOKED_LETTERS.includes(ch)) n++;
  return n;
}

/** Number of apostrophe spellings (b' d' k' y' 'y) that `normalize` would fix. */
export function countAsciiHookCandidates(text: string): number {
  return [...text.matchAll(new RegExp(hookSource, "gu"))].length;
}

/** Number of quote-wrapped words like 'yaro' that `normalize` deliberately leaves alone. */
export function countQuotedLookalikes(text: string): number {
  return [...text.matchAll(new RegExp(quotedSource, "gu"))].length;
}
