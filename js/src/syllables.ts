/** Hausa syllabification with syllable weight (light CV, heavy CVV / CVC). Mirrors syllables.py. */

export interface Syllable {
  text: string;
  weight: "light" | "heavy";
}

const CLEAN_RE = /[^\p{L}\p{N}_ƁƊƘƳɓɗƙƴ'’ʼ]/gu;

type Kind = "C" | "VL" | "VS";
const PATTERNS: Array<[RegExp, Kind]> = [
  [/(?:ts|sh|ch|gy|ky|ƙy|fy|gw|kw|ƙw|hw)/uy, "C"],
  [/[ɓɗƙƴbdfghjklmnrstwyz'’ʼ]/uy, "C"],
  [/(?:aa|ee|ii|oo|uu|ai|au)/uy, "VL"],
  [/[aeiou]/uy, "VS"],
];

function tokenize(word: string): Array<[string, Kind]> {
  const low = word.toLowerCase();
  const tokens: Array<[string, Kind]> = [];
  let i = 0;
  while (i < low.length) {
    let matched = false;
    for (const [re, kind] of PATTERNS) {
      re.lastIndex = i;
      const m = re.exec(low);
      if (m) {
        tokens.push([word.slice(i, i + m[0].length), kind]);
        i += m[0].length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      // Unknown symbol: treat as a consonant so nothing is dropped.
      const cp = word.codePointAt(i) as number;
      const ch = String.fromCodePoint(cp);
      tokens.push([ch, "C"]);
      i += ch.length;
    }
  }
  return tokens;
}

/** Split one Hausa word into syllables. */
export function syllabify(word: string): Syllable[] {
  const clean = word.replace(CLEAN_RE, "");
  if (!clean) return [];
  const tokens = tokenize(clean);
  const n = tokens.length;
  const out: Syllable[] = [];
  let j = 0;
  while (j < n) {
    let onset = "";
    let nucleus = "";
    let coda = "";
    let longVowel = false;
    if (tokens[j]![1] === "C") {
      onset = tokens[j]![0];
      j++;
    }
    if (j < n && (tokens[j]![1] === "VS" || tokens[j]![1] === "VL")) {
      nucleus = tokens[j]![0];
      longVowel = tokens[j]![1] === "VL";
      j++;
    }
    // A consonant closes this syllable if it ends the word or precedes another consonant.
    if (j < n && tokens[j]![1] === "C" && (j === n - 1 || tokens[j + 1]![1] === "C")) {
      coda = tokens[j]![0];
      j++;
    }
    const text = onset + nucleus + coda;
    if (text) out.push({ text, weight: longVowel || coda ? "heavy" : "light" });
  }
  return out;
}
