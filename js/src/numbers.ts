/** Standard Hausa cardinal numbers and number spelling for speech. Mirrors gyara/numbers.py. */

const UNITS = ["", "ɗaya", "biyu", "uku", "huɗu", "biyar", "shida", "bakwai", "takwas", "tara"];
const TENS: Record<number, string> = {
  1: "goma", 2: "ashirin", 3: "talatin", 4: "arba'in", 5: "hamsin",
  6: "sittin", 7: "saba'in", 8: "tamanin", 9: "casa'in",
};

const unit = (n: number): string => UNITS[n] as string;

function below100(n: number): string {
  if (n < 10) return unit(n);
  if (n < 20) {
    const r = n - 10;
    return r === 0 ? "goma" : `goma sha ${unit(r)}`;
  }
  const t = Math.floor(n / 10);
  const r = n % 10;
  return r === 0 ? (TENS[t] as string) : `${TENS[t]} da ${unit(r)}`;
}

function below1000(n: number): string {
  if (n < 100) return below100(n);
  const h = Math.floor(n / 100);
  const r = n % 100;
  const hundreds = h === 1 ? "ɗari" : `ɗari ${unit(h)}`;
  return r === 0 ? hundreds : `${hundreds} da ${below100(r)}`;
}

function digits(s: string): string {
  return [...s].map((d) => (d === "0" ? "sifili" : unit(Number(d)))).join(" ");
}

/** Non-negative integer -> Hausa words. Numbers >= 10^9 are read digit by digit. */
export function hausaCardinal(n: number): string {
  if (!Number.isInteger(n) || n < 0) {
    throw new RangeError("hausaCardinal expects a non-negative integer");
  }
  if (n === 0) return "sifili";
  if (n >= 1_000_000_000) return digits(String(n));
  const parts: string[] = [];
  if (n >= 1_000_000) {
    const m = Math.floor(n / 1_000_000);
    n %= 1_000_000;
    parts.push(m === 1 ? "miliyan" : `miliyan ${below1000(m)}`);
  }
  if (n >= 1000) {
    const th = Math.floor(n / 1000);
    n %= 1000;
    parts.push(th === 1 ? "dubu" : `dubu ${below1000(th)}`);
  }
  if (n > 0) parts.push(below1000(n));
  return parts.join(" da ");
}

// Digit strings of 10+ digits are always >= 10^9, so they are read digit by digit; this also
// avoids losing precision on huge numbers.
function cardinalOfDigits(s: string): string {
  return s.length >= 10 ? digits(s) : hausaCardinal(Number(s));
}

function token(t: string): string {
  const dot = t.indexOf(".");
  if (dot >= 0) {
    const whole = t.slice(0, dot);
    const frac = t.slice(dot + 1);
    let words: string;
    if (!whole) words = "sifili";
    else if (whole.startsWith("0") && whole.length > 1) words = digits(whole);
    else words = cardinalOfDigits(whole);
    return `${words} digo ${digits(frac)}`;
  }
  // A leading zero (phone numbers, codes) is read digit by digit.
  if (t.startsWith("0") && t.length > 1) return digits(t);
  return cardinalOfDigits(t);
}

/**
 * Replace ASCII digit runs in `text` with Hausa words (thousands commas folded).
 * Foreign digit glyphs are not matched: run `normalize` first (prepareForSpeech does).
 */
export function spellOutNumbers(text: string): string {
  const folded = text.replace(/(?<=\d),(?=\d{3}\b)/g, "");
  return folded.replace(/(?<![A-Za-z\d])\d+(?:\.\d+)?(?![A-Za-z\d])/g, (m) => token(m));
}
