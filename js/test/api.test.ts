import { describe, expect, it } from "vitest";
import {
  annotateTones,
  auditText,
  hausaCardinal,
  normalize,
  prepareForSpeech,
  spellOutNumbers,
  syllabify,
} from "../src/index.js";

describe("normalize", () => {
  it("restores hooked letters and keeps case", () => {
    expect(normalize("k'asa da b'aki")).toBe("ƙasa da ɓaki");
    expect(normalize("'Yanci")).toBe("Ƴanci");
  });
  it("does not corrupt ya'ya or quoted words", () => {
    expect(normalize("ya'ya")).toBe("ya'ya");
    expect(normalize("ya ce 'yaro'")).toBe("ya ce 'yaro'");
    expect(normalize("he said 'k'")).toBe("he said 'k'");
  });
  it("folds Arabic-Indic digits", () => {
    expect(normalize("Zaben ٢٠١٥")).toBe("Zaben 2015");
    expect(normalize("٢", { digits: false })).toBe("٢");
  });
});

describe("numbers", () => {
  it("spells cardinals", () => {
    expect(hausaCardinal(2026)).toBe("dubu biyu da ashirin da shida");
    expect(() => hausaCardinal(-1)).toThrow(RangeError);
  });
  it("reads codes digit by digit and handles huge numbers without precision loss", () => {
    expect(spellOutNumbers("0500")).toBe("sifili biyar sifili sifili");
    expect(spellOutNumbers("12345678901234567890").startsWith("ɗaya biyu uku")).toBe(true);
  });
});

describe("speech", () => {
  it("prepares text for a TTS voice", () => {
    expect(prepareForSpeech("Farashin ₦500 ne, kashi 50%")).toBe(
      "Farashin naira ɗari biyar ne, kashi hamsin cikin ɗari",
    );
    expect(prepareForSpeech("maka/miki")).toBe("maka ko miki");
  });
});

describe("syllables and tone", () => {
  it("syllabifies with weight", () => {
    expect(syllabify("kwanduna")).toEqual([
      { text: "kwan", weight: "heavy" },
      { text: "du", weight: "light" },
      { text: "na", weight: "light" },
    ]);
  });
  it("marks known words and leaves unknown ones alone", () => {
    expect(annotateTones("kwanduna motoci")).toBe("kwándùnà mótòcí");
    expect(annotateTones("zzzxqk")).toBe("zzzxqk");
    expect(annotateTones("constructor")).toBe("constructor"); // no prototype-key surprises
  });
});

describe("audit", () => {
  it("flags problems, warns on quoted lookalikes", () => {
    expect(auditText("k'asa ٢٠١٥").ok).toBe(false);
    const r = auditText("Ya ce 'yaro' ne");
    expect(r.ok).toBe(true);
    expect(r.warnings.length).toBe(1);
  });
});
