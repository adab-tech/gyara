export { auditText, datasetStats, words } from "./audit.js";
export type { AuditReport, DatasetStats } from "./audit.js";
export { hausaCardinal, spellOutNumbers } from "./numbers.js";
export {
  countAsciiHookCandidates,
  countHooked,
  countQuotedLookalikes,
  normalize,
  normalizeDigits,
} from "./orthography.js";
export { prepareForSpeech, stripEnglishGlosses } from "./speech.js";
export { syllabify } from "./syllables.js";
export type { Syllable } from "./syllables.js";
export { annotateTones, melodyFor } from "./tone.js";
export type { Tone, ToneLexicon, ToneOptions } from "./tone.js";
export const version = "0.1.0";
