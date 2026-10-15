/* Tone voice: impose a Hausa tone melody (H / L / F per syllable) on a flat base recording with TD-PSOLA.
 * Line-for-line port of tonevoice.py (the reference implementation).  No dependencies. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ToneVoice = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var RANGE_ST = 6.5;     // distance between a high and a low tone, in semitones
  var EASE_S = 0.014;     // how far inside a syllable's span the target is already reached
  var UNV_HOP_S = 0.005;  // hop for unvoiced stretches (copied unchanged)
  var FILL = 0.02;        // weight of the untouched signal in the final normalisation

  /* A base recording in the page's compact form:
   * { sr, n, f0, runGap, pcm: base64 of little-endian int16, marks: [first, delta, delta...], nuclei: [[s,e]...] } */
  function decode(e) {
    var bin = atob(e.pcm);
    var n = bin.length >> 1;
    var x = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var lo = bin.charCodeAt(2 * i), hi = bin.charCodeAt(2 * i + 1);
      var v = (hi << 8) | lo;
      if (v >= 32768) v -= 65536;
      x[i] = v / 32768;
    }
    var marks = new Float64Array(e.marks.length), acc = 0;
    for (var k = 0; k < e.marks.length; k++) { acc += e.marks[k]; marks[k] = acc; }
    return { sr: e.sr, n: n, x: x, marks: marks, nuclei: e.nuclei, f0: e.f0, runGap: e.runGap };
  }

  function contourPoints(nuclei, tones, rangeSt) {
    var hi = rangeSt / 2, lo = -rangeSt / 2, pts = [];
    for (var k = 0; k < nuclei.length; k++) {
      var s = nuclei[k][0], e = nuclei[k][1];
      var a = Math.min(EASE_S, 0.15 * (e - s));
      var t0 = s + a, t1 = Math.max(s + a + 1e-3, e - a);
      var tone = tones[k], v0, v1;
      if (tone === 'H') { v0 = hi; v1 = hi; } else if (tone === 'L') { v0 = lo; v1 = lo; } else { v0 = hi; v1 = lo; }
      pts.push([t0, v0]); pts.push([t1, v1]);
    }
    return pts;
  }

  function contourSt(pts, t) {
    if (t <= pts[0][0]) return pts[0][1];
    var last = pts.length - 1;
    if (t >= pts[last][0]) return pts[last][1];
    var lo = 0, hi = last;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (pts[mid][0] <= t) lo = mid; else hi = mid; }
    var ta = pts[lo][0], va = pts[lo][1], tb = pts[hi][0], vb = pts[hi][1];
    if (tb - ta <= 1e-9) return vb;
    var u = (t - ta) / (tb - ta);
    return va + (vb - va) * 0.5 * (1 - Math.cos(Math.PI * u));
  }

  function voicedRuns(m, runGap) {
    var runs = [], start = 0;
    if (!m.length) return runs;
    for (var i = 1; i < m.length; i++) if (m[i] - m[i - 1] > runGap) { runs.push([start, i - 1]); start = i; }
    runs.push([start, m.length - 1]);
    return runs;
  }

  function psola(v, f0OfT) {
    var x = v.x, n = v.n, sr = v.sr, m = v.marks, runGap = v.runGap;
    var runs = voicedRuns(m, runGap);
    var Pa = new Float64Array(m.length), i;
    var diffs = [];
    for (i = 1; i < m.length; i++) { var d0 = m[i] - m[i - 1]; if (d0 < 0.013 * sr) diffs.push(d0); }
    diffs.sort(function (a, b) { return a - b; });
    var Pmed = diffs.length ? diffs[diffs.length >> 1] : 0.009 * sr;
    for (i = 0; i < m.length; i++) {
      var s = 0, c = 0;
      if (i > 0 && m[i] - m[i - 1] <= runGap) { s += m[i] - m[i - 1]; c++; }
      if (i + 1 < m.length && m[i + 1] - m[i] <= runGap) { s += m[i + 1] - m[i]; c++; }
      Pa[i] = c ? s / c : Pmed;
    }
    var runLo = [], runHi = [];
    for (var r = 0; r < runs.length; r++) {
      runLo.push(m[runs[r][0]] - Pa[runs[r][0]] / 2);
      runHi.push(m[runs[r][1]] + Pa[runs[r][1]] / 2);
    }
    var y = new Float64Array(n + 4096), ws = new Float64Array(n + 4096);
    var H = Math.round(UNV_HOP_S * sr);
    var t = 0, guard = 0, k, idx;
    while (t < n && guard < 200000) {
      guard++;
      k = -1;
      for (var q = 0; q < runs.length; q++) if (runLo[q] <= t && t <= runHi[q]) { k = q; break; }
      if (k >= 0) {
        var a = runs[k][0], b = runs[k][1];
        var j = a, best = Math.abs(m[a] - t);
        for (idx = a + 1; idx <= b; idx++) { var dd = Math.abs(m[idx] - t); if (dd < best) { best = dd; j = idx; } else if (m[idx] > t) break; }
        var P = Pa[j], cc = Math.round(m[j]), L = Math.round(P);
        var loI = cc - L, hiI = cc + L + 1;
        var segLo = Math.max(loI, 0), segHi = Math.min(hiI, n);
        var dst = Math.round(t) - L + (segLo - loI);
        if (dst >= 0) {
          for (idx = segLo; idx < segHi; idx++) {
            var w = 0.5 * (1 + Math.cos(Math.PI * (idx - cc) / (L + 1)));
            y[dst + idx - segLo] += x[idx] * w;
            ws[dst + idx - segLo] += w;
          }
        }
        t += sr / f0OfT(t / sr);
      } else {
        var c2 = Math.round(t);
        var lo2 = Math.max(c2 - H, 0), hi2 = Math.min(c2 + H + 1, n);
        for (idx = lo2; idx < hi2; idx++) {
          var w2 = 0.5 * (1 + Math.cos(Math.PI * (idx - c2) / (H + 1)));
          y[idx] += x[idx] * w2;
          ws[idx] += w2;
        }
        t += H;
      }
    }
    var out = new Float32Array(n);
    for (i = 0; i < n; i++) out[i] = (y[i] + FILL * x[i]) / (ws[i] + FILL);   // where no window reaches (the very edges) the original shows through
    return out;
  }

  /* tones: array like ['L','L','H'], one per syllable. Returns Float32Array at v.sr, peak-normalised. */
  function render(v, tones, opts) {
    opts = opts || {};
    var rangeSt = opts.rangeSt || RANGE_ST;
    var center = opts.centerHz || v.f0;
    var pts = contourPoints(v.nuclei, tones, rangeSt);
    var f0 = function (t) { return center * Math.pow(2, contourSt(pts, t) / 12); };
    var y = psola(v, f0);
    var peak = 0, i;
    for (i = 0; i < y.length; i++) { var a = Math.abs(y[i]); if (a > peak) peak = a; }
    var g = peak > 0 ? 0.7 / peak : 1;
    for (i = 0; i < y.length; i++) y[i] *= g;
    return y;
  }

  function toWav(samples, sr) {
    var n = samples.length, buf = new ArrayBuffer(44 + n * 2), dv = new DataView(buf);
    function str(o, s) { for (var i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); }
    str(0, 'RIFF'); dv.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    str(36, 'data'); dv.setUint32(40, n * 2, true);
    for (var i = 0; i < n; i++) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      dv.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
    }
    return new Uint8Array(buf);
  }

  return { RANGE_ST: RANGE_ST, decode: decode, contourPoints: contourPoints, contourSt: contourSt, psola: psola, render: render, toWav: toWav };
});
