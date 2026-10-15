const fs = require('fs'), path = require('path');
const TV = require('./tonevoice.js');
const [jobsPath, voicePath, outDir] = process.argv.slice(2);
const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8'));
const voice = JSON.parse(fs.readFileSync(voicePath, 'utf8'));
fs.mkdirSync(outDir, { recursive: true });
const cache = {};
let t0 = Date.now(), n = 0;
for (const j of jobs) {
  const v = cache[j.key] || (cache[j.key] = TV.decode(voice[j.key]));
  const y = TV.render(v, j.tones);
  fs.writeFileSync(path.join(outDir, `${j.key}_${j.tones.join('')}.wav`), TV.toWav(y, v.sr));
  n++;
}
console.log(`rendered ${n} in ${Date.now() - t0} ms`);
