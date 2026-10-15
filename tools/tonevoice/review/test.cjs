// Data-specific smoke test for the 50-word review page (it expects those words at those tile positions).
// Needs playwright-core (PLAYWRIGHT_CORE) and a Chromium (CHROMIUM). In-memory stand-in for claude.use("db") and a fake AudioContext
// that records every buffer the page plays (so the in-page audio pipeline can be analysed afterwards).
const { chromium } = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const fs = require('fs');
const DIR = process.env.REVIEW_DIR || __dirname + '/out';   // folder holding index.test.html (see build.py)

const fakeDbInit = () => {
  const store = (window.__store = window.__store || {});
  const listeners = [];
  const snapFor = () => {
    const docs = Object.keys(store).map((id) => ({ id, exists: true, data: () => JSON.parse(JSON.stringify(store[id])), metadata: {} }));
    return { docs, size: docs.length, empty: !docs.length, docChanges: () => [], metadata: {} };
  };
  const fire = () => listeners.forEach((fn) => fn(snapFor()));
  window.__writes = [];
  window.claude = {
    use: async (name) => {
      if (name !== 'db') return null;
      return Object.freeze({
        doc: (p) => ({ set: async (data) => { await new Promise((r) => setTimeout(r, 15)); window.__writes.push(p); store[p.split('/')[1]] = JSON.parse(JSON.stringify(data)); fire(); } }),
        collection: () => ({ onSnapshot: (next) => { listeners.push(next); setTimeout(() => next(snapFor()), 10); return () => {}; } }),
      });
    },
  };
};
const fakeAudioInit = () => {
  window.__played = [];
  window.__sourcesStopped = 0;
  class FakeAC {
    constructor() { this.state = 'running'; this.destination = {}; window.__acCount = (window.__acCount || 0) + 1; }
    resume() { return Promise.resolve(); }
    createBuffer(ch, len, sr) { const d = new Float32Array(len); return { numberOfChannels: ch, length: len, sampleRate: sr, getChannelData: () => d }; }
    createBufferSource() {
      const s = { buffer: null, onended: null, connect() {}, stop() { window.__sourcesStopped++; if (s._t) clearTimeout(s._t); },
        start() { window.__played.push({ sr: s.buffer.sampleRate, n: s.buffer.length, data: Array.from(s.buffer.getChannelData(0)) });
                  s._t = setTimeout(() => { if (s.onended) s.onended(); }, 60); } };
      return s;
    }
  }
  window.AudioContext = FakeAC; window.webkitAudioContext = FakeAC;
};

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, args: ['--no-sandbox'] });
  const errors = [];
  const results = [];
  const check = (name, cond, extra) => { results.push([cond ? 'PASS' : 'FAIL', name, cond ? '' : (extra || '')]); };
  const wire = (p, tag) => {
    p.on('pageerror', (e) => errors.push(tag + ' pageerror: ' + e.message));
    p.on('console', (m) => { if (m.type() === 'error' && !/fonts\.(googleapis|gstatic)|ERR_|Failed to load resource/.test(m.text())) errors.push(tag + ' console: ' + m.text()); });
  };
  const settle = async (pg) => { await pg.waitForTimeout(40); await pg.waitForFunction(() => document.querySelector('#save') && document.querySelector('#save').textContent === 'Saved', null, { timeout: 5000 }); };
  const newPage = async (opts = {}) => {
    const ctx = await browser.newContext({ viewport: opts.viewport || { width: 1100, height: 900 }, colorScheme: opts.scheme || 'light' });
    if (opts.store) await ctx.addInitScript((s) => { window.__store = s; }, opts.store);
    if (opts.fakeDb !== false) await ctx.addInitScript(fakeDbInit);
    if (opts.fakeAudio !== false) await ctx.addInitScript(fakeAudioInit);
    const p = await ctx.newPage();
    wire(p, opts.tag || 'p');
    await p.goto('file://' + DIR + '/index.test.html');
    await p.waitForSelector('.word');
    return p;
  };
  const played = (p) => p.evaluate(() => window.__played.length);
  const lastPlayed = (p) => p.evaluate(() => { const x = window.__played[window.__played.length - 1]; return x ? { sr: x.sr, n: x.n } : null; });
  const captures = [];
  const grab = async (p, label) => { const c = await p.evaluate(() => window.__played[window.__played.length - 1]); captures.push({ label, sr: c.sr, data: c.data }); };

  /* ---- 1. basic page + audio controls ---- */
  const page = await newPage();
  check('title', (await page.title()) === 'Hausa Tone Review');
  check('50 map tiles', (await page.locator('.tile').count()) === 50);
  check('opens on word 1', (await page.textContent('.count')) === '1 of 50' && (await page.textContent('.word')) === 'ajami');
  check('hear button present and enabled', (await page.locator('.hear').count()) === 1 && !(await page.locator('.hear').isDisabled()));
  check('hear button names the proposal', /Hear the proposal/.test(await page.textContent('.hear')));
  check('three play buttons (Wiktionary, Newman) for ajami', (await page.locator('#card .play').count()) === 2, String(await page.locator('#card .play').count()));
  check('Newman differs: adopt button shown', (await page.locator('.adopt').count()) === 1);
  check('Newman row shows his marked spelling and meaning', /ajàmì/.test(await page.textContent('.sources')) && /arabic script/.test(await page.textContent('.sources')));

  await page.click('.hear');
  check('click plays one buffer', (await played(page)) === 1);
  const lp = await lastPlayed(page);
  check('buffer is 16 kHz and 0.5 to 2 s long', lp && lp.sr === 16000 && lp.n > 8000 && lp.n < 32000, JSON.stringify(lp));
  check('button shows playing', (await page.getAttribute('.hear', 'data-playing')) === 'true');
  await page.waitForTimeout(120);
  check('button returns to idle when the sound ends', (await page.getAttribute('.hear', 'data-playing')) === 'false');
  await grab(page, 'b1-01 proposed');

  // second click while playing stops it
  await page.click('.hear');
  await page.click('.hear');
  check('pressing again while playing stops it', (await page.evaluate(() => window.__sourcesStopped)) >= 1);

  // Newman play button plays a different pattern
  const before = await played(page);
  await page.click('#card .play[data-pid="new0"]');
  check('Newman play button plays', (await played(page)) === before + 1);
  await grab(page, 'b1-01 newman');
  const a = captures.find((c) => c.label === 'b1-01 proposed').data, b = captures.find((c) => c.label === 'b1-01 newman').data;
  let diff = 0; for (let i = 0; i < Math.min(a.length, b.length); i++) diff += Math.abs(a[i] - b[i]);
  check('proposal and Newman audio differ', diff > 50, String(diff));
  await page.waitForTimeout(100);

  // keyboard
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  const k0 = await played(page);
  await page.keyboard.press('p');
  check('P plays the shown pattern', (await played(page)) === k0 + 1);
  await page.waitForTimeout(100);
  await page.keyboard.press('n');
  check('N plays the first Newman pattern', (await played(page)) === k0 + 2);
  await page.waitForTimeout(100);

  /* ---- 2. adopt Newman ---- */
  await page.click('.adopt');
  await page.waitForFunction(() => document.querySelector('.count').textContent === '2 of 50');
  await settle(page);
  let d = await page.evaluate(() => JSON.parse(JSON.stringify(window.__store)));
  check('adopting Newman saves Wrong with his pattern and moves on', d['b1-01'] && d['b1-01'].verdict === 'wrong' && d['b1-01'].tones.join('') === 'HLL' && d['b1-01'].proposed.join('') === 'LLH', JSON.stringify(d['b1-01']));

  /* ---- 3. right / wrong / not sure still work ---- */
  await page.click('.vbtn[data-v="ok"]');
  await page.waitForFunction(() => document.querySelector('.count').textContent === '3 of 50');
  d = await page.evaluate(() => JSON.parse(JSON.stringify(window.__store)));
  check('Right saves ok with the proposal', d['b1-02'] && d['b1-02'].verdict === 'ok' && d['b1-02'].tones.join('') === 'LLH', JSON.stringify(d['b1-02']));
  await page.click('.vbtn[data-v="wrong"]');
  check('Wrong shows editable chips and the listen button says my pattern', (await page.locator('button.chip').count()) === 3 && /Hear my pattern/.test(await page.textContent('.hear')) === false, await page.textContent('.hear'));
  await page.click('button.chip[data-chip="0"]');   // L -> F
  await settle(page);
  check('after editing, listen button says my pattern', /Hear my pattern/.test(await page.textContent('.hear')), await page.textContent('.hear'));
  const c0 = await played(page);
  await page.click('.hear');
  check('listen plays my edited pattern', (await played(page)) === c0 + 1);
  await grab(page, 'b1-03 mine FLL?');
  await page.waitForTimeout(100);

  /* ---- 4. word with two Newman rows at phone width ---- */
  const phone = await newPage({ viewport: { width: 400, height: 900 }, tag: 'phone' });
  await phone.click('.tile[data-go="12"]');
  check('faɗa shows two Newman rows', (await phone.locator('.nrow').count()) === 2, String(await phone.locator('.nrow').count()));
  check('no horizontal scroll at 400px on the busiest card', !(await phone.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)));
  await phone.click('.tile[data-go="16"]');   // kare: HL guard / LH dog
  check('kare shows both Newman senses', /dog/.test(await phone.textContent('.sources')) && /guard/.test(await phone.textContent('.sources')));
  await phone.click('#card .play[data-pid="new1"]');
  await grab(phone, 'b1-17 newman[1] LH');
  await phone.click('.tile[data-go="24"]');   // masassaƙi, 4 syllables
  check('4-syllable word fits phone width', !(await phone.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)));
  await phone.click('#card .play[data-pid="wik"]');
  await grab(phone, 'b1-25 wiktionary');
  await phone.click('.tile[data-go="28"]');   // mutum: falling tone
  await phone.click('.hear');
  await grab(phone, 'b1-29 proposed LF');

  /* ---- 5. autoplay ---- */
  const au = await newPage({ tag: 'auto' });
  const n0 = await played(au);
  await au.check('#auto');
  check('switching autoplay on plays the current word', (await played(au)) === n0 + 1);
  await au.waitForTimeout(100);
  await au.click('.vbtn[data-v="ok"]');
  await au.waitForFunction(() => document.querySelector('.count').textContent === '2 of 50');
  await au.waitForTimeout(80);
  check('with autoplay on, the next word plays by itself', (await played(au)) === n0 + 2, String(await played(au) - n0));
  check('autoplay choice is remembered', await au.evaluate(() => window.localStorage.getItem('hausa-tone-review:v1:auto')) === 'on');
  await au.uncheck('#auto');
  await au.waitForTimeout(100);
  const n1 = await played(au);
  await au.keyboard.press('ArrowRight');
  await au.waitForTimeout(80);
  check('autoplay off: moving on is silent', (await played(au)) === n1);

  /* ---- 6. blind mode ---- */
  const bl = await newPage({ tag: 'blind' });
  await bl.click('[data-mode="blind"]');
  check('blind: hear button is disabled until the pattern is complete and says my pattern', (await bl.locator('.hear').isDisabled()) && /Hear my pattern/.test(await bl.textContent('.hear')));
  check('blind: no source play buttons before reveal', (await bl.locator('#card .play').count()) === 0);
  const nb = await bl.locator('button.chip').count();
  for (let i = 0; i < nb; i++) await bl.click(`button.chip[data-chip="${i}"]`);   // all H
  check('blind: hear enabled once every syllable is set', !(await bl.locator('.hear').isDisabled()));
  const b0 = await played(bl);
  await bl.click('.hear');
  check('blind: plays my pattern', (await played(bl)) === b0 + 1);
  await bl.keyboard.press('n');
  check('blind: N does nothing before reveal', (await played(bl)) === b0 + 1);
  await bl.waitForTimeout(100);

  /* ---- 7. reload keeps answers; no storage still works ---- */
  const snapshot = await page.evaluate(() => JSON.parse(JSON.stringify(window.__store)));
  const re = await newPage({ store: snapshot, tag: 'reload' });
  await re.waitForTimeout(300);
  check('reload opens on the first unreviewed word', (await re.textContent('.count')) === '4 of 50', await re.textContent('.count'));
  const nodb = await newPage({ fakeDb: false, tag: 'nodb', viewport: { width: 400, height: 800 } });
  await nodb.waitForTimeout(200);
  check('without storage the page still says so and still plays', /this browser only/.test(await nodb.textContent('#save')));
  await nodb.click('.hear');
  check('plays without storage', (await played(nodb)) === 1);

  /* ---- 8. real AudioContext (no fake): must not throw ---- */
  const real = await newPage({ fakeAudio: false, tag: 'real' });
  await real.click('.hear');
  await real.waitForTimeout(300);
  const msg = await real.textContent('#audiomsg');
  check('real AudioContext: no error message after pressing play', msg === '', msg);

  /* ---- 9. CSV and screenshots ---- */
  await page.evaluate(() => { navigator.clipboard.writeText = async (t) => { window.__copied = t; }; });
  await page.click('#copy');
  await page.waitForTimeout(100);
  const csv = await page.evaluate(() => window.__copied || '');
  check('copy results still builds a CSV with header + 50 rows', csv.split('\n').length === 51 && csv.startsWith('n,word,meaning,proposed,verdict,your_tones,note,mode'), csv.split('\n').length + ' lines');

  const shots = [['desktop-light', 1100, 900, 'light'], ['desktop-dark', 1100, 900, 'dark'], ['phone-light', 400, 900, 'light'], ['phone-dark', 400, 900, 'dark']];
  for (const [name, w, h, scheme] of shots) {
    const p = await newPage({ viewport: { width: w, height: h }, scheme, store: snapshot, tag: 'shot-' + name });
    await p.click('.tile[data-go="12"]');
    await p.waitForTimeout(250);
    await p.screenshot({ path: `${DIR}/shot-${name}.png`, fullPage: true });
  }

  fs.writeFileSync(DIR + '/captures.json', JSON.stringify(captures));
  await browser.close();
  results.forEach((r) => console.log(r.join('  ')));
  console.log('\nconsole/page errors:', errors.length ? errors : 'none');
  console.log(results.filter((r) => r[0] === 'FAIL').length ? 'SOME FAILED' : 'ALL PASSED');
})();
