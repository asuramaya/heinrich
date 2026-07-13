// capture_observatory.mjs — regenerate the README's Observatory screenshots.
//
// The images in docs/observatory-*.png ARE this script's output. Nothing here draws:
// it boots the real viewer against a real .mri, drives it the way a reader would
// (load a model, search a token, pin it, ask for a prompt's readout), and photographs
// the result. Same contract as paper/figures/capture_horn.mjs — the figure is the
// instrument's, the script only frames and clicks.
//
// Reproduction (three steps, no hidden state):
//
//   1. Companion serving the captures on disk:
//        heinrich companion --mri-root web/.data
//   2. Headless Chrome with a software GL context (swiftshader), CDP on 9222:
//        google-chrome --headless=new --remote-debugging-port=9222 \
//          --use-angle=swiftshader --enable-unsafe-swiftshader \
//          --disable-features=LocalNetworkAccessChecks --no-sandbox \
//          --window-size=1600,1000 --user-data-dir=/tmp/heinrich-shots about:blank
//   3. node docs/capture_observatory.mjs
//
// A fresh --user-data-dir matters: the SPA persists pins/PC-pairs/layer in
// localStorage, and a stale one silently reframes every shot.

const CDP = process.env.CDP_PORT || '9222';
const BASE = process.env.BASE || 'http://127.0.0.1:8377/';
const OUTDIR = new URL('./', import.meta.url).pathname;

const MODEL = process.env.MODEL || 'smollm2-135m';
const MODE = process.env.MODE || 'raw';
const PIN_A = process.env.PIN_A || 'Paris';
const PIN_B = process.env.PIN_B || 'London';
const PROMPT = process.env.PROMPT || 'The capital of France is';
const LAYER = parseInt(process.env.LAYER || '18', 10);

// ---------------------------------------------------------------- CDP plumbing
const list = await (await fetch(`http://127.0.0.1:${CDP}/json/list`)).json();
const page = list.find(t => t.type === 'page');
if (!page) { console.error('no CDP page target — is Chrome up with --remote-debugging-port?'); process.exit(2); }
const ws = new WebSocket(page.webSocketDebuggerUrl);
let _id = 0; const _pending = new Map();
ws.addEventListener('message', ev => { const m = JSON.parse(ev.data); if (m.id && _pending.has(m.id)) { _pending.get(m.id)(m); _pending.delete(m.id); } });
await new Promise(r => ws.addEventListener('open', r));
const cmd = (method, params = {}) => new Promise((res, rej) => {
  const id = ++_id; _pending.set(id, m => m.error ? rej(new Error(method + ' ' + JSON.stringify(m.error))) : res(m.result));
  ws.send(JSON.stringify({ id, method, params }));
});
const evl = async (e, awaitPromise = true) => {
  const r = await cmd('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result?.value;
};
const sleep = ms => new Promise(r => setTimeout(r, ms));
const poll = async (expr, ms, every = 500) => {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { try { if (await evl(expr)) return true; } catch (_) {} await sleep(every); }
  return false;
};
const fs = await import('fs');
const shoot = async name => {
  const shot = await cmd('Page.captureScreenshot', { format: 'png' });
  const out = `${OUTDIR}observatory-${name}.png`;
  fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
  console.log('  wrote', out.replace(/.*\/docs\//, 'docs/'));
};

const loadModel = async () => {
  await evl(`(()=>{const s=document.getElementById('sm');s.value=${JSON.stringify(MODEL)};
    const d=document.getElementById('smd');if(d)d.value=${JSON.stringify(MODE)};
    document.getElementById('bl').click();})()`);
  if (!await poll(`(__heinrich.state && __heinrich.state.model===${JSON.stringify(MODEL)} && __heinrich.state.nL>0)`, 60000)) {
    console.error('model did not load'); process.exit(4);
  }
};

// ------------------------------------------------------------------ the session
await cmd('Page.enable'); await cmd('Runtime.enable');
await cmd('Page.navigate', { url: BASE });
await sleep(1500);
if (!await poll("(typeof __heinrich!=='undefined' && document.getElementById('sm') && document.getElementById('sm').options.length>0)", 20000)) {
  console.error('viewer never came up — is the companion serving at ' + BASE + '?'); process.exit(3);
}

console.log(`loading ${MODEL} / ${MODE}`);
await loadModel();
const st = await evl('__heinrich.state');
console.log(`  ${st.model}/${st.mode} — ${st.nL} layers, ${st.nN} tokens, ${st.nK} PCs`);

// WHICH TWO TOKENS? Not my taste — the instrument's own answer. dirDiscover ranks
// this layer's PCs by bimodality against a random baseline (from the producer's
// falsification.json) and hands back the extremes of the winner. We pin those, and
// we frame the clouds on the PCs that actually separate them, instead of the random
// pair a fresh load picks. A pretty screenshot of an arbitrary axis is a lie about
// what the tool is for.
await evl(`__heinrich.setLayer(${LAYER})`);
await sleep(1500);
const disc = await evl(`__heinrich.dirDiscover(${LAYER}, 40)`);
if (!disc || !disc.discovered || !disc.discovered.length) { console.error('dirDiscover found nothing'); process.exit(5); }

// CONSTRAINT, not a preference: the full-vocab reveal reads a producer-side precompute
// that only carries the leading PCs (vocab_pc16.bin). A PC outside it renders the 2,000
// sample and nothing more. So we frame on the most bimodal direction the vocabulary can
// ACTUALLY be drawn in — probed against the live endpoint, not against a hardcoded 16,
// so this keeps working if the producer ever widens the precompute.
const hasVocabColumn = pc => evl(`(async()=>{
  const s=__heinrich.state;
  const url='/api/vocab-pc-column/'+encodeURIComponent(s.model)+'/'+encodeURIComponent(s.mode)
    +'?layer=${LAYER}&pc=${pc}';
  const r=await fetch(url);
  if(!r.ok) return false;
  return !(r.headers.get('content-type')||'').includes('json');   // out-of-range answers 200 + JSON
})()`);

// Four distinct PCs, not three: the Δ lanes derive their axes from the A and B pairs,
// and a repeated PC collapses a whole viewport to the identity diagonal — a lane that
// looks like a rendering bug and says nothing.
const renderable = [];
for (const d of disc.discovered) {
  if (renderable.length >= 4) break;
  if (await hasVocabColumn(d.pc)) renderable.push(d);
}
if (!renderable.length) { console.error('no discovered PC is inside the vocab precompute'); process.exit(6); }
const top = renderable[0];
const A = top.top_pos[0], B = top.top_neg[0];
const best = disc.discovered[0];
if (best.pc !== top.pc) console.log(`  (PC${best.pc} scored higher but is outside the vocab precompute — not framing on what we cannot draw)`);

// The pins and the primary axis come from discovery. The remaining axes may not: only a
// few of the top-ranked PCs fall inside the precompute, so top up from the LEADING PCs,
// which always render. Say so rather than implying every axis was discovered.
const axes = renderable.map(d => d.pc);
const discoveredAxes = axes.length;
for (let pc = 0; axes.length < 4; pc++) if (!axes.includes(pc)) axes.push(pc);
const [pcX, pcY, pcZ, pcW] = axes;
if (discoveredAxes < 4) console.log(`  axes: ${discoveredAxes} discovered (PC${axes.slice(0, discoveredAxes).join(', PC')})`
  + `, topped up from the leading PCs (PC${axes.slice(discoveredAxes).join(', PC')}) — only these render the vocabulary`);
console.log(`  discover L${LAYER}: PC${top.pc} bimodality=${top.bimodality.toFixed(3)} `
  + `(${top.random_baseline_percentile?.toFixed(0) ?? '?'}th pctile of random) var=${(top.variance_share * 100).toFixed(1)}%`);
console.log(`  A = #${A.idx} ${JSON.stringify(A.text)} (proj ${A.proj.toFixed(1)})`);
console.log(`  B = #${B.idx} ${JSON.stringify(B.text)} (proj ${B.proj.toFixed(1)})`);

// The viewer deep-links its entire frame through the URL hash (_loadState), so the
// shot is reproducible by pasting the URL — not by replaying my clicks.
const hash = `m=${encodeURIComponent(MODEL)}&d=${encodeURIComponent(MODE)}&l=${LAYER}`
  + `&a=${A.idx}&b=${B.idx}&p=${pcX}-${pcY},${pcZ}-${pcW}`;
console.log(`  frame: ${BASE}#${hash}`);
await evl(`(()=>{location.hash=${JSON.stringify(hash)};})()`);
await loadModel();                       // reload picks the pins/PCs up out of the hash
await sleep(5000);                       // clouds, trajectories, flowers rebuild
const pins = await evl('__heinrich.pins');
console.log(`  pinned A=#${pins.a} B=#${pins.b}, layer ${LAYER}`);
if (pins.a !== A.idx || pins.b !== B.idx) console.error('  ! pins did not take — frame is not the one reported');

// Run the prompt through the LIVE model FIRST. The River and the live-landing lanes are
// empty until something is asked of the model, and a cockpit photographed before the
// question is a cockpit with dead instruments in it.
console.log(`live forward + river: ${JSON.stringify(PROMPT)}`);
await evl(`__heinrich.gotoTab('live')`).catch(() => {});
await evl(`__heinrich.liveForward(${JSON.stringify(PROMPT)})`).catch(e => console.error('  liveForward:', e.message));
await evl(`__heinrich.riverFetch(${JSON.stringify(PROMPT)})`);
const riverBuilt = await poll("(__heinrich.river && __heinrich.river.strands && __heinrich.river.strands.length>0)", 90000);
if (riverBuilt) {
  const river = await evl('__heinrich.river');
  console.log(`  commit: L${river.commit?.l} -> ${JSON.stringify(river.commit?.text)} | ${river.strands.length} strands`);
} else {
  console.error('  river did not build');
}
await sleep(1500);

// 1. THE COCKPIT — every lane at once, which is the thing the screenshot has to sell.
console.log('shot: cockpit');
await shoot('cockpit');

// 2. THE READOUT RIVER — the live lane (needs weights: local companion only).
if (riverBuilt) {
  console.log('shot: river');
  await evl(`__heinrich.maximize('r0')`);   // maximizing auto-hides the floating panel
  await sleep(1400);
  await shoot('river');
  await evl('__heinrich.maximize(null)'); await sleep(800);
} else {
  console.error('  river SKIPPED (omit, never fake)');
}

// 3. THE VOCABULARY CLOUD — the zoom-progressive full-vocab reveal. The cloud holds a
// 2,000-token sample until you zoom past REVEAL_ZOOM_ON (1.5), at which point the LOD
// pulls the REAL 48,660-row vocabulary in at its true frozen coordinates. Order matters:
// maximize, then zoom (that arms the reveal), then build the grid — arming it after the
// build leaves the lane showing the sample and calling it the vocabulary.
console.log('shot: cloud');
await evl(`__heinrich.maximize('0')`);
await sleep(1200);
await evl('__heinrich.setVocabReveal(true)');
await evl('__heinrich.zoomCloud(0,1.7)');   // just past REVEAL_ZOOM_ON (1.5) — push further
await sleep(800);                            // and the camera ends up INSIDE the cloud
await evl('__heinrich.revBuild(0)');
const revealed = await poll('(__heinrich.vocabReveal.shown>0)', 20000, 700);
const rev = await evl('__heinrich.vocabReveal');
console.log(`  reveal: ${rev.shown}/${rev.total} tokens at zoom ${rev.zoom.toFixed(2)}${revealed ? '' : '  ! never fired'}`);
await sleep(1200);
if (revealed) await shoot('cloud');
else {
  // A skipped shot must say WHY, or the next person re-runs this blind. The grid
  // bails silently if the view's per-layer sample normalization isn't built yet,
  // or if a vocab PC column doesn't come back.
  const why = await evl(`(()=>{const v=__heinrich.views[0];const L=__heinrich.state.cL;
    return {layer:L, pcX:v.pcX, pcY:v.pcY, pcZ:v.pcZ,
            layerMx:!!(v._layerMx&&v._layerMx[L]), clampBounds:!!(v._clampBounds&&v._clampBounds[L]),
            rev:v._rev?{active:v._rev.active,shown:v._rev.shown,grid:!!v._rev.grid}:null,
            enabled:__heinrich.vocabReveal.enabled, zoom:__heinrich.vocabReveal.zoom};})()`);
  console.error('  vocab reveal never fired — SKIPPED (omit, never fake). why:', JSON.stringify(why));
}
await evl('__heinrich.maximize(null)'); await sleep(800);

// 4. THE WEIGHT FLOWER — radial alignment of the readout against the pinned pair.
console.log('shot: flower');
await evl(`__heinrich.maximize('l0')`);
await sleep(1600);
await shoot('flower');
await evl('__heinrich.maximize(null)'); await sleep(800);

// 5. THE PC SPECTRUM — the decomposition itself, layer by layer.
console.log('shot: spectrum');
await evl(`__heinrich.maximize('pcs')`);
await sleep(1600);
await shoot('spectrum');
await evl('__heinrich.maximize(null)'); await sleep(600);

console.log('done');
ws.close(); process.exit(0);
