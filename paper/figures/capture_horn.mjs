// capture_horn.mjs — regenerate figures/horn_smollm2-135m.png (Figure 1).
//
// This is the provenance script named in the paper: the figure IS its output.
// Reproduction recipe (three steps, no hidden state):
//
//   1. Companion running with the model's .mri on disk:
//        heinrich companion --mri-root web/.data --pair <code>
//   2. Headless Chrome with a software GL context (swiftshader), CDP on 9222:
//        google-chrome --headless=new --remote-debugging-port=9222 \
//          --use-angle=swiftshader --enable-unsafe-swiftshader \
//          --disable-features=LocalNetworkAccessChecks --no-sandbox \
//          --window-size=1600,1000 --user-data-dir=/tmp/heinrich-cap about:blank
//   3. node paper/figures/capture_horn.mjs
//
// It loads smollm2-135m, fetches the Readout River for the paper's prompt,
// maximizes the River lane (which hides the floating panel), and photographs it.
// The frame, the readout formula, and the honest strand/label rules are the
// live instrument's — nothing here draws; it only frames and captures.
const CDP = process.env.CDP_PORT || '9222';
const OUT = process.env.OUT || new URL('./horn_smollm2-135m.png', import.meta.url).pathname;
const MODEL = 'smollm2-135m', MODE = 'raw';
const PROMPT = process.env.PROMPT || 'The capital of France is';

const list = await (await fetch(`http://127.0.0.1:${CDP}/json/list`)).json();
const page = list.find(t => t.type === 'page');
if (!page) { console.error('no CDP page target — is Chrome running with --remote-debugging-port?'); process.exit(2); }
const ws = new WebSocket(page.webSocketDebuggerUrl);
let _id = 0; const _p = new Map();
ws.addEventListener('message', ev => { const m = JSON.parse(ev.data); if (m.id && _p.has(m.id)) { _p.get(m.id)(m); _p.delete(m.id); } });
await new Promise(r => ws.addEventListener('open', r));
const cmd = (method, params = {}) => new Promise((res, rej) => { const id = ++_id; _p.set(id, m => m.error ? rej(new Error(method + ' ' + JSON.stringify(m.error))) : res(m.result)); ws.send(JSON.stringify({ id, method, params })); });
const evl = async (e, a = true) => { const r = await cmd('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: a }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result?.value; };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const poll = async (e, ms, every = 500) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { try { if (await evl(e)) return true; } catch (_) {} await sleep(every); } return false; };

await cmd('Page.enable'); await cmd('Runtime.enable');
await cmd('Page.navigate', { url: 'http://127.0.0.1:8377/' });
await sleep(1500);
await poll("(typeof __heinrich!=='undefined' && document.getElementById('sm') && document.getElementById('sm').options.length>0)", 15000);
await evl(`(()=>{const s=document.getElementById('sm');s.value=${JSON.stringify(MODEL)};const d=document.getElementById('smd');if(d)d.value=${JSON.stringify(MODE)};document.getElementById('bl').click();return s.value;})()`);
if (!await poll(`(__heinrich.state && __heinrich.state.model===${JSON.stringify(MODEL)} && __heinrich.state.nL>0)`, 40000)) { console.error('model did not load'); process.exit(3); }
try { await evl(`__heinrich.gotoTab && __heinrich.gotoTab('Live')`); } catch (_) {}
await evl(`__heinrich.riverFetch(${JSON.stringify(PROMPT)})`);
if (!await poll("(__heinrich.river && __heinrich.river.strands && __heinrich.river.strands.length>0)", 60000)) { console.error('river did not build'); process.exit(4); }
const river = await evl('__heinrich.river');
console.log('commit:', JSON.stringify(river.commit), '| strands:', river.strands.length, '| scale:', river.scale);
await evl(`__heinrich.maximize('r0')`);   // maximizing the lane auto-hides the Live panel
await sleep(1100);
const shot = await cmd('Page.captureScreenshot', { format: 'png' });
const fs = await import('fs');
fs.writeFileSync(OUT, Buffer.from(shot.data, 'base64'));
console.log('wrote', OUT);
ws.close(); process.exit(0);
