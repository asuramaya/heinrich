// Heinrich companion Worker — serves the MRI explorer from R2.
//
// Design: the recorded MRI is static. Every endpoint here is either a plain
// R2 GET of a precomputed JSON, or a byte-range read of an immutable .bin
// index (the format heinrich's mri-decompose already writes for O(1) seeks).
// The Worker never runs a model and never parses numpy .npz — token metadata
// is precomputed to tokens.json at capture time.
//
// MVP endpoint coverage (the SPA boot path → first cloud render):
//   GET /api/models
//   GET /api/decomp-meta/<model>/<mode>
//   GET /api/serve-meta/<model>/<mode>
//   GET /api/decomp/<model>/<mode>?layer=N
//   GET /api/cloud-bundle/<model>/<mode>?full=..&medium=..&step=N
// Anything else under /api → 501 (lazy/interactive endpoints, wired next).

const JSON_HEADERS = { "content-type": "application/json" };
const IMMUTABLE = "public, max-age=31536000, immutable";

function mriPrefix(model, mode) {
  return `${model}/${mode}.mri`;
}

function jsonResponse(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: JSON_HEADERS });
}

async function r2json(env, key) {
  const obj = await env.MRI.get(key);
  if (!obj) return null;
  return await obj.json();
}

// Range read → ArrayBuffer (or null if missing).
async function r2range(env, key, offset, length) {
  const obj = await env.MRI.get(key, { range: { offset, length } });
  if (!obj) return null;
  return await obj.arrayBuffer();
}

// Minimal NPY v1.0 reader: returns {dtype, shape, dataOffset, buffer}.
function parseNpy(buf) {
  const u8 = new Uint8Array(buf);
  // magic \x93NUMPY, ver(2), headerLen uint16 LE at offset 8
  const dv = new DataView(buf);
  const headerLen = dv.getUint16(8, true);
  const header = new TextDecoder().decode(u8.subarray(10, 10 + headerLen));
  const dtype = /'descr':\s*'([^']+)'/.exec(header)[1];
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(header)[1];
  const shape = shapeStr.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  return { dtype, shape, dataOffset: 10 + headerLen, buffer: buf };
}

// Read L{NN}_scores.npy and return Float32Array[N*K] + dims.
async function readLayerScores(env, prefix, layer) {
  const key = `${prefix}/decomp/L${String(layer).padStart(2, "0")}_scores.npy`;
  const obj = await env.MRI.get(key);
  if (!obj) return null;
  const buf = await obj.arrayBuffer();
  const { dtype, shape, dataOffset } = parseNpy(buf);
  const [n, k] = shape;
  const dv = new DataView(buf, dataOffset);
  const out = new Float32Array(n * k);
  if (dtype === "<f2") {
    for (let i = 0; i < out.length; i++) out[i] = dv.getFloat16(i * 2, true);
  } else if (dtype === "<f4") {
    for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
  } else {
    throw new Error(`unsupported dtype ${dtype}`);
  }
  return { scores: out, n, k };
}

// /api/decomp/<m>/<mode>?layer=N  → JSON the SPA loadAll() expects.
async function decompLayer(env, prefix, layer) {
  const layerData = await readLayerScores(env, prefix, layer);
  if (!layerData) return jsonResponse({ error: "no layer" }, 404);
  const toks = await r2json(env, `${prefix}/decomp/tokens.json`);
  const { scores, n, k } = layerData;
  // scores as [N][K] nested arrays (matches the Python _load_decomp shape)
  const rows = new Array(n);
  for (let i = 0; i < n; i++) rows[i] = Array.from(scores.subarray(i * k, i * k + k));
  return jsonResponse({
    layer,
    n_tokens: n,
    K: k,
    scores: rows,
    scripts: toks?.scripts ?? [],
    texts: toks?.token_texts ?? [],
    token_ids: toks?.token_ids ?? [],
  });
}

// /api/serve-meta — the SPA reads full_k (PC count) + pc_vars (per-layer
// variance) from here. Synthesize from all_scores.bin's header + variance block
// (mirrors companion_serve._synthesize_serve_meta) so no precompute is needed.
async function serveMeta(env, prefix) {
  const decompMeta = (await r2json(env, `${prefix}/decomp/meta.json`)) ?? {};
  const allKey = `${prefix}/decomp/all_scores.bin`;
  const headBuf = await r2range(env, allKey, 0, 20);
  if (!headBuf) return jsonResponse({ error: "No decomposition. Run mri-decompose." }, 404);
  const hv = new DataView(headBuf);
  const magic = new TextDecoder().decode(new Uint8Array(headBuf, 0, 4));
  if (magic !== "HEI2") return jsonResponse({ error: "bad all_scores magic" }, 500);
  const totalLayers = hv.getUint32(4, true);
  const nTok = hv.getUint32(8, true);
  const varK = hv.getUint32(16, true);
  // variance block: [total_layers x var_k] float32 right after the 20B header
  const varBuf = await r2range(env, allKey, 20, totalLayers * varK * 4);
  const vdv = new DataView(varBuf);
  const pc_vars = [];
  let o = 0;
  for (let l = 0; l < totalLayers; l++) {
    const row = [];
    for (let k = 0; k < varK; k++) { row.push(vdv.getFloat32(o, true)); o += 4; }
    pc_vars.push(row);
  }
  // full_k from the PC index header (authoritative count of stored PCs)
  let full_k = varK;
  const pcHead = await r2range(env, `${prefix}/decomp/pc_scores.bin`, 0, 16);
  if (pcHead) full_k = new DataView(pcHead).getUint32(12, true);
  return jsonResponse({
    version: "worker", source: "decomp",
    n_layers: totalLayers,
    n_real_layers: decompMeta.n_real_layers ?? totalLayers,
    n_tokens: nTok,
    full_k,
    pc_vars,
    steps: {},
  });
}

// /api/cloud-bundle — repack PC slabs from pc_scores.bin into the CLDB format
// the SPA decoder expects (companion.py _cloud_bundle).
async function cloudBundle(env, prefix, fullPcs, medPcs, step) {
  const key = `${prefix}/decomp/pc_scores.bin`;
  const headBuf = await r2range(env, key, 0, 16);
  if (!headBuf) return jsonResponse({ error: "no pc index" }, 404);
  const hdv = new DataView(headBuf);
  const magic = new TextDecoder().decode(new Uint8Array(headBuf, 0, 4));
  if (magic !== "PCSC") return jsonResponse({ error: "bad pc magic" }, 500);
  const nLayers = hdv.getUint32(4, true);
  const nTok = hdv.getUint32(8, true);
  const fullK = hdv.getUint32(12, true);
  const pcStride = nLayers * nTok * 2; // bytes per PC

  // dedupe, drop medium PCs already in full, and DROP out-of-range PCs
  // (the SPA can request PCs beyond this model's count; reading past the file
  // is a 416 "range not satisfiable" from R2 → would 500 the whole cloud).
  const fullIds = [...new Set(fullPcs)].filter((p) => p >= 0 && p < fullK);
  const fullSet = new Set(fullIds);
  const medIds = [...new Set(medPcs)].filter((p) => p >= 0 && p < fullK && !fullSet.has(p));

  const fullSlabs = [];
  for (const pc of fullIds) {
    const slab = await r2range(env, key, 16 + pc * pcStride, pcStride);
    fullSlabs.push(new Uint8Array(slab));
  }

  // medium: subsample tokens by `step` within each layer row
  const nSample = step > 1 ? Math.ceil(nTok / step) : nTok;
  const medSlabs = [];
  for (const pc of medIds) {
    const slab = await r2range(env, key, 16 + pc * pcStride, pcStride);
    const src = new Uint8Array(slab);
    const out = new Uint8Array(nLayers * nSample * 2);
    let o = 0;
    for (let li = 0; li < nLayers; li++) {
      const rowBase = li * nTok * 2;
      for (let t = 0; t < nTok; t += step) {
        const s = rowBase + t * 2;
        out[o++] = src[s];
        out[o++] = src[s + 1];
      }
    }
    medSlabs.push(out);
  }

  // header: <4sIIIIIII  CLDB ver nFull nMed nLayers nTok nSample step
  const head = new ArrayBuffer(4 + 7 * 4);
  const dv = new DataView(head);
  new Uint8Array(head).set([0x43, 0x4c, 0x44, 0x42]); // 'CLDB'
  dv.setUint32(4, 1, true);
  dv.setUint32(8, fullIds.length, true);
  dv.setUint32(12, medIds.length, true);
  dv.setUint32(16, nLayers, true);
  dv.setUint32(20, nTok, true);
  dv.setUint32(24, nSample, true);
  dv.setUint32(28, step, true);

  const idBytes = new ArrayBuffer((fullIds.length + medIds.length) * 4);
  const idv = new DataView(idBytes);
  let off = 0;
  for (const id of fullIds) { idv.setUint32(off, id, true); off += 4; }
  for (const id of medIds) { idv.setUint32(off, id, true); off += 4; }

  const parts = [head, idBytes, ...fullSlabs, ...medSlabs];
  return new Response(new Blob(parts), {
    headers: { "content-type": "application/octet-stream" },
  });
}

// ---- Phase-1 data endpoints: pure R2 reads + precomputed JSON ----

async function serveRaw(env, key) {
  const obj = await env.MRI.get(key);
  if (!obj) return new Response("not found", { status: 404 });
  return new Response(obj.body, {
    headers: { "content-type": "application/octet-stream", "cache-control": IMMUTABLE },
  });
}

function octet(parts) {
  return new Response(new Blob(parts), { headers: { "content-type": "application/octet-stream" } });
}

// pc_scores.bin (PCSC): header <4sIII magic,n_layers,n_tok,full_k ; payload [K x layers x tok] f16
async function pcHeader(env, prefix) {
  const h = await r2range(env, `${prefix}/decomp/pc_scores.bin`, 0, 16);
  if (!h) return null;
  const dv = new DataView(h);
  return { nLayers: dv.getUint32(4, true), nTok: dv.getUint32(8, true), fullK: dv.getUint32(12, true) };
}

// /api/pc-full?pc=N → <III n_layers,n_tok,pc + f16[n_layers*n_tok]
async function pcFull(env, prefix, pc) {
  const hd = await pcHeader(env, prefix);
  if (!hd) return jsonResponse({ error: "no pc index" }, 404);
  if (pc >= hd.fullK) return jsonResponse({ error: `pc ${pc} out of range` }, 400);
  const stride = hd.nLayers * hd.nTok * 2;
  const slab = await r2range(env, `${prefix}/decomp/pc_scores.bin`, 16 + pc * stride, stride);
  const head = new ArrayBuffer(12); const hv = new DataView(head);
  hv.setUint32(0, hd.nLayers, true); hv.setUint32(4, hd.nTok, true); hv.setUint32(8, pc, true);
  return octet([head, slab]);
}

// /api/pc-column?pc=N&layer=L → raw f16[n_tok]
async function pcColumn(env, prefix, pc, layer) {
  const hd = await pcHeader(env, prefix);
  if (!hd) return jsonResponse({ error: "no pc index" }, 404);
  const pcStride = hd.nLayers * hd.nTok * 2;
  const off = 16 + pc * pcStride + layer * hd.nTok * 2;
  const col = await r2range(env, `${prefix}/decomp/pc_scores.bin`, off, hd.nTok * 2);
  return octet([col]);
}

// token_scores.bin (TOKS): header <4sIII magic,n_tok,n_layers,full_k ; [N x layers x K] f16
async function tokHeader(env, prefix) {
  const h = await r2range(env, `${prefix}/decomp/token_scores.bin`, 0, 16);
  if (!h) return null;
  const dv = new DataView(h);
  return { nTok: dv.getUint32(4, true), nLayers: dv.getUint32(8, true), fullK: dv.getUint32(12, true) };
}

// /api/token-pca?token=N → <II n_layers,full_k + f32[n_layers*full_k]
async function tokenPca(env, prefix, token) {
  const hd = await tokHeader(env, prefix);
  if (!hd) return jsonResponse({ error: "no token index" }, 404);
  const stride = hd.nLayers * hd.fullK * 2;
  const slab = await r2range(env, `${prefix}/decomp/token_scores.bin`, 16 + token * stride, stride);
  const sdv = new DataView(slab); const n = hd.nLayers * hd.fullK;
  const out = new ArrayBuffer(8 + n * 4); const odv = new DataView(out);
  odv.setUint32(0, hd.nLayers, true); odv.setUint32(4, hd.fullK, true);
  for (let i = 0; i < n; i++) odv.setFloat32(8 + i * 4, sdv.getFloat16(i * 2, true), true);
  return octet([out]);
}

// /api/token-layer?token=N&layer=L → <II K,layer + f16[K]
async function tokenLayer(env, prefix, token, layer) {
  const hd = await tokHeader(env, prefix);
  if (!hd) return jsonResponse({ error: "no token index" }, 404);
  const stride = hd.nLayers * hd.fullK * 2;
  const off = 16 + token * stride + layer * hd.fullK * 2;
  const row = await r2range(env, `${prefix}/decomp/token_scores.bin`, off, hd.fullK * 2);
  const head = new ArrayBuffer(8); const hv = new DataView(head);
  hv.setUint32(0, hd.fullK, true); hv.setUint32(4, layer, true);
  return octet([head, row]);
}

// vocab_scores.bin (VSCR): header <4sIII magic,n_rows,n_layers,K ; [rows x layers x K] f16
// /api/vocab-token?row=R → <II n_layers,K + f32[n_layers*K] — same wire format as token-pca
async function vocabToken(env, prefix, row) {
  const key = `${prefix}/decomp/vocab_scores.bin`;
  const h = await r2range(env, key, 0, 16);
  if (!h) return jsonResponse({ error: "no full-vocab projection" }, 404);
  const dv = new DataView(h);
  const nRows = dv.getUint32(4, true), nLayers = dv.getUint32(8, true), K = dv.getUint32(12, true);
  if (row < 0 || row >= nRows) return jsonResponse({ error: `vocab row ${row} out of range` }, 400);
  const stride = nLayers * K * 2;
  const slab = await r2range(env, key, 16 + row * stride, stride);
  const sdv = new DataView(slab); const n = nLayers * K;
  const out = new ArrayBuffer(8 + n * 4); const odv = new DataView(out);
  odv.setUint32(0, nLayers, true); odv.setUint32(4, K, true);
  for (let i = 0; i < n; i++) odv.setFloat32(8 + i * 4, sdv.getFloat16(i * 2, true), true);
  return octet([out]);
}

// token_neurons.bin (TOKN): header <4sIII magic,n_tok,n_layers,intermediate ; [N x layers x inter] f16
// /api/neuron-field?token=N → raw f16[n_layers*intermediate]
async function neuronField(env, prefix, token) {
  const key = `${prefix}/decomp/token_neurons.bin`;
  const h = await r2range(env, key, 0, 16);
  if (!h) return jsonResponse({ error: "no neuron index" }, 404);
  const dv = new DataView(h);
  const nLayers = dv.getUint32(8, true), inter = dv.getUint32(12, true);
  const stride = nLayers * inter * 2;
  const slab = await r2range(env, key, 16 + token * stride, stride);
  return octet([slab]);
}

// /api/token-bio?token=N → JSON {token_idx, max_per_layer}
async function tokenBio(env, prefix, token) {
  const obj = await env.MRI.get(`${prefix}/decomp/gate_heatmap.npy`);
  if (!obj) return jsonResponse({ error: "no gate heatmap" }, 404);
  const buf = await obj.arrayBuffer();
  const { shape, dataOffset, dtype } = parseNpy(buf);
  const nLayers = shape[1]; const dv = new DataView(buf, dataOffset);
  const row = [];
  for (let l = 0; l < nLayers; l++) {
    const i = token * nLayers + l;
    const v = dtype === "<f2" ? dv.getFloat16(i * 2, true) : dv.getFloat32(i * 4, true);
    row.push(Math.round(v * 100) / 100);
  }
  return jsonResponse({ token_idx: token, max_per_layer: row });
}

// /api/token-hover?token=N&layer=L → <III K,inter,layer + f16[K] + f16[inter] (gate×up)
async function tokenHover(env, prefix, token, layer) {
  const ls = await readLayerScores(env, prefix, layer);
  if (!ls) return jsonResponse({ error: "no layer scores" }, 404);
  const { scores, k } = ls;
  const pcRow = scores.subarray(token * k, token * k + k);
  const gateObj = await env.MRI.get(`${prefix}/mlp/L${String(layer).padStart(2, "0")}_gate.npy`);
  const upObj = await env.MRI.get(`${prefix}/mlp/L${String(layer).padStart(2, "0")}_up.npy`);
  let inter = 0, neur = new Float32Array(0);
  if (gateObj && upObj) {
    const gb = await gateObj.arrayBuffer(); const ub = await upObj.arrayBuffer();
    const gp = parseNpy(gb), up = parseNpy(ub);
    inter = gp.shape[1];
    const gdv = new DataView(gb, gp.dataOffset), udv = new DataView(ub, up.dataOffset);
    neur = new Float32Array(inter);
    for (let j = 0; j < inter; j++) {
      const idx = token * inter + j;
      neur[j] = gdv.getFloat16(idx * 2, true) * udv.getFloat16(idx * 2, true);
    }
  }
  const head = new ArrayBuffer(12); const hv = new DataView(head);
  hv.setUint32(0, k, true); hv.setUint32(4, inter, true); hv.setUint32(8, layer, true);
  const pcBytes = new ArrayBuffer(k * 2); const pdv = new DataView(pcBytes);
  for (let i = 0; i < k; i++) pdv.setFloat16(i * 2, pcRow[i], true);
  const nBytes = new ArrayBuffer(inter * 2); const ndv = new DataView(nBytes);
  for (let j = 0; j < inter; j++) ndv.setFloat16(j * 2, neur[j], true);
  return octet([head, pcBytes, nBytes]);
}

// /api/token-bundle?full=..&hover=..&layer=N → TKBD bundle (companion._token_bundle)
async function tokenBundle(env, prefix, fullTokens, hoverTokens, layer) {
  const fullIds = [...new Set(fullTokens)], hoverIds = [...new Set(hoverTokens)];
  const tokenIds = [...new Set([...fullIds, ...hoverIds])];
  const fullSet = new Set(fullIds), hoverSet = new Set(hoverIds);
  const mkHead = (n) => { const h = new ArrayBuffer(16); const hv = new DataView(h);
    new Uint8Array(h).set([0x54, 0x4b, 0x42, 0x44]); hv.setUint32(4, 1, true); hv.setUint32(8, layer, true); hv.setUint32(12, n, true); return h; };
  const hd = await tokHeader(env, prefix);
  if (!tokenIds.length || !hd) return octet([mkHead(0)]);
  const { nLayers, fullK } = hd; const stride = nLayers * fullK * 2;
  // gate/up for the hover layer (real layers only)
  let gp = null, up = null, gdv = null, udv = null, inter = 0;
  if (hoverIds.length) {
    const lp = String(layer).padStart(2, "0");
    const g = await env.MRI.get(`${prefix}/mlp/L${lp}_gate.npy`);
    const u = await env.MRI.get(`${prefix}/mlp/L${lp}_up.npy`);
    if (g && u) { const gb = await g.arrayBuffer(), ub = await u.arrayBuffer();
      gp = parseNpy(gb); up = parseNpy(ub); inter = gp.shape[1];
      gdv = new DataView(gb, gp.dataOffset); udv = new DataView(ub, up.dataOffset); }
  }
  const entries = [], payloads = [];
  for (const t of tokenIds) {
    let flags = 0, hoverK = 0, hoverInter = 0;
    const slab = await r2range(env, `${prefix}/decomp/token_scores.bin`, 16 + t * stride, stride);
    const sdv = new DataView(slab);
    if (fullSet.has(t)) { flags |= 1; const n = nLayers * fullK;
      const pb = new ArrayBuffer(n * 4); const pv = new DataView(pb);
      for (let i = 0; i < n; i++) pv.setFloat32(i * 4, sdv.getFloat16(i * 2, true), true);
      payloads.push(pb); }
    if (hoverSet.has(t) && layer >= 0 && layer < nLayers) {
      flags |= 2; hoverK = fullK;
      const hb = new ArrayBuffer(fullK * 2); const hv2 = new DataView(hb);
      for (let i = 0; i < fullK; i++) hv2.setFloat16(i * 2, sdv.getFloat16((layer * fullK + i) * 2, true), true);
      payloads.push(hb);
      if (gdv) { flags |= 4; hoverInter = inter;
        const nb = new ArrayBuffer(inter * 2); const ndv = new DataView(nb);
        for (let j = 0; j < inter; j++) { const idx = t * inter + j; ndv.setFloat16(j * 2, gdv.getFloat16(idx * 2, true) * udv.getFloat16(idx * 2, true), true); }
        payloads.push(nb); }
    }
    const eh = new ArrayBuffer(24); const ev = new DataView(eh);
    ev.setUint32(0, t, true); ev.setUint32(4, flags, true); ev.setUint32(8, nLayers, true);
    ev.setUint32(12, fullK, true); ev.setUint32(16, hoverK, true); ev.setUint32(20, hoverInter, true);
    entries.push(eh);
  }
  return octet([mkHead(tokenIds.length), ...entries, ...payloads]);
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (!path.startsWith("/api/")) {
      // Three surfaces: landing (/), the viz SPA (/observatory), docs (/docs).
      // Serve the real asset if it exists; otherwise fall back to that surface's
      // index (client-side routes / the hash-state viz / VitePress SPA nav).
      const res = await env.ASSETS.fetch(request);
      if (res.status !== 404) return res;
      const idx = path === "/observatory" || path.startsWith("/observatory/")
          ? "/observatory/index.html"
          : path === "/book" || path.startsWith("/book/")
          ? "/book/index.html"
          : path === "/docs" || path.startsWith("/docs/")
          ? "/docs/index.html"
          : "/index.html";
      return env.ASSETS.fetch(new Request(new URL(idx, url), request));
    }

    try {
      if (path === "/api/models") {
        const models = (await r2json(env, "models.json")) ?? [];
        return jsonResponse(models);
      }

      const parts = path.split("/"); // ['', 'api', '<ep>', '<model>', '<mode>']
      const ep = parts[2];
      const model = parts[3];
      const mode = parts[4];
      const prefix = model && mode ? mriPrefix(model, mode) : null;

      // Live/navigation endpoints have no backend in a static deploy. The SPA's
      // long-poll loops expect the local companion to hold the request ~25s; a
      // worker returns instantly, so without a signal they'd busy-loop the
      // network. `static: true` tells the SPA to stop the loop after one probe.
      // Capability manifest — the inverted contract. The edge is the minimal
      // node: it serves the artifact and nothing live. The SPA composes from this.
      if (ep === "capabilities") return jsonResponse({
        backend: "worker", artifact: true, models: "r2",
        live: false, steer: false, weights: false, mcp: false, write: false,
      });
      if (ep === "poll") return jsonResponse({ cmd: "none", static: true });
      if (ep === "navigate") return jsonResponse({});
      if (ep === "chat-poll") return jsonResponse({ reply: null, static: true });
      if (ep === "chat-drain") return jsonResponse({ messages: [] });
      if (ep === "live-status") return jsonResponse({ loaded: false, static: true, status: "static deployment — no live backend" });
      if (ep === "signals") return jsonResponse([]);

      const q = (k, d) => url.searchParams.get(k) ?? d;
      if (prefix) {
        if (ep === "pc-full") return await pcFull(env, prefix, parseInt(q("pc", "0")));
        if (ep === "pc-column") return await pcColumn(env, prefix, parseInt(q("pc", "0")), parseInt(q("layer", "0")));
        if (ep === "token-pca") return await tokenPca(env, prefix, parseInt(q("token", "0")));
        if (ep === "vocab-token") return await vocabToken(env, prefix, parseInt(q("row", "-1")));
        if (ep === "vocab-meta") {
          const vm = await r2json(env, `${prefix}/decomp/vocab_meta.json`);
          return vm ? jsonResponse(vm) : jsonResponse({ error: "no full-vocab projection" }, 404);
        }
        if (ep === "vocab-tokens") {
          const vt = await r2json(env, `${prefix}/decomp/vocab_tokens.json`);
          return vt ? jsonResponse(vt) : jsonResponse({ error: "no full-vocab projection" }, 404);
        }
        if (ep === "token-layer") return await tokenLayer(env, prefix, parseInt(q("token", "0")), parseInt(q("layer", "0")));
        if (ep === "token-hover") return await tokenHover(env, prefix, parseInt(q("token", "0")), parseInt(q("layer", "0")));
        if (ep === "neuron-field") return await neuronField(env, prefix, parseInt(q("token", "0")));
        if (ep === "token-bio") return await tokenBio(env, prefix, parseInt(q("token", "0")));
        if (ep === "gate-heatmap") return await serveRaw(env, `${prefix}/decomp/gate_heatmap.npy`);
        if (ep === "delta-scores") return await serveRaw(env, `${prefix}/decomp/delta_scores.bin`);
        if (ep === "weight-align-all") {
          const wa = await r2json(env, `${prefix}/decomp/weight_alignment.json`);
          return wa ? jsonResponse(wa) : jsonResponse([], 200);
        }
        if (ep === "weight-align") {
          const wa = await r2json(env, `${prefix}/decomp/weight_alignment.json`);
          const L = parseInt(q("layer", "0"));
          if (Array.isArray(wa)) { const hit = wa.find(x => x.layer === L) ?? wa[L] ?? {}; return jsonResponse(hit); }
          return jsonResponse({});
        }
        if (ep === "token-bundle") {
          const csv = (s) => (s ? s.split(",").filter(Boolean).map(Number) : []);
          return await tokenBundle(env, prefix, csv(q("full")), csv(q("hover")), parseInt(q("layer", "0")));
        }
        if (ep === "token-neurons") return jsonResponse(await r2json(env, `${prefix}/decomp/neuron_importance.json`) ?? { token_idx: parseInt(q("token", "0")), layers: [] });
        if (ep === "falsification") return jsonResponse(await r2json(env, `${prefix}/decomp/falsification.json`) ?? { random_baseline: [], top_pcs: [] });
        if (ep === "token-predicts") {
          // Captured-vocab logit lens, precomputed into token_predicts.bin (TPRD):
          // [N, L, K] of (uint32 mri_idx, f16 prob, f16 logit), token-major.
          // Range-read one (token,layer) slice, join mri_idx→text via tokens.json.
          const key = `${prefix}/decomp/token_predicts.bin`;
          const h = await r2range(env, key, 0, 16);
          if (!h) return jsonResponse({ top_k: [] });
          const hv = new DataView(h);
          if (new TextDecoder().decode(new Uint8Array(h, 0, 4)) !== "TPRD") return jsonResponse({ top_k: [] });
          const N = hv.getUint32(4, true), L = hv.getUint32(8, true), K = hv.getUint32(12, true);
          const token = parseInt(q("token", "0")), layer = parseInt(q("layer", "0"));
          const want = Math.min(parseInt(q("k", "8")) || 8, K);
          if (token < 0 || token >= N || layer < 0 || layer >= L) return jsonResponse({ top_k: [] });
          const stride = K * 8;
          const slab = await r2range(env, key, 16 + (token * L + layer) * stride, stride);
          if (!slab) return jsonResponse({ top_k: [] });
          const dv = new DataView(slab);
          const toks = await r2json(env, `${prefix}/decomp/tokens.json`);
          const texts = (toks && toks.token_texts) || [];
          const out = [];
          for (let i = 0; i < want; i++) {
            const mri_idx = dv.getUint32(i * 8, true);
            const prob = dv.getFloat16(i * 8 + 4, true);
            const logit = dv.getFloat16(i * 8 + 6, true);
            if (!Number.isFinite(prob)) continue;
            out.push({ text: texts[mri_idx] ?? `#${mri_idx}`, prob, logit, mri_idx });
          }
          return jsonResponse({ top_k: out });
        }
        if (ep === "token-resolve") {
          // Cross-model compare resolves a token by text (tokenizers fragment
          // differently, so raw index equality isn't the same concept). Pure
          // lookup in the target model's tokens.json — no compute.
          const toks = await r2json(env, `${prefix}/decomp/tokens.json`);
          const txt = q("text", "");
          const idx = toks && Array.isArray(toks.token_texts) ? toks.token_texts.indexOf(txt) : -1;
          const out = { idx, text: txt };
          if (idx < 0) {
            // Not in the sampled cloud — the full-vocab projection may still
            // know it (every tokenizer token is addressable there).
            const vt = await r2json(env, `${prefix}/decomp/vocab_tokens.json`);
            if (Array.isArray(vt)) {
              const row = vt.indexOf(txt);
              if (row >= 0) out.vocab_row = row;
            }
          }
          return jsonResponse(out);
        }
        if (ep === "norms") return jsonResponse(await r2json(env, `${prefix}/norms.json`) ?? {});
        if (ep === "baselines") return jsonResponse(await r2json(env, `${prefix}/baselines.json`) ?? {});
        if (ep === "token-attn") return jsonResponse({ token_idx: parseInt(q("token", "0")), layers: [] });
      }

      if (ep === "decomp-meta" && prefix) {
        const meta = await r2json(env, `${prefix}/decomp/meta.json`);
        return meta ? jsonResponse(meta) : jsonResponse({ error: "no decomp meta" }, 404);
      }

      if (ep === "serve-meta" && prefix) {
        return await serveMeta(env, prefix);
      }

      if (ep === "decomp" && prefix) {
        const layer = parseInt(url.searchParams.get("layer") ?? "0", 10);
        return await decompLayer(env, prefix, layer);
      }

      if (ep === "cloud-bundle" && prefix) {
        const csv = (s) => (s ? s.split(",").filter(Boolean).map(Number) : []);
        const full = csv(url.searchParams.get("full"));
        const med = csv(url.searchParams.get("medium"));
        const step = parseInt(url.searchParams.get("step") ?? "10", 10);
        return await cloudBundle(env, prefix, full, med, step);
      }

      return jsonResponse({ error: `endpoint not yet wired: ${ep}` }, 501);
    } catch (err) {
      return jsonResponse({ error: String(err?.stack || err) }, 500);
    }
  },
};
