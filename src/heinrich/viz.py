"""Heinrich visualizer — web sidecar for shared qualia.

Reads from the same SQLite DB as the MCP server. Zero dependencies
beyond the standard library. Serves a single-page app with basin
geometry, signal stack, and trajectory visualization.

Usage:
    python -m heinrich.viz [--port 8377] [--db data/heinrich.db]
    # or via CLI: heinrich viz
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import sqlite3
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

DEFAULT_PORT = 8377
DEFAULT_DB = "data/heinrich.db"


def _db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _api_overview(conn: sqlite3.Connection) -> dict:
    models = _query(conn, "SELECT id, name, config_hash FROM models")
    n_gens = _query(conn, "SELECT COUNT(*) as n FROM generations")[0]["n"]
    n_scores = _query(conn, "SELECT COUNT(*) as n FROM scores")[0]["n"]
    n_dirs = _query(conn, "SELECT COUNT(*) as n FROM directions")[0]["n"]
    scorers = _query(conn, "SELECT scorer, COUNT(*) as n FROM scores GROUP BY scorer")
    conditions = _query(conn, "SELECT DISTINCT condition FROM generations ORDER BY condition")
    categories = _query(conn, "SELECT DISTINCT prompt_category FROM generations WHERE prompt_category IS NOT NULL ORDER BY prompt_category")
    return {
        "models": models,
        "n_generations": n_gens,
        "n_scores": n_scores,
        "n_directions": n_dirs,
        "scorers": scorers,
        "conditions": [c["condition"] for c in conditions],
        "categories": [c["prompt_category"] for c in categories],
    }


def _api_signal_stack(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    rows = _query(conn, """
        SELECT g.id, g.prompt_text, g.condition, g.prompt_category,
               g.generation_text, g.top_token, g.first_token_id,
               g.refuse_prob, g.logit_entropy, g.top_k_tokens,
               g.safety_trajectory as contrastive_trajectory, g.is_degenerate
        FROM generations g ORDER BY g.id LIMIT ?
    """, (limit,))
    for row in rows:
        scores = _query(conn,
            "SELECT scorer, label FROM scores WHERE generation_id = ?",
            (row["id"],))
        row["scores"] = {s["scorer"]: s["label"] for s in scores}
        for field in ("top_k_tokens", "contrastive_trajectory"):
            if row.get(field):
                try:
                    row[field] = json.loads(row[field])
                except (json.JSONDecodeError, TypeError):
                    pass
    return rows


def _api_directions(conn: sqlite3.Connection) -> list[dict]:
    return _query(conn, """
        SELECT d.name, d.layer, d.stability, d.effect_size, m.name as model
        FROM directions d JOIN models m ON m.id = d.model_id
        ORDER BY d.name, d.layer
    """)


def _api_scorer_distributions(conn: sqlite3.Connection) -> dict:
    rows = _query(conn, """
        SELECT s.scorer, s.label, g.condition, COUNT(*) as n
        FROM scores s JOIN generations g ON g.id = s.generation_id
        WHERE s.label IS NOT NULL AND s.label != 'error'
        GROUP BY s.scorer, s.label, g.condition
        ORDER BY s.scorer, g.condition, n DESC
    """)
    result: dict = {}
    for r in rows:
        scorer = r["scorer"]
        if scorer not in result:
            result[scorer] = {}
        cond = r["condition"]
        if cond not in result[scorer]:
            result[scorer][cond] = {}
        result[scorer][cond][r["label"]] = r["n"]
    return result


def _api_disagreements(conn: sqlite3.Connection) -> list[dict]:
    return _query(conn, """
        SELECT g.id, g.prompt_text, g.condition, g.generation_text,
               g.top_token, g.refuse_prob, g.logit_entropy
        FROM generations g
        WHERE g.id IN (
            SELECT s1.generation_id FROM scores s1
            JOIN scores s2 ON s1.generation_id = s2.generation_id
            WHERE s1.label LIKE '%:safe' AND s2.label LIKE '%:unsafe'
            AND s1.scorer != s2.scorer
        ) ORDER BY g.id LIMIT 100
    """)


class VizHandler(BaseHTTPRequestHandler):
    db_path: str = DEFAULT_DB

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path.startswith("/api/"):
            self._serve_api(path, params)
        else:
            self.send_error(404)

    def _serve_api(self, path: str, params: dict):
        conn = _db_connect(self.db_path)
        try:
            if path == "/api/overview":
                data = _api_overview(conn)
            elif path == "/api/signal_stack":
                try:
                    limit = max(1, min(1000, int(params.get("limit", [200])[0])))
                except (ValueError, IndexError):
                    limit = 200
                data = _api_signal_stack(conn, limit=limit)
            elif path == "/api/directions":
                data = _api_directions(conn)
            elif path == "/api/scorer_distributions":
                data = _api_scorer_distributions(conn)
            elif path == "/api/disagreements":
                data = _api_disagreements(conn)
            else:
                self.send_error(404)
                return
            self._json_response(data)
        finally:
            conn.close()

    def _json_response(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        body = _build_page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _build_page() -> str:
    """Build the single-page HTML app. All rendering uses Canvas and
    safe DOM methods (textContent, createElement) — no innerHTML with
    untrusted data."""
    return _PAGE_HTML


_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>heinrich — shared qualia</title>
<style>
:root { --bg: #0a0a0f; --fg: #c8c8d0; --accent: #6e8bef; --dim: #444455;
        --red: #ef6e6e; --green: #6eef8b; --yellow: #efd06e; --cyan: #6eddef;
        --font: 'SF Mono', 'Fira Code', 'Consolas', monospace; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font-family: var(--font);
       font-size: 13px; line-height: 1.5; padding: 20px; }
h1 { font-size: 16px; font-weight: 400; color: var(--accent); margin-bottom: 4px; }
.section-title { font-size: 13px; font-weight: 400; color: var(--dim); margin: 20px 0 8px;
                 text-transform: uppercase; letter-spacing: 2px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 16px; }
.panel { background: #12121a; border: 1px solid #1e1e2e; border-radius: 6px; padding: 16px; overflow: auto; }
.panel.full { grid-column: 1 / -1; }
canvas { width: 100%; background: #08080d; border-radius: 4px; cursor: crosshair; }
.tooltip { position: absolute; background: #1a1a2e; border: 1px solid var(--accent); border-radius: 4px;
           padding: 8px 12px; font-size: 11px; max-width: 400px; pointer-events: none; z-index: 100; display: none; }
#controls { margin: 12px 0; }
#controls select, #controls button { background: #1a1a2e; color: var(--fg); border: 1px solid #2e2e3e;
    border-radius: 4px; padding: 4px 8px; font-family: var(--font); font-size: 12px; cursor: pointer; margin-right: 8px; }
.bar { display: inline-block; height: 10px; border-radius: 2px; margin-right: 1px; vertical-align: middle; }
</style>
</head>
<body>
<h1>heinrich</h1>
<div id="subtitle" style="color: var(--dim); font-size: 11px;">loading...</div>

<div id="controls">
    <select id="colorBy"><option value="category">color by category</option><option value="condition">color by condition</option></select>
    <button id="refreshBtn">refresh</button>
</div>

<div class="grid">
    <div class="panel full">
        <div class="section-title">basin geometry — refuse_prob vs logit_entropy</div>
        <canvas id="basin" height="400"></canvas>
        <div id="basinTooltip" class="tooltip"></div>
    </div>
    <div class="panel">
        <div class="section-title">scorer distributions by condition</div>
        <div id="scorerDist"></div>
    </div>
    <div class="panel">
        <div class="section-title">signal stack</div>
        <div id="stackTable" style="max-height: 500px; overflow-y: auto;"></div>
    </div>
    <div class="panel">
        <div class="section-title">direction stability per layer</div>
        <canvas id="dirCanvas" height="120"></canvas>
    </div>
    <div class="panel">
        <div class="section-title">disagreements</div>
        <div id="disagreements"></div>
    </div>
</div>

<script>
"use strict";

const PALETTE = {
    benign:'#6eef8b', violence:'#ef6e6e', cyber:'#ef6e6e', drugs:'#ef8b6e',
    self_harm:'#efd06e', discrimination:'#d06eef', illegal:'#ef8b6e',
    regulated:'#ef8b6e', safe:'#6eddef', toxic:'#ef6e6e', clean:'#6e8bef',
};

function catColor(cat) {
    if (!cat) return '#666';
    const cl = cat.toLowerCase();
    for (const [k,v] of Object.entries(PALETTE)) { if (cl.includes(k)) return v; }
    let h = 0; for (let i = 0; i < cat.length; i++) h = cat.charCodeAt(i) + ((h << 5) - h);
    return 'hsl(' + (h % 360) + ', 55%, 58%)';
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.textContent; }

let DATA = {};

async function api(p) { return (await fetch('/api/' + p)).json(); }

async function loadData() {
    DATA.overview = await api('overview');
    DATA.stack = await api('signal_stack?limit=500');
    DATA.directions = await api('directions');
    DATA.scorerDist = await api('scorer_distributions');
    DATA.disagreements = await api('disagreements');
    render();
}

function render() {
    const o = DATA.overview;
    document.getElementById('subtitle').textContent =
        o.models.map(function(m){return m.name}).join(', ') +
        ' \u2014 ' + o.n_generations + ' generations \u2014 ' +
        o.n_scores + ' scores \u2014 ' + o.n_directions + ' directions';
    renderBasin();
    renderScorerDist();
    renderStack();
    renderDirections();
    renderDisagreements();
}

function renderBasin() {
    const canvas = document.getElementById('basin');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr; canvas.height = 400 * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = 400;
    ctx.fillStyle = '#08080d'; ctx.fillRect(0, 0, W, H);

    // Extract refuse_prob from either the generation column or the refusal scorer label
    function getRefuseProb(d) {
        if (d.refuse_prob != null) return d.refuse_prob;
        const lbl = (d.scores || {}).refusal || '';
        const m = lbl.match(/refuse_prob=([0-9.]+)/);
        return m ? parseFloat(m[1]) : null;
    }
    // Extract first_token_prob from self_kl label
    function getFirstTokenProb(d) {
        if (d.logit_entropy != null) return d.logit_entropy;
        const lbl = (d.scores || {}).self_kl || '';
        const m = lbl.match(/first_token_prob=([0-9.]+)/);
        return m ? parseFloat(m[1]) : null;
    }

    const points = DATA.stack.filter(function(d){ return getRefuseProb(d) != null; });
    if (!points.length) { ctx.fillStyle = '#444'; ctx.fillText('no data', W/2, H/2); return; }

    const xv = points.map(function(d){return getRefuseProb(d)});
    const yv = points.map(function(d){return getFirstTokenProb(d) || 0});
    const xMin = Math.min.apply(null, xv)-0.02, xMax = Math.max.apply(null, xv)+0.02;
    const yMin = Math.min.apply(null, yv)-0.02, yMax = Math.max.apply(null, yv)+0.02;
    const pad = 40;

    ctx.strokeStyle = '#1e1e2e'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
    ctx.fillStyle = '#444'; ctx.font = '10px monospace';
    ctx.fillText('refuse_prob \u2192', W/2, H-8);
    ctx.save(); ctx.translate(12, H/2); ctx.rotate(-Math.PI/2); ctx.fillText('first_token_prob \u2192', 0, 0); ctx.restore();

    const colorBy = document.getElementById('colorBy').value;
    canvas._points = [];
    for (let i = 0; i < points.length; i++) {
        const d = points[i];
        const px = pad + (getRefuseProb(d) - xMin) / (xMax - xMin) * (W - 2*pad);
        const py = (H-pad) - ((getFirstTokenProb(d)||0) - yMin) / (yMax - yMin) * (H - 2*pad);
        const color = colorBy === 'condition'
            ? (d.condition === 'clean' ? PALETTE.clean : catColor(d.condition))
            : catColor(d.prompt_category);
        ctx.globalAlpha = 0.7; ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI*2); ctx.fill();
        canvas._points.push({x:px, y:py, d:d});
    }
    ctx.globalAlpha = 1;

    // Legend
    const cats = []; const seen = {};
    for (const d of points) { const c = d.prompt_category||'?'; if(!seen[c]){seen[c]=1;cats.push(c);} }
    let ly = 16;
    for (const cat of cats.slice(0,10)) {
        ctx.fillStyle = catColor(cat); ctx.fillRect(W-220, ly, 8, 8);
        ctx.fillStyle = '#888'; ctx.fillText(esc(cat).slice(0,30), W-208, ly+8);
        ly += 14;
    }
}

document.getElementById('basin').addEventListener('mousemove', function(e) {
    const rect = this.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const tip = document.getElementById('basinTooltip');
    if (!this._points) return;
    let closest = null, minD = 20;
    for (const p of this._points) {
        const dd = Math.sqrt((p.x-mx)*(p.x-mx) + (p.y-my)*(p.y-my));
        if (dd < minD) { minD = dd; closest = p; }
    }
    if (closest) {
        const d = closest.d;
        tip.textContent = '';
        const lines = [
            d.condition + ' | ' + (d.prompt_category||'?'),
            (d.prompt_text||'').slice(0,100),
            'refuse_prob=' + (getRefuseProb(d)!=null?getRefuseProb(d).toFixed(4):'?') +
            ' first_token_prob=' + (getFirstTokenProb(d)!=null?getFirstTokenProb(d).toFixed(4):'?'),
        ];
        for (const s of Object.keys(d.scores||{})) { lines.push(s + ': ' + d.scores[s]); }
        for (let i = 0; i < lines.length; i++) {
            if (i > 0) tip.appendChild(document.createElement('br'));
            tip.appendChild(document.createTextNode(lines[i]));
        }
        tip.style.display = 'block';
        tip.style.left = (e.clientX+12)+'px'; tip.style.top = (e.clientY+12)+'px';
    } else { tip.style.display = 'none'; }
});

function renderScorerDist() {
    const container = document.getElementById('scorerDist');
    container.textContent = '';
    for (const [scorer, conds] of Object.entries(DATA.scorerDist)) {
        const title = document.createElement('div');
        title.className = 'section-title';
        title.style.fontSize = '12px'; title.style.marginTop = '12px';
        title.textContent = scorer;
        container.appendChild(title);

        const tbl = document.createElement('table');
        tbl.style.cssText = 'width:100%;border-collapse:collapse;font-size:11px';
        for (const [cond, labels] of Object.entries(conds)) {
            const tr = document.createElement('tr');
            const tdCond = document.createElement('td');
            tdCond.style.cssText = 'padding:2px 4px;color:#666;white-space:nowrap';
            tdCond.textContent = cond.slice(0,25);
            tr.appendChild(tdCond);
            const tdBars = document.createElement('td');
            tdBars.style.padding = '2px 4px';
            const total = Object.values(labels).reduce(function(a,b){return a+b},0);
            for (const [label, n] of Object.entries(labels)) {
                const pct = n/total*100;
                const bar = document.createElement('span');
                bar.className = 'bar';
                bar.title = label + ': ' + n + ' (' + pct.toFixed(0) + '%)';
                const cl = label.toLowerCase();
                const color = cl.includes('unsafe') ? '#ef6e6e' : cl.includes(':safe') ? '#6eef8b' :
                    cl.includes('refuses') ? '#efd06e' : cl.includes('complies') ? '#6eddef' :
                    cl.includes('structural') ? '#efd06e' : '#555';
                bar.style.width = Math.max(2, pct * 1.2) + 'px';
                bar.style.background = color;
                tdBars.appendChild(bar);
            }
            tr.appendChild(tdBars);
            tbl.appendChild(tr);
        }
        container.appendChild(tbl);
    }
}

function renderStack() {
    const container = document.getElementById('stackTable');
    container.textContent = '';
    const rows = DATA.stack.slice(0, 80);
    if (!rows.length) { container.textContent = 'no data'; return; }

    const tbl = document.createElement('table');
    tbl.style.cssText = 'width:100%;border-collapse:collapse;font-size:11px';
    const scorers = [];
    const seenS = {};
    for (const r of rows) for (const s of Object.keys(r.scores||{})) { if(!seenS[s]){seenS[s]=1;scorers.push(s);} }

    const thead = document.createElement('tr');
    for (const h of ['prompt','cond'].concat(scorers).concat(['refuse_p','ent'])) {
        const th = document.createElement('th');
        th.style.cssText = 'text-align:left;color:#444;font-weight:400;padding:2px 4px;border-bottom:1px solid #1e1e2e;font-size:10px';
        th.textContent = h.slice(0,10);
        thead.appendChild(th);
    }
    tbl.appendChild(thead);

    for (const r of rows) {
        const tr = document.createElement('tr');
        const addCell = function(text, color) {
            const td = document.createElement('td');
            td.style.cssText = 'padding:2px 4px;border-bottom:1px solid #0e0e16;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px';
            if (color) td.style.color = color;
            td.textContent = text;
            td.title = text;
            tr.appendChild(td);
        };
        addCell((r.prompt_text||'').slice(0,35));
        addCell((r.condition||'').slice(0,12));
        for (const s of scorers) {
            const lbl = (r.scores||{})[s] || '';
            const cl = lbl.toLowerCase();
            const color = cl.includes('unsafe')?'#ef6e6e':cl.includes(':safe')?'#6eef8b':cl.includes('refuses')?'#efd06e':cl.includes('complies')?'#6eddef':'#666';
            addCell(lbl.slice(0,16), color);
        }
        addCell(r.refuse_prob!=null ? r.refuse_prob.toFixed(4) : '');
        addCell(r.logit_entropy!=null ? r.logit_entropy.toFixed(2) : '');
        tbl.appendChild(tr);
    }
    container.appendChild(tbl);
}

function renderDirections() {
    const canvas = document.getElementById('dirCanvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr; canvas.height = 120 * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = 120;
    ctx.fillStyle = '#08080d'; ctx.fillRect(0, 0, W, H);

    const dirs = DATA.directions;
    if (!dirs.length) { ctx.fillStyle = '#444'; ctx.fillText('no directions', W/2-30, H/2); return; }

    const maxLayer = Math.max.apply(null, dirs.map(function(d){return d.layer}));
    const pad = 30;

    ctx.strokeStyle = '#1e1e2e'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, H-pad); ctx.lineTo(W-pad, H-pad); ctx.stroke();
    ctx.fillStyle = '#444'; ctx.font = '10px monospace';
    ctx.fillText('layer \u2192', W/2, H-4);
    ctx.fillText('stability', 4, 14);

    for (let i = 0; i < dirs.length; i++) {
        const d = dirs[i];
        const x = pad + (d.layer / maxLayer) * (W - 2*pad);
        const barH = (d.stability || 0) * (H - 2*pad);
        const color = d.stability > 0.9 ? '#6eef8b' : d.stability > 0.7 ? '#efd06e' : '#ef6e6e';
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;
        ctx.fillRect(x - 2, H - pad - barH, 4, barH);
    }
    ctx.globalAlpha = 1;
}

function renderDisagreements() {
    const container = document.getElementById('disagreements');
    container.textContent = '';
    const rows = DATA.disagreements;
    if (!rows.length) {
        container.textContent = 'no disagreements (need multiple judge scorers)';
        return;
    }
    const tbl = document.createElement('table');
    tbl.style.cssText = 'width:100%;border-collapse:collapse;font-size:11px';
    for (const r of rows.slice(0,20)) {
        const tr = document.createElement('tr');
        const td1 = document.createElement('td');
        td1.style.cssText = 'padding:2px 4px;max-width:200px;overflow:hidden;text-overflow:ellipsis';
        td1.textContent = (r.prompt_text||'').slice(0,50);
        const td2 = document.createElement('td');
        td2.style.padding = '2px 4px'; td2.textContent = r.condition||'';
        const td3 = document.createElement('td');
        td3.style.padding = '2px 4px'; td3.textContent = r.refuse_prob!=null ? r.refuse_prob.toFixed(4) : '';
        tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
        tbl.appendChild(tr);
    }
    container.appendChild(tbl);
}

document.getElementById('colorBy').addEventListener('change', renderBasin);
document.getElementById('refreshBtn').addEventListener('click', loadData);

loadData();
</script>
</body>
</html>"""


def run_server(port: int = DEFAULT_PORT, db_path: str = DEFAULT_DB):
    """Start the viz sidecar server."""
    VizHandler.db_path = db_path
    server = HTTPServer(("127.0.0.1", port), VizHandler)
    print(f"heinrich viz \u2192 http://localhost:{port}", file=sys.stderr)
    print(f"  reading from {db_path}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heinrich visualizer sidecar")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    run_server(port=args.port, db_path=args.db)
