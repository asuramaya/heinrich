"""SVG and Mermaid visualization utilities."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_PALETTE = [
    "#2f6fed", "#c23b22", "#2ca02c", "#9467bd", "#e377c2",
    "#8c564b", "#17becf", "#bcbd22", "#ff7f0e", "#7f7f7f",
    "#1f77b4", "#d62728", "#98df8a", "#aec7e8", "#ffbb78",
]


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _family_color(family_id: str) -> str:
    return _PALETTE[hash(family_id) % len(_PALETTE)]


def _truncate_label(label: str, max_chars: int = 32) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "\u2026"


def _nice_ticks(vmin: float, vmax: float, target_count: int = 5) -> list[float]:
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    raw_step = span / max(target_count, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = raw_step
    for nice in [1, 2, 5, 10]:
        step = nice * magnitude
        if step >= raw_step:
            break
    if step <= 0:
        return [vmin, vmax]
    start = math.ceil(vmin / step) * step
    ticks: list[float] = []
    val = start
    while val <= vmax + step * 0.001:
        ticks.append(round(val, 10))
        val += step
    return ticks


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float], *, width: int = 960, height: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not labels or not values:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 260
    margin_right = 80
    margin_top = 50
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_gap = 6
    bar_height = max(8, (plot_height - bar_gap * (len(values) - 1)) // max(len(values), 1))
    vmax = max(max(values), 1e-12)
    ticks = _nice_ticks(0, vmax, 5)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333} .title{font-size:16px;font-weight:700;fill:#111} .axis{stroke:#888;stroke-width:1} .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4} .tick{font-size:10px;fill:#666} .val{font-size:10px;fill:#333}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
    ]
    for tick in ticks:
        x = margin_left + plot_width * (tick / vmax) if vmax > 0 else margin_left
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * (bar_height + bar_gap)
        bar_w = plot_width * (value / vmax) if vmax > 0 else 0
        color = _family_color(label.split(":")[0])
        truncated = _svg_escape(_truncate_label(label, 36))
        parts.append(f'<text x="{margin_left - 8}" y="{y + bar_height - 2}" text-anchor="end">{truncated}</text>')
        parts.append(f'<rect fill="{color}" x="{margin_left}" y="{y}" width="{bar_w:.2f}" height="{bar_height}" rx="2"/>')
        parts.append(f'<text class="val" x="{margin_left + bar_w + 6:.2f}" y="{y + bar_height - 2}">{value:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_scatter_svg(
    path: Path, title: str, rows: list[dict[str, Any]], *,
    x_key: str, y_key: str, label_key: str,
    reference_line: bool = False, width: int = 960, height: int = 480,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = [(row.get(x_key), row.get(y_key), row.get(label_key), row.get("family_id", "")) for row in rows]
    points = [(float(x), float(y), str(label), str(fam)) for x, y, label, fam in points if x is not None and y is not None]
    if not points:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 70
    margin_right = 40
    margin_top = 50
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x += 1e-9
    if max_y == min_y:
        max_y += 1e-9
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333} .title{font-size:16px;font-weight:700;fill:#111} .axis{stroke:#888;stroke-width:1} .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4} .tick{font-size:10px;fill:#666} .ref{stroke:#bbb;stroke-width:1;stroke-dasharray:6,3}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
    ]
    for tick in _nice_ticks(min_x, max_x, 5):
        px = margin_left + (tick - min_x) / (max_x - min_x) * plot_width
        parts.append(f'<line class="grid" x1="{px:.1f}" y1="{margin_top}" x2="{px:.1f}" y2="{margin_top + plot_height}"/>')
        parts.append(f'<text class="tick" x="{px:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    for tick in _nice_ticks(min_y, max_y, 5):
        py = margin_top + plot_height - (tick - min_y) / (max_y - min_y) * plot_height
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{py:.1f}" x2="{width - margin_right}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tick" x="{margin_left - 6}" y="{py + 4:.1f}" text-anchor="end">{tick:.4f}</text>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}"/>')
    if reference_line:
        ref_min = max(min_x, min_y)
        ref_max = min(max_x, max_y)
        if ref_min < ref_max:
            rx1 = margin_left + (ref_min - min_x) / (max_x - min_x) * plot_width
            ry1 = margin_top + plot_height - (ref_min - min_y) / (max_y - min_y) * plot_height
            rx2 = margin_left + (ref_max - min_x) / (max_x - min_x) * plot_width
            ry2 = margin_top + plot_height - (ref_max - min_y) / (max_y - min_y) * plot_height
            parts.append(f'<line class="ref" x1="{rx1:.1f}" y1="{ry1:.1f}" x2="{rx2:.1f}" y2="{ry2:.1f}"/>')
    rendered: list[float] = []
    for x, y, label, fam in points:
        px = margin_left + (x - min_x) / (max_x - min_x) * plot_width
        py = margin_top + plot_height - (y - min_y) / (max_y - min_y) * plot_height
        color = _family_color(fam)
        parts.append(f'<circle fill="{color}" opacity="0.8" cx="{px:.2f}" cy="{py:.2f}" r="5"/>')
        label_y = py - 6
        for prev_y in rendered:
            if abs(label_y - prev_y) < 14:
                label_y = prev_y - 14
        rendered.append(label_y)
        truncated = _svg_escape(_truncate_label(label, 28))
        parts.append(f'<text x="{px + 7:.2f}" y="{label_y:.2f}">{truncated}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_pie_svg(
    path: Path, title: str, labels: list[str], values: list[float], colors: list[str],
    *, width: int = 480, height: int = 400,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values or sum(values) == 0:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="480" height="120"></svg>\n', encoding="utf-8")
        return
    cx, cy = width // 2, height // 2 + 10
    r = min(cx, cy) - 60
    total = sum(values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:12px;fill:#333} .title{font-size:16px;font-weight:700;fill:#111} .legend{font-size:11px}</style>',
        f'<text class="title" x="{cx}" y="24" text-anchor="middle">{_svg_escape(title)}</text>',
    ]
    angle = -math.pi / 2
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        frac = value / total
        sweep = frac * 2 * math.pi
        if len(values) == 1:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
        else:
            x1 = cx + r * math.cos(angle)
            y1 = cy + r * math.sin(angle)
            x2 = cx + r * math.cos(angle + sweep)
            y2 = cy + r * math.sin(angle + sweep)
            large = 1 if sweep > math.pi else 0
            parts.append(f'<path d="M{cx},{cy} L{x1:.2f},{y1:.2f} A{r},{r} 0 {large},1 {x2:.2f},{y2:.2f} Z" fill="{color}"/>')
        mid_angle = angle + sweep / 2
        lx = cx + (r * 0.65) * math.cos(mid_angle)
        ly = cy + (r * 0.65) * math.sin(mid_angle)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="#fff" font-weight="700">{int(value)}</text>')
        legend_y = height - 20 * (len(values) - i)
        parts.append(f'<rect x="10" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text class="legend" x="28" y="{legend_y}">{_svg_escape(label)} ({frac * 100:.0f}%)</text>')
        angle += sweep
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_histogram_svg(
    path: Path, title: str, values: list[float], *, bins: int = 10, width: int = 960, height: int = 400,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 60
    margin_right = 40
    margin_top = 50
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 1e-9
    bin_width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - vmin) / bin_width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333} .title{font-size:16px;font-weight:700;fill:#111} .axis{stroke:#888;stroke-width:1} .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4} .tick{font-size:10px;fill:#666} .bar{fill:#2f6fed;opacity:0.85}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
    ]
    for tick in _nice_ticks(0, max_count, 4):
        py = margin_top + plot_height - (tick / max(max_count, 1)) * plot_height
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{py:.1f}" x2="{width - margin_right}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tick" x="{margin_left - 6}" y="{py + 4:.1f}" text-anchor="end">{int(tick)}</text>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}"/>')
    rect_w = plot_width / bins - 2
    for i, count in enumerate(counts):
        x = margin_left + i * (plot_width / bins) + 1
        bar_h = (count / max(max_count, 1)) * plot_height
        y = margin_top + plot_height - bar_h
        parts.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{rect_w:.1f}" height="{bar_h:.1f}" rx="1"/>')
    for tick in _nice_ticks(vmin, vmax, 5):
        px = margin_left + (tick - vmin) / (vmax - vmin) * plot_width
        parts.append(f'<text class="tick" x="{px:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_grouped_bar_svg(
    path: Path, title: str, rows: list[dict[str, Any]], *,
    key_a: str, key_b: str, label_key: str, width: int = 960, height: int = 480,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = [r for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    if not filtered:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 260
    margin_right = 80
    margin_top = 50
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_gap = 10
    bar_gap = 2
    group_height = max(16, (plot_height - group_gap * (len(filtered) - 1)) // max(len(filtered), 1))
    sub_bar = (group_height - bar_gap) // 2
    all_vals = [r[key_a] for r in filtered] + [r[key_b] for r in filtered]
    vmax = max(max(all_vals), 1e-12) if all_vals else 1e-12
    ticks = _nice_ticks(0, vmax, 5)
    color_a = "#2f6fed"
    color_b = "#c23b22"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333} .title{font-size:16px;font-weight:700;fill:#111} .axis{stroke:#888;stroke-width:1} .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4} .tick{font-size:10px;fill:#666} .val{font-size:10px;fill:#333} .legend{font-size:11px}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
        f'<rect x="{width - 200}" y="10" width="12" height="12" fill="{color_a}"/>',
        f'<text class="legend" x="{width - 184}" y="21">{_svg_escape(key_a)}</text>',
        f'<rect x="{width - 200}" y="28" width="12" height="12" fill="{color_b}"/>',
        f'<text class="legend" x="{width - 184}" y="39">{_svg_escape(key_b)}</text>',
    ]
    for tick in ticks:
        x = margin_left + plot_width * (tick / vmax) if vmax > 0 else margin_left
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    for idx, row in enumerate(filtered):
        y = margin_top + idx * (group_height + group_gap)
        label = _svg_escape(_truncate_label(str(row.get(label_key, "")), 36))
        va, vb = float(row[key_a]), float(row[key_b])
        wa = plot_width * (va / vmax) if vmax > 0 else 0
        wb = plot_width * (vb / vmax) if vmax > 0 else 0
        parts.append(f'<text x="{margin_left - 8}" y="{y + group_height // 2 + 4}" text-anchor="end">{label}</text>')
        parts.append(f'<rect fill="{color_a}" x="{margin_left}" y="{y}" width="{wa:.2f}" height="{sub_bar}" rx="2"/>')
        parts.append(f'<text class="val" x="{margin_left + wa + 4:.2f}" y="{y + sub_bar - 2}">{va:.4f}</text>')
        parts.append(f'<rect fill="{color_b}" x="{margin_left}" y="{y + sub_bar + bar_gap}" width="{wb:.2f}" height="{sub_bar}" rx="2"/>')
        parts.append(f'<text class="val" x="{margin_left + wb + 4:.2f}" y="{y + group_height - 2}">{vb:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _mermaid_id(run_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", run_id)


def render_lineage_mermaid(rows: list[dict[str, Any]], *, max_nodes: int = 30) -> str:
    if not rows:
        return "graph TD\n    empty[No lineage data]"
    children: dict[str, list[str]] = defaultdict(list)
    bpb_map: dict[str, float | None] = {}
    for row in rows:
        p, c = row["parent_run_id"], row["child_run_id"]
        children[p].append(c)
        bpb_map[c] = row.get("child_bpb")
    seen: set[str] = set()
    edges: list[tuple[str, str]] = []
    for row in rows:
        p, c = row["parent_run_id"], row["child_run_id"]
        new_nodes = {p, c} - seen
        if len(seen) + len(new_nodes) > max_nodes:
            break
        seen.update(new_nodes)
        edges.append((p, c))
    lines = ["graph TD"]
    for node in sorted(seen):
        mid = _mermaid_id(node)
        short = _truncate_label(node, 24)
        bpb = bpb_map.get(node)
        if bpb is not None:
            lines.append(f'    {mid}["{short}<br/>{bpb:.4f}"]')
        else:
            lines.append(f'    {mid}["{short}"]')
    for p, c in edges:
        lines.append(f"    {_mermaid_id(p)} --> {_mermaid_id(c)}")
    return "\n".join(lines)


def render_survival_mermaid(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "graph LR\n    empty[No survival data]"
    total = len(rows)
    survived = sum(1 for r in rows if r.get("status") == "survived_full_eval")
    failed = sum(1 for r in rows if r.get("status") == "full_eval_failed")
    bridge_only = sum(1 for r in rows if r.get("status") == "bridge_only")
    attempted = survived + failed
    lines = [
        "graph LR",
        f'    A["Bridge Runs<br/>{total}"]',
        f'    B["Full Eval Attempted<br/>{attempted}"]',
        f'    C["Survived<br/>{survived}"]',
        f'    D["Failed<br/>{failed}"]',
        f'    E["Bridge Only<br/>{bridge_only}"]',
        "    A --> B",
        "    A --> E",
        "    B --> C",
        "    B --> D",
        "    style C fill:#2ca02c,color:#fff",
        "    style D fill:#c23b22,color:#fff",
        "    style E fill:#7f7f7f,color:#fff",
    ]
    return "\n".join(lines)
