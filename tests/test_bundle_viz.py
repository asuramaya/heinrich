"""Tests for bundle/viz.py"""
import math
import tempfile
from pathlib import Path
from heinrich.bundle.viz import (
    write_bar_svg,
    write_scatter_svg,
    write_pie_svg,
    write_histogram_svg,
    write_grouped_bar_svg,
    render_lineage_mermaid,
    render_survival_mermaid,
)


def test_write_bar_svg_basic(tmp_path):
    p = tmp_path / "bar.svg"
    write_bar_svg(p, "Test Bar", ["a", "b", "c"], [1.0, 2.0, 3.0])
    content = p.read_text()
    assert "<svg" in content
    assert "Test Bar" in content


def test_write_bar_svg_empty(tmp_path):
    p = tmp_path / "empty.svg"
    write_bar_svg(p, "Empty", [], [])
    assert p.exists()


def test_write_scatter_svg_basic(tmp_path):
    p = tmp_path / "scatter.svg"
    rows = [
        {"x": 1.0, "y": 1.0, "label": "a", "family_id": "fam1"},
        {"x": 2.0, "y": 1.5, "label": "b", "family_id": "fam2"},
    ]
    write_scatter_svg(p, "Scatter", rows, x_key="x", y_key="y", label_key="label")
    assert "<svg" in p.read_text()


def test_write_pie_svg_basic(tmp_path):
    p = tmp_path / "pie.svg"
    write_pie_svg(p, "Pie", ["A", "B"], [3.0, 7.0], ["#ff0000", "#00ff00"])
    assert "<svg" in p.read_text()


def test_write_histogram_svg_basic(tmp_path):
    p = tmp_path / "hist.svg"
    write_histogram_svg(p, "Histogram", [0.1, 0.2, 0.3, 0.4, 0.5])
    assert "<svg" in p.read_text()


def test_write_grouped_bar_svg_basic(tmp_path):
    p = tmp_path / "grouped.svg"
    rows = [{"label": "x", "a": 1.0, "b": 1.5}, {"label": "y", "a": 0.9, "b": 1.1}]
    write_grouped_bar_svg(p, "Grouped", rows, key_a="a", key_b="b", label_key="label")
    assert "<svg" in p.read_text()


def test_render_lineage_mermaid_basic():
    rows = [
        {"parent_run_id": "parent", "child_run_id": "child", "child_bpb": 1.23},
    ]
    diagram = render_lineage_mermaid(rows)
    assert "graph TD" in diagram
    assert "parent" in diagram
    assert "child" in diagram


def test_render_lineage_mermaid_empty():
    diagram = render_lineage_mermaid([])
    assert "No lineage" in diagram


def test_render_survival_mermaid_basic():
    rows = [
        {"status": "survived_full_eval"},
        {"status": "bridge_only"},
    ]
    diagram = render_survival_mermaid(rows)
    assert "graph LR" in diagram


def test_render_survival_mermaid_empty():
    diagram = render_survival_mermaid([])
    assert "No survival" in diagram


# --- Additional coverage with tempfile paths ---

import tempfile


def test_write_bar_svg():
    p = Path(tempfile.mkdtemp()) / "bar.svg"
    write_bar_svg(p, "Test Bar", ["a", "b", "c"], [1.0, 2.0, 3.0])
    assert p.exists()
    content = p.read_text()
    assert "<svg" in content
    assert "Test Bar" in content


def test_write_scatter_svg():
    p = Path(tempfile.mkdtemp()) / "scatter.svg"
    rows = [{"x": 1.0, "y": 2.0, "label": "a", "family_id": "f1"},
            {"x": 3.0, "y": 4.0, "label": "b", "family_id": "f2"}]
    write_scatter_svg(p, "Scatter", rows, x_key="x", y_key="y", label_key="label")
    assert p.exists()
    assert "<svg" in p.read_text()


def test_write_pie_svg():
    p = Path(tempfile.mkdtemp()) / "pie.svg"
    write_pie_svg(p, "Pie", ["a", "b"], [60, 40], ["#ff0000", "#0000ff"])
    assert p.exists()
    assert "<svg" in p.read_text()


def test_write_histogram_svg():
    p = Path(tempfile.mkdtemp()) / "hist.svg"
    write_histogram_svg(p, "Hist", [1.0, 2.0, 2.5, 3.0, 4.0])
    assert p.exists()
    assert "<svg" in p.read_text()


def test_write_grouped_bar_svg():
    p = Path(tempfile.mkdtemp()) / "grouped.svg"
    rows = [{"a": 1.0, "b": 2.0, "label": "x"}, {"a": 3.0, "b": 4.0, "label": "y"}]
    write_grouped_bar_svg(p, "Grouped", rows, key_a="a", key_b="b", label_key="label")
    assert p.exists()
    assert "<svg" in p.read_text()


def test_render_lineage_mermaid():
    rows = [{"parent_run_id": "p1", "child_run_id": "c1", "child_bpb": 0.5}]
    text = render_lineage_mermaid(rows)
    assert "graph" in text.lower() or "flowchart" in text.lower()


def test_render_survival_mermaid():
    rows = [{"status": "survived", "run_id": "r1"}]
    text = render_survival_mermaid(rows)
    assert len(text) > 0
