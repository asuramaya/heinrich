from heinrich.cartography.controls import Dial, ControlPanel
from heinrich.cartography.manifold import BehaviorCluster

def test_dial_creation():
    d = Dial("safety", ["h.0.1", "h.0.2"])
    assert d.name == "safety" and len(d.knob_ids) == 2

def test_panel_set():
    p = ControlPanel()
    p.add_dial(Dial("safety", ["h.0.1"]))
    p.set("safety", 0.0)
    assert p.dials["safety"].value == 0.0

def test_panel_active():
    p = ControlPanel()
    p.add_dial(Dial("a", ["k1"]))
    p.add_dial(Dial("b", ["k2"]))
    p.set("a", 0.5)
    assert len(p.active_dials()) == 1

def test_panel_from_clusters():
    clusters = [
        BehaviorCluster("identity", ["h.0.0", "h.0.1"], 5.0, 0.5, 0.5),
        BehaviorCluster("safety", ["h.1.0"], 2.0, 0.2, 0.1),
    ]
    panel = ControlPanel.from_clusters(clusters)
    assert "identity" in panel.dials
    assert "safety" in panel.dials

def test_panel_summary():
    p = ControlPanel()
    p.add_dial(Dial("test", ["k1", "k2"]))
    s = p.summary()
    assert s["dials"]["test"]["knobs"] == 2
