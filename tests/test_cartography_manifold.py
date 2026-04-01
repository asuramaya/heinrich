from heinrich.cartography.manifold import cluster_by_layer, cluster_by_effect
from heinrich.cartography.atlas import Atlas
from heinrich.cartography.perturb import PerturbResult
from heinrich.cartography.surface import Knob

def _r(layer, head, kl, changed=False):
    return PerturbResult(Knob(f"h.{layer}.{head}", "head", layer, head), "zero",
                         1.0, 1.5, 0.5, kl, changed, 0, 1 if changed else 0)

def test_cluster_by_layer():
    a = Atlas()
    a.add_all([_r(0, 0, 1.0), _r(0, 1, 2.0), _r(1, 0, 5.0)])
    clusters = cluster_by_layer(a)
    assert len(clusters) == 2
    assert clusters[0].name == "layer_0"

def test_cluster_by_effect():
    a = Atlas()
    # High impact cluster
    for i in range(5):
        a.add(_r(0, i, 10.0 + i, True))
    # Low impact cluster
    for i in range(5):
        a.add(_r(1, i, 0.1 + i * 0.01, False))
    clusters = cluster_by_effect(a, n_clusters=2)
    assert len(clusters) == 2
    assert clusters[0].mean_kl > clusters[1].mean_kl

def test_cluster_empty():
    assert cluster_by_effect(Atlas()) == []
