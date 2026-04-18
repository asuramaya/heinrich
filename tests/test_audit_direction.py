"""Smoke tests for profile/audit.py: 5-test direction audit pipeline.

Uses Qwen2-0.5B + cities/common_claim CSVs for speed. Validates:
- audit_direction() runs end-to-end and returns the expected keys
- verdict is one of {robust_feature, partial, falsified}
- tests_passed booleans match the underlying metrics
"""
from __future__ import annotations
import os
import pytest

pytest.importorskip("mlx")

CITIES = "papers/lie-detection/data/cities.csv"
CC = "papers/lie-detection/data/common_claim.csv"


@pytest.mark.skipif(
    not os.path.exists(CITIES), reason="cities.csv dataset not available"
)
def test_audit_direction_smoke():
    from heinrich.profile.audit import audit_direction

    r = audit_direction(
        model_id="Qwen/Qwen2-0.5B-Instruct",
        datasets=[CITIES, CC],
        layer=14,
        n_per_class=40,
        n_bootstrap=20,
        n_permutation=50,
        seed=42,
    )
    for key in [
        "in_domain_test_acc", "cohens_d", "boot_p5", "perm_p95", "snr",
        "transfer", "vocab", "tests_passed", "verdict",
    ]:
        assert key in r, f"missing key: {key}"
    assert r["verdict"] in {"robust_feature", "partial", "falsified"}
    # Transfer dict covers exactly the extra datasets
    assert set(r["transfer"].keys()) == {CC}
    # tests_passed has all 5 flags
    assert set(r["tests_passed"].keys()) == {"in_domain", "bootstrap",
                                              "permutation", "transfer", "vocab"}
    # Bootstrap p5 is a cosine ∈ [-1, 1]
    assert -1.0 <= r["boot_p5"] <= 1.0
    # SNR is non-negative
    assert r["snr"] >= 0
    # Vocab structure
    v = r["vocab"]
    assert "pos_top" in v and "neg_top" in v
    assert isinstance(v["vocab_pass"], bool)
