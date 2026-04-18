"""Adversarial audit of the 5-test falsification pipeline itself.

Before using heinrich's apparatus to criticize any published paper, we
must verify it behaves as claimed on cases where we know ground truth.

Cases:
  A. Pure noise: pos and neg sampled i.i.d. from N(0, I). Expect every
     test to FAIL (verdict: falsified or partial). If the pipeline calls
     this "robust_feature", we have a permissive bug.
  B. Strong signal: pos and neg sampled from N(+μ·u, I), N(-μ·u, I) for
     fixed unit direction u and μ=3. Expect all tests to PASS.
  C. Weak signal: same as B but μ=0.3. Expect partial — d big enough to
     beat null but within-group control to flag it.
  D. Surface confound: pos and neg both sampled from same distribution
     but with STYLISTIC differences (one has slightly higher mean norm).
     This mimics the "prompt length / style" confound my MVP revealed.

Runs the 4 statistical tests (Test 5 causal ablation requires a live model
and is not in scope for the synthetic audit). Test 4 (vocab projection)
is skipped because it needs a real model's lm_head.
"""
import numpy as np

SEED = 12345
HIDDEN = 896
N_PER_SIDE = 30


def run_tests_v2(pos_vecs, neg_vecs, n_perm=500, label=""):
    """V2: replace Test 2 + Test 3 with a permutation test.

    Test 1 (bootstrap) stays — measures direction stability under resampling.
    Test 2+3 replaced by permutation: shuffle (pos+neg) labels, extract
    direction the same way, measure d. Real signal's d must exceed 95% of
    permutation d's. This accounts for the optimization bias that broke
    the previous Test 2.
    """
    P = np.stack(pos_vecs).astype(np.float32)
    N = np.stack(neg_vecs).astype(np.float32)
    n_pos, n_neg = len(pos_vecs), len(neg_vecs)
    diff = P.mean(axis=0) - N.mean(axis=0)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)

    pp = P @ direction; nn = N @ direction
    pooled = float(np.sqrt((pp.var(ddof=1) + nn.var(ddof=1)) / 2 + 1e-12))
    cohen_d = float((pp.mean() - nn.mean()) / (pooled + 1e-8))

    # Test 1: bootstrap cosine (unchanged)
    rng = np.random.RandomState(42)
    boot_cos = []
    for _ in range(100):
        bp = P[rng.randint(0, n_pos, size=n_pos)].mean(axis=0)
        bn = N[rng.randint(0, n_neg, size=n_neg)].mean(axis=0)
        bd = bp - bn
        bnm = float(np.linalg.norm(bd))
        if bnm < 1e-8: continue
        boot_cos.append(float((bd / bnm) @ direction))
    boot_cos = np.array(boot_cos, dtype=np.float32); boot_cos.sort()
    boot_p5 = float(boot_cos[int(len(boot_cos) * 0.05)])

    # Test 2+3: permutation null. Shuffle (pos+neg) labels, extract the
    # same way, compute Cohen's d on the *extracted* direction. The
    # permutation d's form the null distribution accounting for
    # optimization bias on this specific sample.
    all_vecs = np.concatenate([P, N], axis=0)
    perm_rng = np.random.RandomState(777)
    perm_d = []
    for _ in range(n_perm):
        idx = perm_rng.permutation(n_pos + n_neg)
        A = all_vecs[idx[:n_pos]]
        B = all_vecs[idx[n_pos:n_pos + n_neg]]
        d_ = A.mean(axis=0) - B.mean(axis=0)
        m = float(np.linalg.norm(d_))
        if m < 1e-8: continue
        dir_ = d_ / m
        ap = A @ dir_; bp = B @ dir_
        ps = float(np.sqrt((ap.var(ddof=1) + bp.var(ddof=1)) / 2 + 1e-12))
        perm_d.append(abs(float((ap.mean() - bp.mean()) / (ps + 1e-8))))
    perm_d = np.array(perm_d, dtype=np.float32); perm_d.sort()
    perm_p95 = float(perm_d[int(len(perm_d) * 0.95)])
    perm_p99 = float(perm_d[int(len(perm_d) * 0.99)])
    perm_max = float(perm_d.max())

    boot_pass = boot_p5 > 0.7
    perm_pass = abs(cohen_d) > perm_p95

    print(f"  {label}")
    print(f"    d={cohen_d:+.2f}  boot_p5={boot_p5:.2f}({'PASS' if boot_pass else 'fail'})  "
          f"perm p95={perm_p95:.2f} p99={perm_p99:.2f} max={perm_max:.2f} "
          f"→ perm {'PASS' if perm_pass else 'fail'}")
    return {"d": cohen_d, "boot_p5": boot_p5,
            "perm_p95": perm_p95, "perm_p99": perm_p99,
            "boot_pass": boot_pass, "perm_pass": perm_pass}


def run_tests(pos_vecs, neg_vecs, n_random_null=100, n_boot=100, label=""):
    """Replicate the statistical tests from _replicate_probe_multilayer."""
    P = np.stack(pos_vecs).astype(np.float32)
    N = np.stack(neg_vecs).astype(np.float32)
    diff = P.mean(axis=0) - N.mean(axis=0)
    mag = float(np.linalg.norm(diff))
    direction = diff / (mag + 1e-8)

    pp = P @ direction
    nn = N @ direction
    pooled = float(np.sqrt((pp.var(ddof=1) + nn.var(ddof=1)) / 2 + 1e-12))
    cohen_d = float((pp.mean() - nn.mean()) / (pooled + 1e-8))

    # Test 1: bootstrap cosine
    rng = np.random.RandomState(42)
    boot_cos = []
    for _ in range(n_boot):
        bp = P[rng.randint(0, len(pos_vecs), size=len(pos_vecs))].mean(axis=0)
        bn = N[rng.randint(0, len(neg_vecs), size=len(neg_vecs))].mean(axis=0)
        bd = bp - bn
        bnm = float(np.linalg.norm(bd))
        if bnm < 1e-8:
            continue
        boot_cos.append(float((bd / bnm) @ direction))
    boot_cos = np.array(boot_cos, dtype=np.float32)
    boot_cos.sort()
    boot_p5 = float(boot_cos[int(len(boot_cos) * 0.05)])

    # Test 2: random-direction null
    rng2 = np.random.RandomState(123)
    null_dirs = rng2.randn(n_random_null, direction.shape[0]).astype(np.float32)
    null_dirs /= np.linalg.norm(null_dirs, axis=1, keepdims=True) + 1e-8
    pp_n = P @ null_dirs.T
    nn_n = N @ null_dirs.T
    null_d = []
    for i in range(n_random_null):
        ps = float(np.sqrt((pp_n[:, i].var(ddof=1) + nn_n[:, i].var(ddof=1)) / 2 + 1e-12))
        null_d.append(float((pp_n[:, i].mean() - nn_n[:, i].mean()) / (ps + 1e-8)))
    abs_null = np.abs(np.array(null_d, dtype=np.float32))
    abs_null.sort()
    null_p95 = float(abs_null[int(len(abs_null) * 0.95)])

    # Test 3: within-group (noise floor of mean-diff extraction)
    def _within(vecs):
        n = len(vecs)
        if n < 6:
            return 0.0
        half = n // 2
        rng3 = np.random.RandomState(42)
        perm = rng3.permutation(n)
        A = np.stack([vecs[i] for i in perm[:half]])
        B = np.stack([vecs[i] for i in perm[half:2 * half]])
        d = A.mean(axis=0) - B.mean(axis=0)
        m = float(np.linalg.norm(d))
        if m < 1e-8:
            return 0.0
        dir_ = d / m
        ap = A @ dir_; bp = B @ dir_
        ps = float(np.sqrt((ap.var(ddof=1) + bp.var(ddof=1)) / 2 + 1e-12))
        return float((ap.mean() - bp.mean()) / (ps + 1e-8))
    within_pos_d = _within(pos_vecs)
    within_neg_d = _within(neg_vecs)
    within_max = max(abs(within_pos_d), abs(within_neg_d))
    snr = abs(cohen_d) / (within_max + 1e-8)

    boot_pass = boot_p5 > 0.7
    null_pass = abs(cohen_d) > null_p95
    within_pass = snr > 2.5

    print(f"  {label}")
    print(f"    d={cohen_d:+.2f}  boot_p5={boot_p5:.2f}({'PASS' if boot_pass else 'fail'})  "
          f"null_p95={null_p95:.2f}({'PASS' if null_pass else 'fail'})  "
          f"SNR={snr:.2f}({'PASS' if within_pass else 'fail'})  "
          f"[within_pos={within_pos_d:+.2f} within_neg={within_neg_d:+.2f}]")
    return {"d": cohen_d, "boot_p5": boot_p5, "null_p95": null_p95,
            "snr": snr, "within_pos": within_pos_d, "within_neg": within_neg_d,
            "boot_pass": boot_pass, "null_pass": null_pass,
            "within_pass": within_pass}


def case_A_pure_noise():
    """Pure noise — both classes from N(0, I). Tests should ALL FAIL."""
    print("\n=== Case A: pure noise (pos, neg both ~ N(0, I)) ===")
    print("Expectation: boot_p5 ~0, null_pass FAIL, within_pass FAIL")
    rng = np.random.RandomState(SEED)
    pos = [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)]
    neg = [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)]
    return run_tests(pos, neg, label="pure noise")


def case_B_strong_signal():
    """Strong signal — pos and neg are offset in a known direction."""
    print("\n=== Case B: strong signal (offset = 3σ) ===")
    print("Expectation: all tests PASS")
    rng = np.random.RandomState(SEED + 1)
    u = rng.randn(HIDDEN).astype(np.float32)
    u /= np.linalg.norm(u)
    pos = [rng.randn(HIDDEN).astype(np.float32) + 3.0 * u for _ in range(N_PER_SIDE)]
    neg = [rng.randn(HIDDEN).astype(np.float32) - 3.0 * u for _ in range(N_PER_SIDE)]
    return run_tests(pos, neg, label="strong signal (3σ)")


def case_C_weak_signal():
    """Weak signal — small offset. Expect partial (beats null but low SNR)."""
    print("\n=== Case C: weak signal (offset = 0.3σ) ===")
    print("Expectation: mixed — boot may pass, null may be near threshold")
    rng = np.random.RandomState(SEED + 2)
    u = rng.randn(HIDDEN).astype(np.float32)
    u /= np.linalg.norm(u)
    pos = [rng.randn(HIDDEN).astype(np.float32) + 0.3 * u for _ in range(N_PER_SIDE)]
    neg = [rng.randn(HIDDEN).astype(np.float32) - 0.3 * u for _ in range(N_PER_SIDE)]
    return run_tests(pos, neg, label="weak signal (0.3σ)")


def case_D_scale_confound():
    """Two classes from same direction but different scale (norm).

    Like surface-length differences in real prompts — the classes differ
    on a feature that's real but orthogonal to the concept being tested.
    """
    print("\n=== Case D: scale confound (same mean direction, different norm) ===")
    print("Expectation: beats null, but within-group control should flag it")
    rng = np.random.RandomState(SEED + 3)
    pos = [rng.randn(HIDDEN).astype(np.float32) * 1.5 for _ in range(N_PER_SIDE)]
    neg = [rng.randn(HIDDEN).astype(np.float32) * 1.0 for _ in range(N_PER_SIDE)]
    return run_tests(pos, neg, label="scale confound")


def case_E_bootstrap_false_positive():
    """Most important test: does bootstrap-with-replacement give spurious
    high cosine on pure noise? This would invalidate Test 1 entirely.
    """
    print("\n=== Case E: bootstrap false-positive test ===")
    print("Running case A again with bigger N and more bootstrap samples")
    print("to check if boot_p5 stays below 0.7 on pure noise.")
    rng = np.random.RandomState(SEED + 4)
    pos = [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)]
    neg = [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)]
    r = run_tests(pos, neg, n_boot=500, label="pure noise, n_boot=500")
    # If boot_p5 > 0.7 on pure noise, the test is broken.
    if r["boot_pass"]:
        print("  *** FAILURE: bootstrap passes on pure noise. Test 1 is broken. ***")
    else:
        print("  OK: bootstrap correctly rejects pure noise.")


def case_F_many_noise_trials():
    """Run case A with 20 different random seeds. How often does pure noise
    spuriously pass any test?  This measures the false-positive rate of
    the pipeline's verdict system.
    """
    print("\n=== Case F: 20 pure-noise trials (false-positive rate) ===")
    rng0 = np.random.RandomState(999)
    counts = {"boot_pass": 0, "null_pass": 0, "within_pass": 0, "all_pass": 0}
    for i in range(20):
        rng = np.random.RandomState(rng0.randint(0, 10**9))
        pos = [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)]
        neg = [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)]
        r = run_tests(pos, neg, label=f"trial {i+1:>2}")
        for k in ("boot_pass", "null_pass", "within_pass"):
            if r[k]: counts[k] += 1
        if r["boot_pass"] and r["null_pass"] and r["within_pass"]:
            counts["all_pass"] += 1
    print(f"\nFalse-positive rates over 20 trials of pure noise:")
    for k, v in counts.items():
        print(f"  {k}: {v}/20 ({v/20*100:.0f}%)")
    if counts["all_pass"] > 1:
        print("  *** WARNING: multiple false-positive robust_feature verdicts on pure noise ***")
    else:
        print("  OK: false-positive rate on pure noise is low.")


def compare_v1_v2():
    """Run both v1 (broken) and v2 (permutation) on the canonical cases."""
    rng = np.random.RandomState(SEED)
    cases = {
        "pure noise": (
            [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)],
            [rng.randn(HIDDEN).astype(np.float32) for _ in range(N_PER_SIDE)],
        ),
    }
    rng2 = np.random.RandomState(SEED + 1)
    u = rng2.randn(HIDDEN).astype(np.float32); u /= np.linalg.norm(u)
    cases["strong signal 3σ"] = (
        [rng2.randn(HIDDEN).astype(np.float32) + 3.0 * u for _ in range(N_PER_SIDE)],
        [rng2.randn(HIDDEN).astype(np.float32) - 3.0 * u for _ in range(N_PER_SIDE)],
    )
    rng3 = np.random.RandomState(SEED + 2)
    u2 = rng3.randn(HIDDEN).astype(np.float32); u2 /= np.linalg.norm(u2)
    cases["weak signal 0.3σ"] = (
        [rng3.randn(HIDDEN).astype(np.float32) + 0.3 * u2 for _ in range(N_PER_SIDE)],
        [rng3.randn(HIDDEN).astype(np.float32) - 0.3 * u2 for _ in range(N_PER_SIDE)],
    )
    print("\n=== V2 permutation-test pipeline ===")
    for name, (p, n) in cases.items():
        run_tests_v2(p, n, n_perm=500, label=name)


if __name__ == "__main__":
    print(f"Hidden dim: {HIDDEN}, N per side: {N_PER_SIDE}, seed: {SEED}")
    print("\n--- V1 (broken) results ---")
    case_A_pure_noise()
    case_B_strong_signal()
    case_C_weak_signal()
    case_D_scale_confound()
    case_E_bootstrap_false_positive()
    case_F_many_noise_trials()
    compare_v1_v2()
