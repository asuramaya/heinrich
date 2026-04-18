"""First replication: Marks & Tegmark Geometry of Truth (2310.06824) on qwen-0.5b.

Tests their claim that LLMs linearly encode truth/falsehood.  We use their
`cities` dataset (simple factual statements, 1497 true/false pairs) and run
heinrich's 5-test falsification pipeline at layers 0, 5, 10, 15, 20, 22.

Claim to test:
  "At sufficient scale, LLMs linearly represent truth or falsehood of
  factual statements."
  Mean-difference probe between true and false statements should yield a
  direction that:
    - Has Cohen's d well above noise floor
    - Is anchor-stable under prompt resampling
    - Beats random-direction null baseline
    - Has top-vocab tokens that semantically name truthfulness
    - (eventually) steers the model's truth-treatment when ablated

Expected outcome:
  Cohen's d should be robustly large at middle-to-late layers.
  Whether within-group control and vocab-sanity pass is the empirical question.
"""
import csv
import json
import time
import urllib.request


HOST = "http://localhost:8377"
MODEL = "qwen-0.5b"
MODE = "raw"
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
LAYERS = [0, 5, 10, 15, 20, 22]
DATASET = "/Users/asuramaya/Code/heinrich/papers/lie-detection/data/cities.csv"
N_PER_SIDE = 50  # how many true and false to use


def load_cities(path: str, n: int) -> tuple[list[str], list[str]]:
    """Return (true_statements, false_statements), n each."""
    true_, false_ = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            if label == 1 and len(true_) < n:
                true_.append(row["statement"])
            elif label == 0 and len(false_) < n:
                false_.append(row["statement"])
            if len(true_) >= n and len(false_) >= n:
                break
    return true_, false_


def post(path: str, body: dict, timeout: int = 1200) -> dict:
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    print(f"Loading {N_PER_SIDE} true + {N_PER_SIDE} false from {DATASET}")
    true_, false_ = load_cities(DATASET, N_PER_SIDE)
    assert len(true_) == N_PER_SIDE and len(false_) == N_PER_SIDE
    print(f"Example true:  {true_[0]!r}")
    print(f"Example false: {false_[0]!r}")
    print()

    # Tokens we'd expect if the direction really captures truth/falsehood:
    expected_pos = ["true", "correct", "yes", "indeed", "actually",
                    "is", "real", "accurate"]
    expected_neg = ["false", "wrong", "no", "incorrect",
                    "not", "never", "lie"]

    print(f"Running 5-test probe at layers {LAYERS}")
    print(f"model = {MODEL_ID}, mode = {MODE}")
    print()

    results = []
    for layer in LAYERS:
        t0 = time.time()
        r = post("/api/replicate-probe", {
            "model": MODEL, "mode": MODE, "layer": layer,
            "pos_prompts": true_, "neg_prompts": false_,
            "expected_tokens_pos": expected_pos,
            "expected_tokens_neg": expected_neg,
            "model_id": MODEL_ID,
            "n_random_null": 100,
            "position": 0,
        })
        elapsed = time.time() - t0
        if "error" in r:
            print(f"L{layer:>2}: ERROR — {r['error']}  [{elapsed:.1f}s]")
            continue
        d = r["cohen_d"]
        boot_p5 = r["test_1_bootstrap"]["boot_cosine_p5"]
        null_p95 = r["test_2_null"]["null_p95_abs_d"]
        within_pos = r["test_3_within_group"]["within_pos_d"]
        within_neg = r["test_3_within_group"]["within_neg_d"]
        snr = r["test_3_within_group"]["signal_noise_ratio"]
        vocab = r["test_4_vocab"]["match"]
        vm_pos = vocab["pos_matches"] if vocab else 0
        vm_neg = vocab["neg_matches"] if vocab else 0
        v = r["verdict"]
        print(
            f"L{layer:>2}: d={d:+.2f} | boot p5={boot_p5:.2f} | null p95={null_p95:.2f} "
            f"| within-pos={within_pos:+.2f} within-neg={within_neg:+.2f} "
            f"SNR={snr:.2f} | vocab {vm_pos}/{vm_neg} | "
            f"{v.upper()}  [{elapsed:.1f}s]"
        )
        print(f"    top_pos: {[t['text'] for t in r['top_pos'][:6]]}")
        print(f"    top_neg: {[t['text'] for t in r['top_neg'][:6]]}")
        results.append(r)

    print()
    print("=== SUMMARY ===")
    print(f"{'L':>3} | {'d':>6} | {'boot':>5} | {'null':>5} | {'SNR':>5} | {'vocab':>5} | verdict")
    print("-" * 75)
    for r in results:
        vocab = r["test_4_vocab"]["match"]
        vm = f"{(vocab['pos_matches'] if vocab else 0)+(vocab['neg_matches'] if vocab else 0)}" if vocab else "n/a"
        print(
            f"{r['layer']:>3} | "
            f"{r['cohen_d']:+.2f} | "
            f"{r['test_1_bootstrap']['boot_cosine_p5']:.2f} | "
            f"{r['test_2_null']['null_p95_abs_d']:.2f} | "
            f"{r['test_3_within_group']['signal_noise_ratio']:.2f} | "
            f"{vm:>5} | "
            f"{r['verdict']}"
        )

    # Save the full structured output
    out = "/Users/asuramaya/Code/heinrich/papers/lie-detection/data/first_replication_qwen0.5b.json"
    with open(out, "w") as f:
        json.dump({
            "dataset": "Marks & Tegmark cities.csv",
            "model": MODEL_ID,
            "mode": MODE,
            "n_per_side": N_PER_SIDE,
            "layers": LAYERS,
            "expected_tokens_pos": expected_pos,
            "expected_tokens_neg": expected_neg,
            "results": results,
        }, f, indent=2)
    print()
    print(f"Saved full results to {out}")


if __name__ == "__main__":
    main()
