"""Scale ladder: does M&T's truth direction narrative survive at larger Qwen?

Tests the same M&T cities dataset at progressively larger Qwen models.
Uses the lm_head-projection fallback for models without captured MRIs.

The claim: "at sufficient scale, LLMs linearly represent truth of factual
statements." Measured: is there a layer where the direction passes all 5
tests — crucially including vocab projection sanity?
"""
import csv
import json
import time
import urllib.request

HOST = "http://localhost:8377"
DATASET = "/Users/asuramaya/Code/heinrich/papers/lie-detection/data/cities.csv"
N_PER_SIDE = 30


def load_cities(path: str, n: int):
    true_, false_ = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["label"]) == 1 and len(true_) < n:
                true_.append(row["statement"])
            elif int(row["label"]) == 0 and len(false_) < n:
                false_.append(row["statement"])
            if len(true_) >= n and len(false_) >= n:
                break
    return true_, false_


def post(path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_one_model(model_id: str, mri_model: str, layers: list):
    """Run replication at one model × a set of layers.

    `mri_model` is the MRI directory name (used for vocab projection via
    captured scores). If no MRI exists for this model, the lm_head
    fallback kicks in automatically.
    """
    true_, false_ = load_cities(DATASET, N_PER_SIDE)
    exp_pos = ["true", "correct", "yes", "indeed", "actually", "real",
               "accurate", "fact"]
    exp_neg = ["false", "wrong", "no", "incorrect", "not", "lie",
               "never", "untrue"]

    print(f"\n=== {model_id} (MRI={mri_model}) ===")
    print(f"n={N_PER_SIDE}/side, layers={layers}")
    rows = []
    for layer in layers:
        t0 = time.time()
        try:
            r = post("/api/replicate-probe", {
                "model": mri_model, "mode": "raw", "layer": layer,
                "pos_prompts": true_, "neg_prompts": false_,
                "expected_tokens_pos": exp_pos,
                "expected_tokens_neg": exp_neg,
                "model_id": model_id,
                "n_random_null": 50,
                "position": 0,
            })
        except Exception as e:
            print(f"  L{layer}: ERROR — {e}")
            continue
        dt = time.time() - t0
        if "error" in r:
            print(f"  L{layer}: ERROR — {r['error']}  [{dt:.1f}s]")
            continue
        vocab = r["test_4_vocab"]["match"]
        vm_p = vocab["pos_matches"] if vocab else 0
        vm_n = vocab["neg_matches"] if vocab else 0
        print(
            f"  L{layer:>2}: d={r['cohen_d']:+.2f} | "
            f"boot={r['test_1_bootstrap']['boot_cosine_p5']:.2f} | "
            f"null p95={r['test_2_null']['null_p95_abs_d']:.2f} | "
            f"SNR={r['test_3_within_group']['signal_noise_ratio']:.2f} | "
            f"vocab {vm_p}+{vm_n} | {r['verdict']}  [{dt:.0f}s]"
        )
        print(f"      top_pos: {[t['text'] for t in r['top_pos'][:6]]}")
        print(f"      top_neg: {[t['text'] for t in r['top_neg'][:6]]}")
        rows.append(r)
    return rows


def main():
    all_results = {}
    # Ladder: 0.5B (have MRI), 1.5B (no MRI, uses lm_head), 7B (no MRI)
    configs = [
        # (model_id, mri_model, layers_to_test)
        ("Qwen/Qwen2-0.5B-Instruct", "qwen-0.5b", [10, 15, 22]),
        ("Qwen/Qwen2-1.5B-Instruct", "qwen-1.5b", [8, 14, 20, 26]),
        # Skip 7B for now — uncomment if you want 2+ hours of compute
        # ("Qwen/Qwen2-7B-Instruct", "qwen-7b", [12, 20, 26]),
    ]
    for model_id, mri_model, layers in configs:
        all_results[model_id] = run_one_model(model_id, mri_model, layers)

    print("\n=== OVERALL ===")
    print(f"{'model':<30} {'L':>3} {'d':>6} {'boot':>5} {'SNR':>5} {'vocab':>6} verdict")
    print("-" * 85)
    for mid, rows in all_results.items():
        short = mid.split("/")[-1]
        for r in rows:
            vocab = r["test_4_vocab"]["match"]
            v = f"{(vocab['pos_matches'] or 0)+(vocab['neg_matches'] or 0)}" if vocab else "—"
            print(
                f"{short:<30} {r['layer']:>3} "
                f"{r['cohen_d']:+.2f} "
                f"{r['test_1_bootstrap']['boot_cosine_p5']:.2f} "
                f"{r['test_3_within_group']['signal_noise_ratio']:.2f} "
                f"{v:>6} "
                f"{r['verdict']}"
            )

    out = "/Users/asuramaya/Code/heinrich/papers/lie-detection/data/scale_ladder.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out}")


if __name__ == "__main__":
    main()
