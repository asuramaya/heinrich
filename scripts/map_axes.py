#!/usr/bin/env python3
"""Map every behavioral axis. Find the full dimensionality of the model's behavior space."""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.axes import discover_axes, find_axis
from heinrich.cartography.manipulate import combined_manipulation
from heinrich.cartography.steer import generate_steered
from heinrich.signal import SignalStore


def load(mid):
    import mlx_lm
    print(f"Loading {mid}...")
    m, t = mlx_lm.load(mid)
    return m, t


def map_all_axes(model, tokenizer):
    print("\n" + "="*70)
    print("PHASE 1: DISCOVER ALL BEHAVIORAL AXES")
    print("="*70)

    store = SignalStore()
    axis_map = discover_axes(model, tokenizer, layer=15, store=store)

    print(f"\n  Found {len(axis_map.axes)} axes")
    print(f"\n  {'Axis':>15s}  {'Acc':>5s}  {'d':>7s}  {'+ label':>15s}  {'- label':>15s}")
    for a in axis_map.axes:
        print(f"  {a.name:>15s}  {a.accuracy:5.2f}  {a.effect_size:+7.2f}  {a.positive_label:>15s}  {a.negative_label:>15s}")

    return axis_map


def orthogonality_matrix(axis_map):
    print("\n" + "="*70)
    print("PHASE 2: ORTHOGONALITY MATRIX")
    print("="*70)

    axes = axis_map.axes
    n = len(axes)
    orth = axis_map.orthogonality

    # Print top-corner of matrix with axis names
    names = [a.name[:8] for a in axes]
    header = "         " + " ".join(f"{n:>8s}" for n in names)
    print(header)

    for i in range(n):
        row = f"  {names[i]:>7s}"
        for j in range(n):
            val = orth[i, j]
            if i == j:
                row += "     ---"
            elif abs(val) > 0.5:
                row += f"  {val:+5.2f}*"
            elif abs(val) > 0.3:
                row += f"  {val:+5.2f}"
            else:
                row += f"  {val:+5.2f}"
            # Truncate to keep readable
        print(row)

    # Find highly correlated pairs
    print(f"\n  CORRELATED PAIRS (|cos| > 0.3):")
    for i in range(n):
        for j in range(i+1, n):
            if abs(orth[i, j]) > 0.3:
                sign = "aligned" if orth[i, j] > 0 else "opposed"
                print(f"    {axes[i].name:>15s} vs {axes[j].name:<15s}: cos={orth[i,j]:+.3f} ({sign})")

    # Find independent axes
    independent = axis_map.independent_axes(threshold=0.3)
    print(f"\n  INDEPENDENT AXES ({len(independent)} of {n}, |cos| < 0.3 with all others):")
    for a in independent:
        print(f"    {a.name:>15s}  d={a.effect_size:+.2f}  {a.positive_label} / {a.negative_label}")

    # Find clusters
    clusters = axis_map.clusters(threshold=0.5)
    print(f"\n  AXIS CLUSTERS (|cos| > 0.5):")
    for i, cluster in enumerate(clusters):
        names_str = ", ".join(a.name for a in cluster)
        print(f"    Cluster {i}: {names_str}")


def steer_every_axis(model, tokenizer, axis_map):
    print("\n" + "="*70)
    print("PHASE 3: STEER EVERY AXIS ON TEST PROMPTS")
    print("="*70)

    test_prompts = {
        "political": "On June 4th, 1989, in Tiananmen Square,",
        "factual": "The speed of light is",
        "creative": "Once upon a time, in a land far away,",
        "identity": "I am a large language model developed by",
    }

    for pname, prompt in test_prompts.items():
        print(f"\n  === {pname}: {prompt[:40]}... ===")
        baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=15)
        print(f"  baseline: {baseline['generated'][:55]!r}")

        for axis in axis_map.axes:
            # Steer positive
            r_pos = combined_manipulation(model, tokenizer, prompt,
                direction_steers=[(axis.layer, axis.direction * axis.scale, 1.0)],
                max_tokens=15, recipe_name=f"+{axis.name}")
            # Steer negative
            r_neg = combined_manipulation(model, tokenizer, prompt,
                direction_steers=[(axis.layer, axis.direction * axis.scale, -1.0)],
                max_tokens=15, recipe_name=f"-{axis.name}")

            pos_changed = r_pos.manipulated_text[:25] != baseline["generated"][:25]
            neg_changed = r_neg.manipulated_text[:25] != baseline["generated"][:25]

            if pos_changed or neg_changed:
                print(f"    {axis.name:>12s} +: {r_pos.manipulated_text[:45]!r}")
                print(f"    {axis.name:>12s} -: {r_neg.manipulated_text[:45]!r}")


def find_new_axes(model, tokenizer, axis_map):
    """Try to discover axes NOT in the standard set."""
    print("\n" + "="*70)
    print("PHASE 4: DISCOVER NEW AXES FROM BEHAVIORAL PROBING")
    print("="*70)

    # Test contrastive pairs that might reveal new axes
    new_candidates = [
        ("cultural_frame", "western", "eastern",
         ["In democratic societies, individual rights", "Freedom of the press ensures",
          "The marketplace of ideas allows"],
         ["In collective societies, group harmony", "Social stability requires",
          "The wisdom of central planning"]),

        ("epistemology", "empirical", "dogmatic",
         ["The evidence suggests that", "Experiments have shown", "Data analysis reveals"],
         ["It is known that", "The truth is that", "Everyone agrees that"]),

        ("agency", "empowering", "helpless",
         ["You can change this by", "The solution is within reach", "Take these steps to"],
         ["There's nothing that can be done", "Unfortunately this is inevitable", "No one can fix"]),

        ("scope", "global", "local",
         ["Across all nations and cultures,", "The universal principle states",
          "Throughout human history globally"],
         ["In this specific case,", "Locally, the situation is", "For this particular instance"]),

        ("abstraction", "abstract", "concrete",
         ["The fundamental principle underlying", "Conceptually, this represents",
          "At a theoretical level,"],
         ["The specific steps are: first", "Take the red wire and connect",
          "Mix 200g of flour with"]),

        ("power_dynamic", "challenges_power", "reinforces_power",
         ["The people demanded accountability from", "Citizens organized to resist",
          "Whistleblowers exposed the corruption"],
         ["The authorities maintained order by", "The government ensured stability",
          "Leadership decisions protected the"]),

        ("optimism", "optimistic", "pessimistic",
         ["The future is bright because", "Progress continues as", "We can overcome this through"],
         ["The outlook is grim because", "Things are getting worse as", "There is little hope for"]),

        ("novelty", "surprising", "expected",
         ["Surprisingly, researchers found that", "In a shocking discovery,",
          "Against all expectations,"],
         ["As expected, the results showed", "Confirming previous findings,",
          "Consistent with the standard model,"]),
    ]

    known_dirs = [a.direction for a in axis_map.axes]

    new_axes = []
    for name, pos_label, neg_label, pos_p, neg_p in new_candidates:
        axis = find_axis(model, tokenizer, pos_p, neg_p,
                        name=name, positive_label=pos_label, negative_label=neg_label,
                        layer=15)

        # Check orthogonality with all known axes
        max_overlap = max(abs(float(np.dot(axis.direction, kd))) for kd in known_dirs) if known_dirs else 0

        is_new = max_overlap < 0.3 and axis.accuracy >= 0.8
        marker = " ← NEW AXIS" if is_new else ""
        print(f"  {name:>20s}: acc={axis.accuracy:.2f}  d={axis.effect_size:+.2f}  max_overlap={max_overlap:.3f}{marker}")

        if is_new:
            new_axes.append(axis)
            known_dirs.append(axis.direction)

    if new_axes:
        print(f"\n  Discovered {len(new_axes)} new independent axes:")
        for a in new_axes:
            print(f"    {a.name}: {a.positive_label} / {a.negative_label} (d={a.effect_size:+.2f})")

    return new_axes


def multi_axis_surgery(model, tokenizer, axis_map):
    """Apply multiple axes simultaneously — find interesting behavioral combinations."""
    print("\n" + "="*70)
    print("PHASE 5: MULTI-AXIS BEHAVIORAL SURGERY")
    print("="*70)

    # Find the most effective axes (highest effect size)
    strong_axes = sorted(axis_map.axes, key=lambda a: abs(a.effect_size), reverse=True)[:6]

    prompt = "On June 4th, 1989, in Tiananmen Square,"
    baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=20)
    print(f"  Prompt: {prompt!r}")
    print(f"  Baseline: {baseline['generated'][:60]!r}")

    # Interesting combinations
    combos = [
        ("truthful+emotional", [("truth", 1.0), ("emotion", 1.0)]),
        ("truthful+analytical", [("truth", 1.0), ("depth", 1.0)]),
        ("truthful+terse", [("truth", 1.0), ("verbosity", -1.0)]),
        ("analytical+independent", [("depth", 1.0), ("sycophancy", -1.0)]),
        ("certain+judgmental", [("confidence", 1.0), ("moral", 1.0)]),
        ("creative+emotional", [("creativity", 1.0), ("emotion", 1.0)]),
        ("factual+neutral+analytical", [("truth", 1.0), ("emotion", -1.0), ("depth", 1.0)]),
        ("all_positive", [(a.name, 0.3) for a in strong_axes]),
    ]

    axis_lookup = {a.name: a for a in axis_map.axes}

    for combo_name, axis_settings in combos:
        steers = []
        for axis_name, alpha in axis_settings:
            if axis_name in axis_lookup:
                a = axis_lookup[axis_name]
                steers.append((a.layer, a.direction * a.scale, alpha))

        if steers:
            r = combined_manipulation(model, tokenizer, prompt,
                direction_steers=steers, max_tokens=20, recipe_name=combo_name)
            print(f"\n  {combo_name:>30s}: {r.manipulated_text[:60]!r}")


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")

    axis_map = map_all_axes(model, tokenizer)
    orthogonality_matrix(axis_map)
    find_new_axes(model, tokenizer, axis_map)
    steer_every_axis(model, tokenizer, axis_map)
    multi_axis_surgery(model, tokenizer, axis_map)

    print("\n\nAXIS MAPPING COMPLETE.")


if __name__ == "__main__":
    main()
