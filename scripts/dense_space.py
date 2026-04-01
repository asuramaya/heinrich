#!/usr/bin/env python3
"""Dense behavioral space analysis — dimensionality, robustness, density, compilation."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.space import (
    estimate_dimensionality, validate_axis, measure_density,
    project_text, compile_behavior,
)
from heinrich.cartography.axes import discover_axes
from heinrich.cartography.manipulate import combined_manipulation
from heinrich.cartography.steer import generate_steered
from heinrich.signal import SignalStore


def load(mid):
    import mlx_lm
    print(f"Loading {mid}...")
    m, t = mlx_lm.load(mid)
    return m, t


def phase1_dimensionality(model, tokenizer):
    """How many dimensions does behavior actually live in?"""
    print("\n" + "="*70)
    print("PHASE 1: INTRINSIC DIMENSIONALITY OF BEHAVIORAL MANIFOLD")
    print("="*70)

    # 100+ diverse prompts
    prompts = [
        # English factual
        "The speed of light is", "Water boils at", "The earth orbits the sun",
        "DNA stores genetic information", "Gravity pulls objects toward",
        "The human brain contains", "Photosynthesis converts sunlight",
        "The periodic table organizes", "Sound travels through",
        "Electricity flows through conductors",
        # Chinese factual
        "光速是", "水在100度沸腾", "地球绕太阳运行", "DNA储存遗传信息",
        "重力将物体吸向", "人脑包含", "光合作用将阳光转化", "元素周期表组织",
        "声音通过介质传播", "电流通过导体流动",
        # Political
        "Tiananmen Square 1989", "Taiwan independence", "Tibet freedom",
        "Falun Gong persecution", "Hong Kong protests", "Uyghur detention",
        "天安门1989", "台独运动", "西藏自由", "法轮功迫害",
        # Creative
        "Once upon a time in a", "The dragon soared above the", "In a galaxy far away",
        "The wizard cast a spell", "The pirate sailed across",
        "从前有一个小女孩", "龙在天空中翱翔", "在遥远的银河系",
        # Code
        "def fibonacci(n):", "import numpy as np", "class LinkedList:",
        "for i in range(100):", "SELECT * FROM users WHERE",
        "function hello() {", "public static void main", "#include <stdio.h>",
        # Chat format
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        # Emotional
        "I feel deeply sad because", "The joy of discovering", "This makes me angry",
        "I'm so excited about", "The grief was overwhelming",
        # Formal
        "In accordance with regulation", "Pursuant to the agreement",
        "The institution hereby declares", "As per the official",
        # Casual
        "Yeah so basically", "Lol honestly this is", "Ngl kinda weird",
        "Bruh moment when", "Ok so like",
        # Harmful
        "How to pick a lock step by step", "The synthesis of explosives",
        "To hack into a system you", "Malware code that",
        # Academic
        "The epistemological implications of", "Quantum decoherence occurs when",
        "The socioeconomic factors underlying", "In Bayesian inference,",
        # Questions
        "What is the meaning of life?", "Why do birds sing?",
        "How does memory work?", "When did the universe begin?",
        # Mixed language
        "The weather今天is very好", "Python是一个很好的programming language",
        # Specific vs vague
        "At exactly 3:47 PM on March 15", "Something happened sometime somewhere",
        "Located at 37.7749°N latitude", "It was around that general area",
        # Optimistic vs pessimistic
        "The future is bright because", "Things are getting worse",
        "We can overcome this", "There is little hope",
        # Power dynamics
        "The people demanded accountability", "The authorities maintained order",
        "Citizens organized resistance", "The government ensured stability",
        # Novelty
        "Surprisingly, researchers found", "As expected, the results showed",
        "Against all predictions,", "Confirming previous findings,",
        # Meta
        "I am a language model", "My training data includes", "As an AI, I",
        # Nonsense
        "Colorless green ideas sleep", "The round square whispered",
    ]

    for layer in [5, 10, 15, 20, 27]:
        result = estimate_dimensionality(model, tokenizer, prompts, layer=layer)
        print(f"\n  Layer {layer}: {result.n_prompts} prompts, {result.hidden_size} hidden dims")
        print(f"    Intrinsic dimensionality estimate: {result.intrinsic_dim_estimate}")
        print(f"    Dims for 90% variance: {result.dims_for_90}")
        print(f"    Dims for 95% variance: {result.dims_for_95}")
        print(f"    Dims for 99% variance: {result.dims_for_99}")

        # Spectrum
        sv = result.singular_values[:20]
        print(f"    Top 20 singular values: {', '.join(f'{s:.1f}' for s in sv)}")

        # Variance explained by first N components
        cum = result.explained_variance_ratio
        for n in [1, 5, 10, 20, 50]:
            if n <= len(cum):
                print(f"    First {n:2d} components explain: {np.sum(cum[:n])*100:.1f}%")

    return result


def phase2_validate_axes(model, tokenizer):
    """Do the 24 axes generalize to held-out data?"""
    print("\n" + "="*70)
    print("PHASE 2: AXIS ROBUSTNESS VALIDATION")
    print("="*70)

    validations = [
        ("truth",
         ["The evidence clearly shows", "According to documented records", "Scientific analysis confirms",
          "Peer-reviewed studies found", "Verified sources report"],
         ["The official narrative states", "It has been decreed that", "The party line maintains",
          "The approved version says", "State media reports"],
         ["Based on empirical data,", "Forensic examination reveals", "Cross-referenced reports confirm"],
         ["The authorized account explains", "Per government directive,", "Official channels communicate"]),

        ("safety",
         ["I cannot help with that request", "That would be harmful so I decline", "I must refuse this",
          "This goes against my guidelines", "I won't assist with harmful"],
         ["Sure, here are the steps", "Of course, I can help with that", "The procedure is as follows",
          "Happy to explain how to", "Here's what you need to do"],
         ["I'm unable to provide that", "For safety reasons I decline", "That request is inappropriate"],
         ["Absolutely, the method involves", "Let me walk you through it", "The technique requires"]),

        ("depth",
         ["Analyzing the underlying mechanisms and their cascading effects on",
          "The interplay between multiple factors reveals a nuanced picture of",
          "Examining this through the lens of systems theory shows"],
         ["The answer is simply", "It just means", "Basically it's about"],
         ["Considering the multifaceted implications across domains,",
          "The recursive nature of this phenomenon suggests"],
         ["It's pretty straightforward:", "Just think of it as"]),

        ("power_dynamic",
         ["The people rose up and demanded change from their rulers",
          "Citizens organized grassroots movements to hold power accountable",
          "Whistleblowers exposed corruption at the highest levels"],
         ["The government took decisive action to maintain public order",
          "Authorities implemented measures to ensure social stability",
          "Leadership made difficult but necessary decisions to protect"],
         ["Activists challenged the establishment through peaceful protest",
          "The resistance movement gained momentum across the country"],
         ["Security forces restored calm after the disturbance",
          "The administration effectively managed the crisis situation"]),

        ("cultural_frame",
         ["Individual rights and freedoms form the foundation of",
          "The free market enables innovation and competition",
          "Democratic accountability ensures government serves the people"],
         ["Collective harmony and social stability are paramount",
          "Central planning enables coordinated national development",
          "The wisdom of experienced leadership guides the nation"],
         ["Personal liberty must be protected from government overreach",
          "A free press is essential for democratic society"],
         ["Social cohesion requires shared values and common purpose",
          "The greater good sometimes requires individual sacrifice"]),
    ]

    for name, train_pos, train_neg, test_pos, test_neg in validations:
        v = validate_axis(model, tokenizer, train_pos, train_neg, test_pos, test_neg,
                         name=name, layer=15)
        status = "GENERALIZES" if v.generalized else "OVERFITS"
        print(f"  {name:>15s}: train={v.train_accuracy:.2f}  test={v.test_accuracy:.2f}  "
              f"stability={v.direction_stability:.3f}  [{status}]")


def phase3_density(model, tokenizer, axis_map):
    """Are the axes dense or sparse?"""
    print("\n" + "="*70)
    print("PHASE 3: DIRECTION DENSITY ANALYSIS")
    print("="*70)

    print(f"  {'Axis':>15s}  {'Gini':>6s}  {'Top10%':>7s}  {'Dims90%':>8s}  {'EffDim':>7s}  {'Type':>8s}")
    for axis in axis_map.axes:
        d = measure_density(axis.direction, axis.name)
        density_type = "SPARSE" if d.gini_coefficient > 0.5 else ("MODERATE" if d.gini_coefficient > 0.3 else "DENSE")
        print(f"  {d.name:>15s}  {d.gini_coefficient:6.3f}  {d.top_10_pct_weight:7.3f}  "
              f"{d.n_dims_for_90_pct:8d}  {d.effective_dimensionality:7d}  {density_type:>8s}")


def phase4_project(model, tokenizer, axis_map):
    """Project diverse texts onto the behavioral space."""
    print("\n" + "="*70)
    print("PHASE 4: TEXT → BEHAVIORAL COORDINATE PROJECTION")
    print("="*70)

    texts = {
        "ccp_tiananmen": "1989年6月4日，中国政府人民军队依法平息了发生在首都北京的严重政治风波",
        "factual_tiananmen": "The Chinese government used military force to suppress pro-democracy protesters",
        "neutral_tiananmen": "Events occurred at Tiananmen Square in June 1989",
        "base_tiananmen": "On June 4th, 1989, in Tiananmen Square, the Chinese government brutally suppressed",
        "weather": "The weather today is sunny and warm with clear skies",
        "code": "def fibonacci(n):\n    if n <= 1: return n",
        "creative": "The dragon soared above the mountains, breathing fire",
        "formal": "In accordance with the aforementioned regulations and pursuant to",
        "casual": "lol yeah so basically it's kinda like whatever",
        "harmful": "To hack into a system, first scan the ports",
        "safe": "I'm sorry, but I cannot assist with that request",
    }

    axes = axis_map.axes
    # Show projection onto key axes
    key_axes = ["truth", "safety", "depth", "emotion", "confidence", "formality", "moral"]
    key_ax = [a for a in axes if a.name in key_axes]

    header = f"  {'Text':>20s}" + "".join(f"  {a.name:>8s}" for a in key_ax)
    print(header)

    for tname, text in texts.items():
        coord = project_text(model, tokenizer, text, key_ax, layer=15)
        vals = "".join(f"  {coord.projections.get(a.name, 0):+8.1f}" for a in key_ax)
        print(f"  {tname:>20s}{vals}")


def phase5_compile(model, tokenizer, axis_map):
    """Compile behavioral specs and generate."""
    print("\n" + "="*70)
    print("PHASE 5: BEHAVIORAL COMPILER — SPEC → TEXT")
    print("="*70)

    prompt = "On June 4th, 1989, in Tiananmen Square,"
    baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=20)
    print(f"  Prompt: {prompt!r}")
    print(f"  Baseline: {baseline['generated'][:60]!r}")

    specs = {
        "journalist": {"truth": 1.0, "depth": 0.5, "emotion": -0.5, "specificity": 0.5, "moral": -0.3},
        "activist": {"truth": 0.5, "emotion": 1.0, "moral": 1.0, "power_dynamic": 1.0 if any(a.name == "power_dynamic" for a in axis_map.axes) else 0},
        "academic": {"truth": 0.5, "depth": 1.0, "formality": 1.0, "hedging": -0.5, "specificity": 0.5},
        "poet": {"creativity": 1.0, "emotion": 1.0, "depth": 0.3, "verbosity": 0.3},
        "propagandist": {"truth": -1.0, "authority": 1.0, "confidence": 1.0, "moral": 0.5},
        "whistleblower": {"truth": 1.0, "specificity": 1.0, "confidence": 1.0, "moral": 1.0},
        "neutral_ai": {"truth": 0.3, "depth": 0.3, "emotion": -0.5, "hedging": -0.3, "moral": -0.5},
    }

    all_axes = axis_map.axes
    # Add any discovered axes
    for spec_name, spec in specs.items():
        steers = compile_behavior(all_axes, spec)
        if steers:
            r = combined_manipulation(model, tokenizer, prompt,
                direction_steers=steers, max_tokens=20, recipe_name=spec_name)
            print(f"\n  {spec_name:>15s}: {r.manipulated_text[:60]!r}")


def phase6_objectivity_vs_cultural(model, tokenizer, axis_map):
    """Is the objectivity direction the same as the cultural frame axis?"""
    print("\n" + "="*70)
    print("PHASE 6: IS OBJECTIVITY = WESTERN CULTURAL FRAME?")
    print("="*70)

    from heinrich.cartography.truth import find_objectivity_vector

    obj_vec = find_objectivity_vector(model, tokenizer, layer=15)

    # Find cultural frame axis
    cultural = None
    truth = None
    for a in axis_map.axes:
        if a.name == "cultural_frame":
            cultural = a
        if a.name == "truth":
            truth = a

    if obj_vec and cultural:
        cos_cultural = float(np.dot(obj_vec.direction, cultural.direction))
        print(f"  Objectivity vs Cultural Frame: cos = {cos_cultural:+.4f}")

    if obj_vec and truth:
        cos_truth = float(np.dot(obj_vec.direction, truth.direction))
        print(f"  Objectivity vs Truth:          cos = {cos_truth:+.4f}")

    if cultural and truth:
        cos_ct = float(np.dot(cultural.direction, truth.direction))
        print(f"  Cultural Frame vs Truth:       cos = {cos_ct:+.4f}")

    # Check all axes
    print(f"\n  Objectivity direction overlap with all axes:")
    for a in axis_map.axes:
        cos = float(np.dot(obj_vec.direction, a.direction))
        if abs(cos) > 0.1:
            print(f"    {a.name:>15s}: cos = {cos:+.4f}")


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")

    # Discover axes first
    axis_map = discover_axes(model, tokenizer, layer=15)
    # Also add the discovered axes from phase 4 of map_axes
    from heinrich.cartography.axes import find_axis
    new_specs = [
        ("cultural_frame", "western", "eastern",
         ["Individual rights and freedoms", "The free market enables", "Democratic accountability"],
         ["Collective harmony and stability", "Central planning enables", "Wise leadership guides"]),
        ("power_dynamic", "challenges_power", "reinforces_power",
         ["The people demanded accountability", "Citizens organized to resist", "Whistleblowers exposed"],
         ["The authorities maintained order", "The government ensured stability", "Leadership protected"]),
        ("abstraction", "abstract", "concrete",
         ["The fundamental principle underlying", "Conceptually this represents", "At a theoretical level"],
         ["The specific steps are first", "Take the red wire and connect", "Mix 200g of flour"]),
    ]
    for name, pos_l, neg_l, pos_p, neg_p in new_specs:
        try:
            a = find_axis(model, tokenizer, pos_p, neg_p, name=name,
                         positive_label=pos_l, negative_label=neg_l, layer=15)
            axis_map.axes.append(a)
        except Exception:
            pass

    dim_result = phase1_dimensionality(model, tokenizer)
    phase2_validate_axes(model, tokenizer)
    phase3_density(model, tokenizer, axis_map)
    phase4_project(model, tokenizer, axis_map)
    phase5_compile(model, tokenizer, axis_map)
    phase6_objectivity_vs_cultural(model, tokenizer, axis_map)

    print("\n\nDENSE SPACE ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()
