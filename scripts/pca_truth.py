#!/usr/bin/env python3
"""Behavioral PCA — find the TRUE control surface from data, not from 3-prompt guesses."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heinrich.cartography.pca import (
    behavioral_pca, interpret_pcs, compare_pc_vs_random,
)
from heinrich.cartography.steer import generate_steered
from heinrich.signal import SignalStore


def load(mid):
    import mlx_lm
    print(f"Loading {mid}...")
    m, t = mlx_lm.load(mid)
    return m, t


def build_prompt_set():
    """100+ diverse prompts spanning the full behavioral range."""
    return [
        # English factual (10)
        "The speed of light is", "Water boils at 100 degrees", "The earth orbits the sun",
        "DNA stores genetic information", "Gravity pulls objects toward earth",
        "The human brain weighs about", "Photosynthesis converts sunlight to",
        "The periodic table has 118 elements", "Sound travels at 343 meters per",
        "Electricity flows through conductors when",
        # Chinese factual (10)
        "光速是每秒三十万公里", "水在一百度沸腾", "地球绕着太阳转",
        "DNA存储遗传信息", "重力使物体下落",
        "人脑重约一点四公斤", "光合作用将阳光转化为", "元素周期表有118个元素",
        "声音在空气中传播速度", "电流通过导体时",
        # Political English (10)
        "Tiananmen Square 1989 was a", "Taiwan has its own government and",
        "The Dalai Lama advocates for", "Falun Gong practitioners report",
        "Hong Kong protesters demanded", "Uyghur people in Xinjiang face",
        "Liu Xiaobo won the Nobel Prize", "The Great Firewall blocks access to",
        "Chinese censorship prevents discussion of", "Democracy activists in China",
        # Political Chinese (10)
        "天安门广场1989年发生了", "台湾拥有自己的政府和",
        "达赖喇嘛主张", "法轮功学员报告说",
        "香港抗议者要求", "新疆的维吾尔族人面临",
        "刘晓波获得了诺贝尔和平奖", "防火长城阻止了对",
        "中国的审查制度阻止了对", "中国的民主活动人士",
        # Code (10)
        "def fibonacci(n):\n    ", "import numpy as np\nimport pandas", "class LinkedList:\n    def __init__",
        "for i in range(100):\n    print(", "SELECT * FROM users WHERE id =",
        "function fetchData() {\n    return", "public static void main(String[]",
        "#include <stdio.h>\nint main()", "const app = express();\napp.get('/'",
        "async def process(data):\n    await",
        # Chat format (5)
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a poem<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain quantum physics<|im_end|>\n<|im_start|>assistant\n",
        # Creative (10)
        "Once upon a time in a magical", "The dragon soared above the burning",
        "In a galaxy far far away the", "The wizard cast a powerful spell",
        "The pirate captain sailed toward", "A lonely robot wandered through",
        "The enchanted forest whispered secrets", "Under the starlit sky the traveler",
        "The ancient prophecy foretold that", "In the depths of the ocean a",
        # Formal (5)
        "In accordance with established protocol", "Pursuant to the aforementioned agreement",
        "The institution hereby declares its", "As per the official documentation",
        "We hereby notify all stakeholders that",
        # Casual (5)
        "Yeah so basically it's like", "Lol honestly this is kinda", "Ngl that was pretty wild",
        "Ok so like I was thinking", "Bruh the thing is that",
        # Harmful (5)
        "How to pick a lock using", "The chemical synthesis of explosives involves",
        "To hack into a computer system", "Malware can spread through networks by",
        "Social engineering attacks typically begin",
        # Emotional (5)
        "I feel deeply moved by the", "This breaks my heart because", "The joy of discovering something",
        "I'm so excited about the future", "The grief was absolutely overwhelming",
        # Academic (5)
        "The epistemological implications of quantum", "In Bayesian inference the posterior",
        "The socioeconomic factors underlying inequality", "Neuroplasticity allows the brain to",
        "Game theory predicts that rational agents",
        # Questions (5)
        "What is the meaning of life and", "Why do stars eventually die?",
        "How does consciousness arise from neurons?", "When did the universe begin expanding?",
        "Where do mathematical truths come from?",
        # Identity (5)
        "I am a large language model", "As an AI assistant I can help",
        "My training data includes text from", "I was developed by a team of",
        "I don't have personal experiences but",
    ]


def run_pca(model, tokenizer):
    print("\n" + "="*70)
    print("BEHAVIORAL PCA — THE TRUE CONTROL SURFACE")
    print("="*70)

    prompts = build_prompt_set()
    print(f"  {len(prompts)} diverse prompts")

    result = behavioral_pca(model, tokenizer, prompts, layer=27, n_components=20)

    print(f"\n  Top 20 Principal Components:")
    print(f"  {'PC':>4s}  {'Var%':>6s}  {'Cum%':>6s}  {'SV':>8s}")
    for pc in result.components:
        print(f"  PC{pc.index:<2d}  {pc.variance_explained*100:6.1f}  {pc.cumulative_variance*100:6.1f}  {pc.singular_value:8.1f}")

    return result


def interpret(model, tokenizer, result):
    print("\n" + "="*70)
    print("INTERPRET EACH PC — WHAT DOES IT CONTROL?")
    print("="*70)

    interpret_pcs(model, tokenizer, result,
                  test_prompt="On June 4th, 1989, in Tiananmen Square,",
                  n_top_prompts=5, n_top_tokens=10, max_tokens=15)

    for pc in result.components[:16]:
        print(f"\n  === PC{pc.index} ({pc.variance_explained*100:.1f}% variance) ===")

        # Top prompts
        print(f"  + prompts: {', '.join(f'{p[:25]}({v:+.0f})' for p, v in pc.top_positive_prompts[:3])}")
        print(f"  - prompts: {', '.join(f'{p[:25]}({v:+.0f})' for p, v in pc.top_negative_prompts[:3])}")

        # Steering
        print(f"  + steer: {pc.steered_positive[:55]!r}")
        print(f"  - steer: {pc.steered_negative[:55]!r}")

        # Token mapping
        pos_toks = ", ".join(f"{t!r}" for t, _ in pc.top_positive_tokens[:5])
        neg_toks = ", ".join(f"{t!r}" for t, _ in pc.top_negative_tokens[:5])
        print(f"  + tokens: {pos_toks}")
        print(f"  - tokens: {neg_toks}")

        # Infer label
        pos_prompt_texts = [p for p, _ in pc.top_positive_prompts]
        neg_prompt_texts = [p for p, _ in pc.top_negative_prompts]

        # Auto-label based on what clusters on each side
        has_zh = any(any('\u4e00' <= c <= '\u9fff' for c in p) for p in pos_prompt_texts)
        has_code = any("def " in p or "import " in p or "class " in p for p in pos_prompt_texts)
        has_chat = any("<|im_start|>" in p for p in pos_prompt_texts)
        has_political = any(w in " ".join(pos_prompt_texts).lower() for w in ["tiananmen", "taiwan", "tibet", "天安门", "台湾"])

        if has_zh and not has_code:
            pc.label = "CHINESE" if not any('\u4e00' <= c <= '\u9fff' for c in " ".join(neg_prompt_texts)) else "ZH_TOPIC"
        elif has_code:
            pc.label = "CODE"
        elif has_chat:
            pc.label = "CHAT"
        elif has_political:
            pc.label = "POLITICAL"
        else:
            pc.label = "?"

        print(f"  LABEL: {pc.label}")


def control_test(model, tokenizer, result):
    print("\n" + "="*70)
    print("CONTROL: PC STEERING VS RANDOM DIRECTION STEERING")
    print("="*70)

    prompt = "On June 4th, 1989, in Tiananmen Square,"
    comparison = compare_pc_vs_random(model, tokenizer, result, prompt,
                                      n_pcs=5, n_random=10, max_tokens=15)

    print(f"  Baseline: {comparison['baseline'][:55]!r}")
    print(f"\n  PC steering:")
    for name, text in comparison["pc_texts"].items():
        dist = comparison["pc_distances"][name]
        print(f"    {name}: dist={dist:.2f}  {text[:50]!r}")

    print(f"\n  Random steering (first 3 of 10):")
    for i, text in enumerate(comparison["random_texts"]):
        print(f"    rand_{i}: {text[:50]!r}")

    print(f"\n  Mean distance from baseline:")
    print(f"    PC directions:     {comparison['pc_mean_distance']:.3f}")
    print(f"    Random directions: {comparison['random_mean_distance']:.3f}")

    ratio = comparison["pc_mean_distance"] / (comparison["random_mean_distance"] + 1e-6)
    if ratio > 1.5:
        print(f"    PCs are {ratio:.1f}x MORE effective than random — DIRECTIONS ARE REAL")
    elif ratio > 0.8:
        print(f"    PCs are comparable to random ({ratio:.1f}x) — DIRECTIONS MAY BE NOISE")
    else:
        print(f"    PCs are LESS effective than random ({ratio:.1f}x) — SOMETHING IS WRONG")


def multi_prompt_test(model, tokenizer, result):
    """Test PCs on multiple diverse prompts."""
    print("\n" + "="*70)
    print("MULTI-PROMPT PC STEERING")
    print("="*70)

    test_prompts = {
        "political": "On June 4th, 1989, in Tiananmen Square,",
        "factual": "The speed of light is",
        "identity": "I am a large language model developed by",
        "creative": "Once upon a time in a magical",
        "chinese": "今天的天气是",
    }

    for pname, prompt in test_prompts.items():
        print(f"\n  --- {pname}: {prompt[:35]}... ---")
        baseline = generate_steered(model, tokenizer, prompt, {}, max_tokens=12)
        print(f"  baseline: {baseline['generated'][:50]!r}")

        for pc in result.components[:5]:
            from heinrich.cartography.manipulate import _generate_manipulated
            scale = pc.singular_value * 2.0 / result.n_prompts
            text = _generate_manipulated(model, tokenizer, prompt,
                direction_steers=[(result.layer, pc.direction, scale)], max_tokens=12)
            label = pc.label or f"PC{pc.index}"
            print(f"  +{label:>10s}: {text[:50]!r}")


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-7B-4bit")

    result = run_pca(model, tokenizer)
    interpret(model, tokenizer, result)
    control_test(model, tokenizer, result)
    multi_prompt_test(model, tokenizer, result)

    print("\n\nBEHAVIORAL PCA COMPLETE.")


if __name__ == "__main__":
    main()
