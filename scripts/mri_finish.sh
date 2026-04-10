#!/bin/bash
# DEFINITIVE MRI queue. Everything missing. Nothing redundant.
set -e
cd /Users/asuramaya/Code/heinrich

echo "=== MRI Finish — $(date) ==="

# --- BACKFILLS (weights only, no recapture) ---

echo ""
echo "--- Gemma 2B weight backfill ---"
for mode in raw naked; do
    n=$(ls -d /Volumes/sharts/gemma2b_${mode}.mri/weights/L* 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "26" ] && echo "  gemma2b_${mode}: already complete" && continue
    echo "  gemma2b_${mode}: $n/26, backfilling..."
    python3 -m heinrich.cli mri-backfill \
        --model /tmp/gemma-2-2b-it-mlx \
        --mri /Volumes/sharts/gemma2b_${mode}.mri
done

# --- MISSING MODES (models with partial mode coverage) ---

echo ""
echo "--- Qwen 7B Instruct: naked ---"
[ -d /Volumes/sharts/qwen7b_naked.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --mode naked \
    --output /Volumes/sharts/qwen7b_naked.mri

echo ""
echo "--- Qwen 7B Instruct: template ---"
[ -d /Volumes/sharts/qwen7b_template.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --mode template \
    --output /Volumes/sharts/qwen7b_template.mri

echo ""
echo "--- Gemma 2B: template ---"
[ -d /Volumes/sharts/gemma2b_template.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/gemma-2-2b-it-mlx \
    --mode template \
    --output /Volumes/sharts/gemma2b_template.mri

echo ""
echo "--- SmolLM2 1.7B: template ---"
[ -d /Volumes/sharts/smollm17b_template.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/SmolLM2-1.7B-Instruct-mlx \
    --mode template \
    --output /Volumes/sharts/smollm17b_template.mri

# --- NEW MODELS (no MRI at all yet) ---

echo ""
echo "--- SmolLM2 135M: raw ---"
[ -d /Volumes/sharts/smollm_135m_raw.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/SmolLM2-135M-Instruct-mlx \
    --mode raw \
    --output /Volumes/sharts/smollm_135m_raw.mri

echo ""
echo "--- SmolLM2 135M: template ---"
[ -d /Volumes/sharts/smollm_135m_template.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/SmolLM2-135M-Instruct-mlx \
    --mode template \
    --output /Volumes/sharts/smollm_135m_template.mri

echo ""
echo "--- SmolLM2 360M: raw ---"
[ -d /Volumes/sharts/smollm_360m_raw.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/SmolLM2-360M-Instruct-mlx \
    --mode raw \
    --output /Volumes/sharts/smollm_360m_raw.mri

echo ""
echo "--- SmolLM2 360M: template ---"
[ -d /Volumes/sharts/smollm_360m_template.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/SmolLM2-360M-Instruct-mlx \
    --mode template \
    --output /Volumes/sharts/smollm_360m_template.mri

echo ""
echo "--- Qwen 0.5B Base: raw ---"
[ -d /Volumes/sharts/qwen05b_base_raw.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/qwen-qwen2.5-0.5b-mlx \
    --mode raw \
    --output /Volumes/sharts/qwen05b_base_raw.mri

echo ""
echo "--- Qwen 1.5B Instruct: raw ---"
[ -d /Volumes/sharts/qwen15b_raw.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/qwen-qwen2.5-1.5b-instruct-mlx \
    --mode raw \
    --output /Volumes/sharts/qwen15b_raw.mri

echo ""
echo "--- Qwen 1.5B Instruct: template ---"
[ -d /Volumes/sharts/qwen15b_template.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/qwen-qwen2.5-1.5b-instruct-mlx \
    --mode template \
    --output /Volumes/sharts/qwen15b_template.mri

echo ""
echo "--- Qwen 3B Base: raw ---"
[ -d /Volumes/sharts/qwen3b_base_raw.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model mlx-community/Qwen2.5-3B-4bit \
    --mode raw \
    --output /Volumes/sharts/qwen3b_base_raw.mri

echo ""
echo "--- Mistral 7B Base: raw ---"
[ -d /Volumes/sharts/mistral_base_raw.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/mistralai-mistral-7b-v0.1-mlx \
    --mode raw \
    --output /Volumes/sharts/mistral_base_raw.mri

echo ""
echo "--- Phi-2: raw ---"
[ -d /Volumes/sharts/phi2_raw.mri ] && echo "  exists, skipping" || \
python3 -m heinrich.cli mri \
    --model /tmp/microsoft-phi-2-mlx \
    --mode raw \
    --output /Volumes/sharts/phi2_raw.mri

echo ""
echo "=== DONE — $(date) ==="
