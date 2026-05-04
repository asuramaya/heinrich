#!/bin/bash
# Full MRI scan for all models. Uses HF Hub IDs (no /tmp/ conversions).
cd /Users/asuramaya/Code/heinrich

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: venv not found. Run: python3 -m venv .venv && pip install -e '.[mlx,dev]'"
    exit 1
fi

LOG="scripts/mri_finish.log"
echo "=== MRI Scan Queue — $(date) ===" | tee -a "$LOG"

free_gb=$(df -g /Volumes/sharts 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$free_gb" ] && [ "$free_gb" -lt 100 ]; then
    echo "WARNING: Only ${free_gb}GB free on /Volumes/sharts" | tee -a "$LOG"
fi

scan() {
    local label="$1" model="$2" output="$3"
    echo "" | tee -a "$LOG"
    echo "=== $label — $(date) ===" | tee -a "$LOG"
    set -o pipefail
    $PYTHON -m heinrich.cli mri-scan --model "$model" --output "$output" 2>&1 | tee -a "$LOG"
    local rc=$?
    set +o pipefail
    [ $rc -ne 0 ] && echo "  FAILED: $label (exit $rc)" | tee -a "$LOG"
}

# Smallest first, all using HF Hub IDs (no /tmp/ dependency)
scan "SmolLM2 135M"    mlx-community/SmolLM2-135M-Instruct       /Volumes/sharts/smollm2-135m
scan "SmolLM2 360M"    mlx-community/SmolLM2-360M-Instruct       /Volumes/sharts/smollm2-360m
scan "SmolLM2 1.7B"    mlx-community/SmolLM2-1.7B-Instruct       /Volumes/sharts/smollm2-1.7b
scan "Qwen 0.5B Base"  mlx-community/Qwen2.5-0.5B-4bit           /Volumes/sharts/qwen-0.5b-base
scan "Qwen 0.5B"       mlx-community/Qwen2.5-0.5B-Instruct-4bit /Volumes/sharts/qwen-0.5b
scan "Qwen 1.5B Base"  mlx-community/Qwen2.5-1.5B-4bit           /Volumes/sharts/qwen-1.5b-base
scan "Qwen 1.5B"       mlx-community/Qwen2.5-1.5B-Instruct-4bit /Volumes/sharts/qwen-1.5b
scan "Phi-2"           mlx-community/phi-2-4bit                   /Volumes/sharts/phi-2
scan "Phi-3"           mlx-community/Phi-3-mini-4k-instruct-4bit /Volumes/sharts/phi-3
scan "Qwen 3B Base"    mlx-community/Qwen2.5-3B-4bit             /Volumes/sharts/qwen-3b-base
scan "Qwen 3B"         mlx-community/Qwen2.5-3B-Instruct-4bit   /Volumes/sharts/qwen-3b
scan "Mistral 7B"      mlx-community/Mistral-7B-Instruct-v0.3-4bit /Volumes/sharts/mistral-7b
scan "Qwen 7B Base"    mlx-community/Qwen2.5-7B-4bit             /Volumes/sharts/qwen-7b-base
scan "Qwen 7B"         mlx-community/Qwen2.5-7B-Instruct-4bit   /Volumes/sharts/qwen-7b

echo "" | tee -a "$LOG"
echo "=== Final Health Check ===" | tee -a "$LOG"
$PYTHON -m heinrich.cli mri-health --dir /Volumes/sharts 2>&1 | tee -a "$LOG"
echo "=== DONE — $(date) ===" | tee -a "$LOG"
