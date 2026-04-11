#!/bin/bash
# Full MRI scan for all models. One command per model, full workup.
cd /Users/asuramaya/Code/heinrich

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: venv not found at $PYTHON. Run: python3 -m venv .venv && pip install -e '.[mlx,dev]'"
    exit 1
fi

LOGFILE="scripts/mri_finish.log"
echo "=== MRI Scan Queue — $(date) ===" | tee -a "$LOGFILE"

# Disk space check
free_gb=$(df -g /Volumes/sharts 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$free_gb" ] && [ "$free_gb" -lt 50 ]; then
    echo "WARNING: Only ${free_gb}GB free on /Volumes/sharts (need ~50GB for full queue)" | tee -a "$LOGFILE"
    echo "Continue? [y/N]"
    read -r ans
    [ "$ans" != "y" ] && exit 1
fi

scan() {
    local label="$1" model="$2" output="$3"
    shift 3
    # Skip /tmp/ models that don't exist
    if [[ "$model" == /tmp/* ]] && [ ! -d "$model" ]; then
        echo "  SKIP: $label (model not at $model)" | tee -a "$LOGFILE"
        return 0
    fi
    echo "" | tee -a "$LOGFILE"
    echo "=== $label ===" | tee -a "$LOGFILE"
    set -o pipefail
    $PYTHON -m heinrich.cli mri-scan \
        --model "$model" --output "$output" "$@" 2>&1 | tee -a "$LOGFILE"
    local rc=$?
    set +o pipefail
    if [ $rc -ne 0 ]; then
        echo "  FAILED: $label (exit $rc)" | tee -a "$LOGFILE"
    fi
}

# Smallest first, largest last
scan "SmolLM2 135M"    /tmp/SmolLM2-135M-Instruct-mlx       /Volumes/sharts/smollm2-135m
scan "SmolLM2 360M"    /tmp/SmolLM2-360M-Instruct-mlx       /Volumes/sharts/smollm2-360m
scan "SmolLM2 1.7B"    /tmp/SmolLM2-1.7B-Instruct-mlx       /Volumes/sharts/smollm2-1.7b
scan "Qwen 0.5B Base"  /tmp/qwen-qwen2.5-0.5b-mlx           /Volumes/sharts/qwen-0.5b-base
scan "Qwen 0.5B"       mlx-community/Qwen2.5-0.5B-Instruct-4bit /Volumes/sharts/qwen-0.5b
scan "Qwen 1.5B"       /tmp/qwen-qwen2.5-1.5b-instruct-mlx  /Volumes/sharts/qwen-1.5b
scan "Qwen 3B Base"    mlx-community/Qwen2.5-3B-4bit         /Volumes/sharts/qwen-3b-base
scan "Qwen 3B"         mlx-community/Qwen2.5-3B-Instruct-4bit /Volumes/sharts/qwen-3b
scan "Phi-2"           /tmp/microsoft-phi-2-mlx              /Volumes/sharts/phi-2
scan "Phi-3"           mlx-community/Phi-3-mini-4k-instruct-4bit /Volumes/sharts/phi-3
scan "Mistral 7B Base" /tmp/mistralai-mistral-7b-v0.1-mlx    /Volumes/sharts/mistral-7b-base
scan "Mistral 7B"      mlx-community/Mistral-7B-Instruct-v0.3-4bit /Volumes/sharts/mistral-7b
scan "Gemma 2B"        /tmp/gemma-2-2b-it-mlx                /Volumes/sharts/gemma-2b
scan "Qwen 7B Base"    mlx-community/Qwen2.5-7B-4bit         /Volumes/sharts/qwen-7b-base
scan "Qwen 7B"         mlx-community/Qwen2.5-7B-Instruct-4bit /Volumes/sharts/qwen-7b

echo "" | tee -a "$LOGFILE"
echo "=== Final Health Check ===" | tee -a "$LOGFILE"
$PYTHON -m heinrich.cli mri-health --dir /Volumes/sharts 2>&1 | tee -a "$LOGFILE"
echo "=== DONE — $(date) ===" | tee -a "$LOGFILE"
