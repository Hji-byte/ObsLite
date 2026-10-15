#!/usr/bin/env bash
# Public entry points: resolve shared environments and paths automatically.
set -euo pipefail
usage() {
  cat <<'HELP'
Usage:
  bash run.sh sft BASE_MODEL [SFT_ADAPTER]
  bash run.sh export BASE_MODEL SFT_ADAPTER
  bash run.sh train BASE_MODEL [mask4|full] [SFT_ADAPTER]
  bash run.sh evaluate RL_ADAPTER_OR_DASH OUTPUT [mask_old_observations|full_history|recent_action_window]

SFT adapter default: models/sft-adapter
Exported SFT model: models/sft-export
Training output: runs/METHOD-grpo20
Use '-' for RL_ADAPTER_OR_DASH to evaluate the exported SFT model.
Run setup.sh first. Training checks the backend automatically; evaluation creates service metadata.
HELP
}
[[ $# -gt 0 ]] || { usage; exit 1; }
if [[ "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
TASK="$1"; shift
case "$TASK" in
  sft) [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 1; } ;;
  export) [[ $# == 2 ]] || { usage; exit 1; } ;;
  train) [[ $# -ge 1 && $# -le 3 ]] || { usage; exit 1; }
    [[ "${2:-mask4}" == mask4 || "${2:-mask4}" == full ]] || { usage; exit 1; } ;;
  evaluate) [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
    case "${3:-mask_old_observations}" in mask_old_observations|full_history|recent_action_window) ;; *) usage; exit 1;; esac ;;
  *) usage; exit 1 ;;
esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Relative resource paths are interpreted from the repository root.
cd "$ROOT"
command -v conda >/dev/null || { echo "Conda is required; run setup.sh first." >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mask-grpo-model
MODEL_PYTHON="$(command -v python)"
ENV_PYTHON="$(conda run -n mask-grpo-webshop python -c 'import sys; print(sys.executable)')"
export JAVA_HOME="$(conda run -n mask-grpo-webshop python -c 'import sys; print(sys.prefix)')"
export PYTHONPATH="$ROOT:$ROOT/external/verl-runtime${PYTHONPATH:+:$PYTHONPATH}"
SFT_EXPORT="$ROOT/models/sft-export"
case "$TASK" in
  sft)
    ADAPTER="${2:-$ROOT/models/sft-adapter}"
    python -m scripts.train_student --model "$1" --data "$ROOT/data" --output "$ADAPTER"
    python -m tools.export_sft --model "$1" --adapter "$ADAPTER" --output "$SFT_EXPORT"
    ;;
  export)
    python -m tools.export_sft --model "$1" --adapter "$2" --output "$SFT_EXPORT"
    ;;
  train)
    METHOD="${2:-mask4}"
    ADAPTER="${3:-$ROOT/models/sft-adapter}"
    RUN="$ROOT/runs/$METHOD-grpo20"
    python -m tools.prepare_experiment --method "$METHOD" --steps 20 \
      --output "$RUN" --model "$SFT_EXPORT" --snapshot "$ROOT/data" \
      --eto-root "$ROOT/external/ETO" --task-list "$ROOT/data" \
      --environment-python "$ENV_PYTHON"
    python -m scripts.launch_rl --prepared "$RUN" --upstream "$ROOT/external/verl-agent" \
      --runtime "$ROOT/external/verl-runtime" --base "$1" --adapter "$ADAPTER" --run
    ;;
  evaluate)
    POLICY="${3:-mask_old_observations}"
    KEEP=4
    [[ "$POLICY" != recent_action_window ]] || KEEP=2
    ADAPTER_ARGS=()
    [[ "$1" == - ]] || ADAPTER_ARGS=(--adapter "$1")
    python -m tools.evaluate_checkpoint \
      --model "$SFT_EXPORT" "${ADAPTER_ARGS[@]}" \
      --model-python "$MODEL_PYTHON" --environment-python "$ENV_PYTHON" \
      --snapshot "$ROOT/data" \
      --dataset "$ROOT/data/train_tasks.jsonl" --eto-root "$ROOT/external/ETO" \
      --output "$2" --context-policy "$POLICY" --keep-recent-history "$KEEP" --run
    ;;
esac
