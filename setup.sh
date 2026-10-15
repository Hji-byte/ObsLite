#!/usr/bin/env bash
# Install the shared model and WebShop environments on Linux.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash setup.sh"
  echo "Creates mask-grpo-model and mask-grpo-webshop, downloads pinned sources, and prepares runtime support."
  echo "Downloads and extracts WebShop catalog and search index. Model weights are prepared separately."
  exit 0
fi
[[ $# == 0 ]] || { echo "Unknown arguments. Use --help." >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "Linux x86-64 is required." >&2; exit 1; }
for command in conda git; do
  command -v "$command" >/dev/null || { echo "Install $command first." >&2; exit 1; }
done
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p external
fetch_source() {
  local url="$1" target="$2" revision="$3"
  if [[ ! -e "$target" ]]; then
    git clone "$url" "$target"
    git -C "$target" checkout --detach "$revision"
  fi
  [[ -d "$target/.git" ]] || { echo "Not a checkout: $target" >&2; exit 1; }
  [[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]] || { echo "Unexpected revision in $target; existing files were preserved." >&2; exit 1; }
  [[ -z "$(git -C "$target" status --porcelain --untracked-files=no)" ]] || { echo "Modified source in $target; existing files were preserved." >&2; exit 1; }
}
fetch_source https://github.com/langfengQ/verl-agent external/verl-agent 20bd331bdbc9026a5668e11362178e10ab7400c8
fetch_source https://github.com/Yifan-Song793/ETO external/ETO a2fc5da38f8d00cfaf3f9b6370d586eebaf72904
for environment in mask-grpo-model mask-grpo-webshop; do
  if ! conda run -n "$environment" python --version >/dev/null 2>&1; then
    conda create -n "$environment" python=3.12 -y
  fi
  conda run -n "$environment" python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
done
conda install -n mask-grpo-webshop -c conda-forge openjdk=11 -y
conda run --no-capture-output -n mask-grpo-model python -m pip install -r requirements/model.txt
conda run --no-capture-output -n mask-grpo-webshop python -m pip install -r requirements/webshop.txt
conda run --no-capture-output -n mask-grpo-webshop python -m scripts.setup_webshop_data --eto-root external/ETO
# The framework is loaded through PYTHONPATH; the simulator bridge loads ETO directly.
# Do not install ETO's historical package requirements over our recorded versions.
if [[ ! -e external/verl-runtime ]]; then
  conda run --no-capture-output -n mask-grpo-model python -m scripts.prepare_rl_runtime --upstream external/verl-agent --output external/verl-runtime
fi
conda run --no-capture-output -n mask-grpo-model python -c 'from pathlib import Path; from scripts.launch_rl import verify_runtime; verify_runtime(Path("external/verl-runtime"), Path("external/verl-agent"))'
PYTHONPATH="$ROOT:$ROOT/external/verl-runtime${PYTHONPATH:+:$PYTHONPATH}" conda run --no-capture-output -n mask-grpo-model python -m scripts.preflight_rl_fast --backend fla --output external/import_check.json
echo "Setup complete. Use bash run.sh --help for training and evaluation; environments and shared paths are handled automatically."
echo "This checks imports and compatibility, not GPU training."
