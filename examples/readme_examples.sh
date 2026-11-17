#!/usr/bin/env bash
# Re-runs every command shown in README.md and stores its output in examples/expected/.
# Needs network access to huggingface.co; every model is pinned to a commit so the output is stable.
set -u
cd "$(dirname "$0")/.."
out=examples/expected
mkdir -p "$out"

run() {
  local name=$1; shift
  { echo "\$ chatprefix $*"; chatprefix "$@"; echo "[exit $?]"; } > "$out/$name.txt" 2>&1
}

run qwen3     Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
run qwen25    Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
run deepseek  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 --check genprompt
run mistral   unsloth/mistral-7b-instruct-v0.3 --revision 60b99dcbc7f853b7cd18472a607e8d045b96f03e --check system --check bos
run llama     unsloth/Llama-3.2-1B-Instruct --revision 5a8abab4a5d6f164389b1079fb721cfab8d7126c --check bos
run reasoning Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --reasoning --check prefix --json
