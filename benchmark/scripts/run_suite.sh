#!/usr/bin/env bash
# Unattended benchmark suite.
#
# Runs a fixed set of configurations back to back and regenerates
# benchmark/results/RESULTS.md from whatever finished. Designed to survive the
# terminal closing: launch with `setsid nohup ... & disown`.
#
# Each run is independent — one failing does not stop the rest, because a
# multi-hour suite that aborts on the last config wastes everything before it.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

BIN=./target/release/mneme-bench
DATA_LOCOMO=benchmark/data/locomo_bench.json
DATA_LME=benchmark/data/longmemeval_bench.json
RESULTS=benchmark/results
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

# DeepSeek only — there is no funded OpenAI key. Local BGE embeddings need no
# key at all. See SUBMISSION.md for why this is NOT a leaderboard-compliant
# configuration (which mandates gpt-4o-mini for the memory system).
export MNEME_EMBED_BACKEND=local
unset OPENAI_API_KEY ANTHROPIC_API_KEY

run() {
  local name=$1; shift
  echo "=== [$(date -Is)] START $name ===" >> "$LOGS/suite.log"
  # shellcheck disable=SC2086
  timeout 21600 "$BIN" "$@" --out "$RESULTS/$name.json" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "=== [$(date -Is)] END $name (exit $rc) ===" >> "$LOGS/suite.log"
  return $rc
}

echo "=== [$(date -Is)] SUITE START ===" >> "$LOGS/suite.log"

# Primary measurement: the current system, fact extraction on.
run locomo_deepseek_facts_on \
  --bench locomo --data "$DATA_LOCOMO" --limit 2 \
  --memory-llm deepseek --judge-llm deepseek --embed local --top-k 5 \
  --fact-extraction true

# Ablation: identical except fact extraction is off. The pair is the point —
# it isolates the contribution of the feature rather than reporting one number.
run locomo_deepseek_facts_off \
  --bench locomo --data "$DATA_LOCOMO" --limit 2 \
  --memory-llm deepseek --judge-llm deepseek --embed local --top-k 5 \
  --fact-extraction false

# LongMemEval has only ever had a 50-item smoke run; this is a second dataset
# rather than a second reading of the same one.
run longmemeval_deepseek_facts_on \
  --bench longmemeval --data "$DATA_LME" --limit 20 \
  --memory-llm deepseek --judge-llm deepseek --embed local --top-k 5 \
  --fact-extraction true

python3 benchmark/scripts/write_results.py >> "$LOGS/suite.log" 2>&1
echo "=== [$(date -Is)] SUITE DONE ===" >> "$LOGS/suite.log"
