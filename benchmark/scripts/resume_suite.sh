#!/usr/bin/env bash
# Resume the benchmark suite after an interruption (machine shutdown, reboot,
# suspend — none of which a detached process survives).
#
# Same configurations as run_suite.sh, but skips any run whose results JSON is
# already on disk, so a suite that died partway through doesn't repay the hours
# it already spent. Delete a JSON to force that run again.
#
#   setsid nohup ./benchmark/scripts/resume_suite.sh > /dev/null 2>&1 &
#
# Kept separate from run_suite.sh deliberately: bash reads a script
# incrementally as it executes, so editing the file a live suite is running
# from can corrupt it mid-run.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

BIN=./target/release/mneme-bench
DATA_LOCOMO=benchmark/data/locomo_bench.json
DATA_LME=benchmark/data/longmemeval_bench.json
RESULTS=benchmark/results
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

export MNEME_EMBED_BACKEND=local
unset OPENAI_API_KEY ANTHROPIC_API_KEY

run() {
  local name=$1; shift
  if [ -s "$RESULTS/$name.json" ]; then
    echo "=== [$(date -Is)] SKIP $name (results already present) ===" >> "$LOGS/suite.log"
    return 0
  fi
  echo "=== [$(date -Is)] START $name ===" >> "$LOGS/suite.log"
  timeout 21600 "$BIN" "$@" --out "$RESULTS/$name.json" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "=== [$(date -Is)] END $name (exit $rc) ===" >> "$LOGS/suite.log"
  return $rc
}

echo "=== [$(date -Is)] SUITE RESUME ===" >> "$LOGS/suite.log"

run locomo_deepseek_facts_on \
  --bench locomo --data "$DATA_LOCOMO" --limit 2 \
  --memory-llm deepseek --judge-llm deepseek --embed local --top-k 5 \
  --fact-extraction true

run locomo_deepseek_facts_off \
  --bench locomo --data "$DATA_LOCOMO" --limit 2 \
  --memory-llm deepseek --judge-llm deepseek --embed local --top-k 5 \
  --fact-extraction false

run longmemeval_deepseek_facts_on \
  --bench longmemeval --data "$DATA_LME" --limit 20 \
  --memory-llm deepseek --judge-llm deepseek --embed local --top-k 5 \
  --fact-extraction true

python3 benchmark/scripts/write_results.py >> "$LOGS/suite.log" 2>&1
echo "=== [$(date -Is)] SUITE DONE ===" >> "$LOGS/suite.log"
