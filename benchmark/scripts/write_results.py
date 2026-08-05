#!/usr/bin/env python3
"""Regenerate benchmark/results/RESULTS.md from the run JSONs on disk.

Runs unattended at the end of run_suite.sh, so it reports only what it can
read: a run whose JSON is missing (crashed, timed out, still going) is listed
as incomplete rather than silently omitted, since a suite summary that quietly
drops a failed configuration is worse than one that says it failed.
"""

import json
import pathlib
import datetime

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

# (filename stem, human label, one-line description)
RUNS = [
    (
        "locomo_deepseek_facts_on",
        "LoCoMo — fact extraction ON",
        "Current system. 2 conversations.",
    ),
    (
        "locomo_deepseek_facts_off",
        "LoCoMo — fact extraction OFF (ablation)",
        "Identical except extraction disabled. 2 conversations.",
    ),
    (
        "longmemeval_deepseek_facts_on",
        "LongMemEval — fact extraction ON",
        "20 items.",
    ),
]

CATEGORY_ORDER = ["single-hop", "multi-hop", "temporal", "open-domain", "cat_5"]


def load(stem):
    path = RESULTS / f"{stem}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def fmt_pct(x):
    return f"{x * 100:.1f}%"


def void_warning(s):
    """Flag a run whose answers were mostly never generated.

    A failed generate_answer is recorded as "Not found in memory" and scores
    0.0 — identical to a genuine retrieval miss. When nearly every answer looks
    like that, the run measured an outage (API credit exhaustion, network) and
    not the memory system, so it must not be read as a score.
    """
    rs = s.get("results") or []
    if not rs:
        return ""
    nf = sum(1 for r in rs if "not found in memory" in str(r.get("predicted", "")).lower())
    if nf / len(rs) < 0.9:
        return ""
    return (
        f"> **VOID — not a measurement.** {nf} of {len(rs)} answers were never "
        "generated (the run recorded the placeholder that a failed answer call "
        "produces, which scores 0.0 just like a real miss). The scores below "
        "reflect an outage, not retrieval quality. Check the run log for the "
        "cause and re-run before drawing any conclusion from this table."
    )


def summary_table(s):
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Questions | {s['total_questions']} |",
        f"| Exact match | {fmt_pct(s['exact_match'])} |",
        f"| Token F1 | {s['f1']:.3f} |",
    ]
    js = s.get("judge_score")
    lines.append(
        f"| **Judge score** | **{js:.3f}** |" if js is not None else "| Judge score | n/a |"
    )
    scored = s.get("judge_scored")
    missing = s.get("judge_missing")
    if scored is not None:
        lines.append(f"| Judge-scored questions | {scored} |")
    if missing:
        # Excluded from the mean above, so it has to be visible.
        lines.append(f"| **Unscored (NOT in judge mean)** | **{missing}** |")
    lines += [
        f"| Avg tokens/query | {s['avg_tokens_per_query']:.0f} |",
        f"| Latency p50 / p95 | {s['p50_latency_ms']}ms / {s['p95_latency_ms']}ms |",
        f"| Models | `{s['model_llm']}`, embed `{s['model_embed']}` |",
    ]
    return "\n".join(lines)


def category_table(s):
    per_cat = s.get("per_category") or {}
    if not per_cat:
        return ""
    keys = [c for c in CATEGORY_ORDER if c in per_cat]
    keys += sorted(k for k in per_cat if k not in CATEGORY_ORDER)
    out = [
        "",
        "| Category | n | Exact match | Token F1 | Judge score |",
        "|----------|---|-------------|----------|-------------|",
    ]
    for k in keys:
        m = per_cat[k]
        js = m.get("judge_score")
        out.append(
            f"| {k} | {m['count']} | {fmt_pct(m['exact_match'])} | "
            f"{m['f1']:.3f} | {js:.3f} |" if js is not None
            else f"| {k} | {m['count']} | {fmt_pct(m['exact_match'])} | {m['f1']:.3f} | n/a |"
        )
    return "\n".join(out)


def main():
    loaded = [(stem, label, desc, load(stem)) for stem, label, desc in RUNS]

    parts = [
        "# Mneme Benchmark Results",
        "",
        f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        "`benchmark/scripts/write_results.py`.",
        "",
        "## Read this before comparing to anything",
        "",
        "These runs are **not** leaderboard-comparable, for three reasons:",
        "",
        "1. **Subsets, not the full benchmark.** LoCoMo runs cover 2 of 10 "
        "conversations; the full set is 1,542 questions. A full DeepSeek run "
        "was measured at 7+ hours.",
        "2. **Wrong model for a submission.** The Agent Memory Leaderboard "
        "mandates gpt-4o-mini for the memory system's Add and Search. These "
        "runs use DeepSeek throughout because no funded OpenAI key was "
        "available. Expect the compliant number to differ — DeepSeek V4 is a "
        "reasoning model and extracts facts better than gpt-4o-mini will.",
        "3. **Self-judged.** The judge is also DeepSeek, so the same model "
        "family generates and scores the answers.",
        "",
        "Use these to compare configurations **against each other** — that is "
        "what the on/off pair is for — not to compare mneme against published "
        "numbers from other systems.",
        "",
        "## Results",
        "",
    ]

    for stem, label, desc, data in loaded:
        parts += [f"### {label}", "", desc, ""]
        if data is None:
            parts += [
                f"**Run did not complete** — no readable `{stem}.json`. "
                f"See `results/logs/{stem}.log`.",
                "",
            ]
            continue
        warn = void_warning(data)
        if warn:
            parts += [warn, ""]
        parts += [summary_table(data), category_table(data), ""]

    on = next((d for s, _, _, d in loaded if s == "locomo_deepseek_facts_on"), None)
    off = next((d for s, _, _, d in loaded if s == "locomo_deepseek_facts_off"), None)
    parts += ["## Fact extraction: on vs off", ""]
    if on and off and on.get("judge_score") is not None and off.get("judge_score") is not None:
        delta = on["judge_score"] - off["judge_score"]
        parts += [
            "| Config | Judge score | Token F1 | Exact match |",
            "|--------|-------------|----------|-------------|",
            f"| Extraction ON | {on['judge_score']:.3f} | {on['f1']:.3f} | "
            f"{fmt_pct(on['exact_match'])} |",
            f"| Extraction OFF | {off['judge_score']:.3f} | {off['f1']:.3f} | "
            f"{fmt_pct(off['exact_match'])} |",
            f"| **Delta** | **{delta:+.3f}** | {on['f1'] - off['f1']:+.3f} | "
            f"{(on['exact_match'] - off['exact_match']) * 100:+.1f}pp |",
            "",
            "Both halves ran on the same conversations with the same models, "
            "so the delta isolates the feature. It is a 2-conversation sample "
            "though — treat the sign as more reliable than the magnitude.",
            "",
        ]
    else:
        parts += ["Both halves of the pair did not complete; no comparison available.", ""]

    parts += [
        "## Superseded history",
        "",
        "Earlier result files in this directory (`locomo_hybrid_v2_full.json` "
        "and everything before it, including the frequently-quoted **0.388** "
        "overall judge score) were produced by a harness with defects found "
        "in a later audit. In those runs:",
        "",
        "- Fact extraction never ran — it existed only in the HTTP `/add` "
        "handler, while the benchmark wrote through `MnemeMemory::remember`.",
        "- Reranking and entity extraction were starved of output tokens on a "
        "DeepSeek judge and silently returned degraded defaults.",
        "- Working-memory candidates were passed downstream truncated at "
        "~100 characters.",
        "- Transport errors and 5xx were not retried, and a failed answer "
        "generation was recorded as a 0.0 indistinguishable from a genuine "
        "retrieval miss.",
        "",
        "Those numbers are a floor on the old system, not a fair measurement "
        "of it, and are not comparable to the runs above. The JSONs are kept "
        "for provenance.",
        "",
    ]

    (RESULTS / "RESULTS.md").write_text("\n".join(parts))
    print(f"wrote {RESULTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
