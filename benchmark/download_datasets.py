#!/usr/bin/env python3
"""
Download and convert LoCoMo and LongMemEval to the JSON format
expected by the Rust benchmark harness.

Output:
  benchmark/data/locomo_bench.json
  benchmark/data/longmemeval_bench.json
"""

import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# LoCoMo QA category names
LOCOMO_CATEGORIES = {
    1: "single-hop",
    2: "multi-hop",
    3: "open-domain",
    4: "temporal",
}


# ─────────────────────────────────────────────────────────────
# LoCoMo  (KimmoZZZ/locomo  →  locomo10.json)
# ─────────────────────────────────────────────────────────────

def convert_locomo():
    print("Downloading LoCoMo (KimmoZZZ/locomo)...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("KimmoZZZ/locomo", filename="locomo10.json", repo_type="dataset")
    with open(path) as f:
        raw = json.load(f)

    conversations = []
    for row in raw:
        conv_id = row["sample_id"]
        conv_data = row["conversation"]

        # Extract sessions in order
        sessions = []
        si = 1
        while True:
            key = f"session_{si}"
            turns_raw = conv_data.get(key)
            if turns_raw is None:
                break
            date_str = conv_data.get(f"session_{si}_date_time", "")
            turns = []
            for t in turns_raw:
                speaker = t.get("speaker", "")
                text = t.get("text", "")
                if text:
                    turns.append(f"{speaker}: {text}" if speaker else text)
            if turns:
                sessions.append({
                    "session_id": f"{conv_id}_s{si}",
                    "session_date": date_str,
                    "turns": turns,
                })
            si += 1

        # Extract QA pairs
        questions = []
        for qi, qa in enumerate(row.get("qa", [])):
            q = str(qa.get("question", "")).strip()
            a = qa.get("answer", "")
            a = str(a).strip() if a is not None else ""
            cat_num = qa.get("category", 0)
            cat = LOCOMO_CATEGORIES.get(cat_num, f"cat_{cat_num}")
            if q and a:
                questions.append({
                    "question_id": f"{conv_id}_q{qi}",
                    "question": q,
                    "answer": a,
                    "category": cat,
                })

        if sessions and questions:
            conversations.append({
                "conversation_id": conv_id,
                "sessions": sessions,
                "questions": questions,
            })

    total_q = sum(len(c["questions"]) for c in conversations)
    print(f"  {len(conversations)} conversations, {total_q} questions")
    out = OUT_DIR / "locomo_bench.json"
    out.write_text(json.dumps(conversations, indent=2, ensure_ascii=False))
    print(f"  -> {out}")
    return True


# ─────────────────────────────────────────────────────────────
# LongMemEval  (longmemeval_oracle)
# ─────────────────────────────────────────────────────────────

def convert_longmemeval():
    print("Loading LongMemEval (longmemeval_oracle)...")
    oracle_path = Path("/tmp/longmemeval/longmemeval_oracle")
    if not oracle_path.exists():
        print("  File not found at /tmp/longmemeval/longmemeval_oracle")
        print("  Attempting re-download...")
        from huggingface_hub import snapshot_download
        snapshot_download("xiaowu0162/longmemeval", repo_type="dataset",
                          local_dir="/tmp/longmemeval", ignore_patterns=["longmemeval_m", "longmemeval_s"])

    with open(oracle_path) as f:
        raw = json.load(f)

    items = []
    for row in raw:
        hist_id = str(row.get("question_id", f"h_{len(items)}"))
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        category = str(row.get("question_type", "general"))

        # Collect all haystack sessions as turns, prepending date headers
        turns = []
        haystack_dates = row.get("haystack_dates", [])
        for si, session in enumerate(row.get("haystack_sessions", [])):
            date_str = haystack_dates[si] if si < len(haystack_dates) else ""
            if date_str:
                turns.append(f"[Session on {date_str}]")
            for t in session:
                role = t.get("role", "")
                content = t.get("content", "")
                if content:
                    turns.append(f"{role}: {content}" if role else content)

        if question and answer and turns:
            items.append({
                "history_id": hist_id,
                "turns": turns,
                "question": question,
                "answer": answer,
                "category": category,
            })

    print(f"  {len(items)} items")
    out = OUT_DIR / "longmemeval_bench.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"  -> {out}")
    return True


if __name__ == "__main__":
    ok1 = convert_locomo()
    print()
    ok2 = convert_longmemeval()
    print()
    if ok1 and ok2:
        print("Both datasets ready.")
    else:
        sys.exit(1)
