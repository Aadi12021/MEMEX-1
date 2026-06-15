"""
run_monitor_test.py

End-to-end integration test: Self-Monitor Daemon + PERCEPT-1 + MEMEX-1

Tests three scenarios:
  1. Low load query — simple, known content, should pass meta-eval cleanly
  2. High load query — complex, multimodal, novel — should trigger threshold adjustment
  3. Uncertain query — ambiguous input designed to trigger REVISE verdict

To run:
    export OPENAI_API_KEY=your_key
    python run_monitor_test.py
"""

import os
from self_monitor_daemon import SelfMonitorDaemon


def print_divider(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def print_report(report):
    print(f"\n  Load level      : {report.load_state.level} ({report.load_state.total:.3f})")
    print(f"  Thresholds adj  : {report.thresholds_adjusted}")
    print(f"  Meta verdict    : {report.meta_score.verdict}")
    print(f"  Meta composite  : {report.meta_score.composite:.2f}")
    print(f"  Meta reasoning  : {report.meta_score.reasoning}")
    print(f"\n  FINAL RESPONSE:\n  {report.final_response[:300]}...")


def main():
    print_divider("SELF-MONITOR DAEMON × PERCEPT-1 × MEMEX-1 — INTEGRATION TEST")

    daemon = SelfMonitorDaemon(profile_path="./semantic_profile.json")

    # ─────────────────────────────────────────────────────────
    # TEST 1: Low load — simple known query
    # ─────────────────────────────────────────────────────────
    print_divider("TEST 1 — Low load, known content")

    report_1 = daemon.run_cycle(
        user_input="What is MEMEX-1 and what language is it built in?",
        source_label="user_input",
    )
    print_report(report_1)

    # ─────────────────────────────────────────────────────────
    # TEST 2: High load — complex novel multimodal query
    # ─────────────────────────────────────────────────────────
    print_divider("TEST 2 — High load, novel + structured data")

    report_2 = daemon.run_cycle(
        user_input=(
            "We are now extending the cognitive twin with a self-monitoring layer "
            "that tracks cognitive load using Sweller's three load types and performs "
            "metacognitive evaluation of every response before finalization. "
            "How does this connect to what we have already built?"
        ),
        structured_data={
            "new_module": "Self-Monitor Daemon",
            "week": 3,
            "components": ["LoadMonitor", "MetaEvaluator"],
            "cognitive_grounding": ["Sweller CLT", "Flavell metacognition"],
        },
        source_label="user_input",
    )
    print_report(report_2)

    # ─────────────────────────────────────────────────────────
    # TEST 3: Ambiguous query — designed to stress meta-eval
    # ─────────────────────────────────────────────────────────
    print_divider("TEST 3 — Ambiguous query (stress meta-evaluator)")

    report_3 = daemon.run_cycle(
        user_input="What should we do next and why does any of this matter?",
        source_label="user_input",
    )
    print_report(report_3)

    # ─────────────────────────────────────────────────────────
    # Final Tier 1 state
    # ─────────────────────────────────────────────────────────
    print_divider("FINAL TIER 1 STATE")
    tier1 = daemon.tier1
    print(f"\n  Messages in Tier 1 : {len(tier1.buffer)}")
    print(f"  Token usage        : {tier1.current_token_count}/{tier1.max_token_limit}")
    for i, msg in enumerate(tier1.buffer):
        preview = msg.content[:80].replace("\n", " ")
        print(f"  [{i+1}] {msg.role}: {preview}...")


if __name__ == "__main__":
    main()
