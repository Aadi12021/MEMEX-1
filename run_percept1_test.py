"""
run_percept1_test.py

End-to-end integration test: PERCEPT-1 + MEMEX-1

Tests three scenarios:
  1. Novel input (no prior) — should be HIGH salience
  2. Input consistent with Tier 3 profile — should be SUPPRESSED or MEDIUM
  3. Multimodal input (text + structured) — validates binder + full pipeline

To run:
    export ANTHROPIC_API_KEY=your_key
    python run_percept1_test.py
"""

import sys
import os

# Allow running from the outputs/percept1 directory alongside MEMEX-1 files
# In production, adjust sys.path to point to your MEMEX-1 root
sys.path.insert(0, os.path.abspath("."))

from percept1_daemon import Percept1Daemon
from memory_tier1 import Tier1WorkingMemory
from memory_tier2 import Tier2EpisodicMemory


def print_divider(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    print_divider("PERCEPT-1 × MEMEX-1 INTEGRATION TEST")

    # Initialize PERCEPT-1 pointed at the MEMEX-1 Tier 3 profile
    daemon = Percept1Daemon(profile_path="./semantic_profile.json")

    # Initialize MEMEX-1 components for downstream handoff
    tier1 = Tier1WorkingMemory(session_id="percept_test_001", max_token_limit=500)
    tier2 = Tier2EpisodicMemory()

    # ─────────────────────────────────────────────────────
    # TEST 1: Fully novel input (empty prior → should be HIGH)
    # ─────────────────────────────────────────────────────
    print_divider("TEST 1 — Novel text input (empty prior)")

    result_1 = daemon.perceive(
        text="We are now adding a perception layer to the cognitive twin. "
             "It processes multimodal inputs using predictive coding.",
        source_label="user_input",
    )

    print(f"\n  Salience    : {result_1.salience}")
    print(f"  Surprise    : {result_1.surprise_score:.4f}")
    print(f"  Prior diff  : {result_1.prior_diff}")

    if result_1.salience != "SUPPRESSED":
        active, evicted = tier1.add_message("user", result_1.to_message_content())
        if evicted:
            tier2.process_eviction(session_id=tier1.session_id, evicted_messages=evicted)
        print(f"\n  ✅ Percept injected into MEMEX-1 Tier 1")
    else:
        print(f"\n  ⬇️  Percept suppressed — not injected into MEMEX-1")

    # ─────────────────────────────────────────────────────
    # TEST 2: Multimodal input (text + structured data)
    # ─────────────────────────────────────────────────────
    print_divider("TEST 2 — Multimodal input (text + structured)")

    result_2 = daemon.perceive(
        text="New sensor reading from the environment.",
        structured={
            "temperature_c": 22.4,
            "location": "lab",
            "activity": "coding",
            "time_of_day": "afternoon",
        },
        source_label="sensor_feed",
    )

    print(f"\n  Salience    : {result_2.salience}")
    print(f"  Surprise    : {result_2.surprise_score:.4f}")
    print(f"  Prior diff  : {result_2.prior_diff}")
    print(f"  Modalities  : {result_2.percept.modalities_present}")

    if result_2.salience != "SUPPRESSED":
        active, evicted = tier1.add_message("user", result_2.to_message_content())
        if evicted:
            tier2.process_eviction(session_id=tier1.session_id, evicted_messages=evicted)
        print(f"\n  ✅ Multimodal percept injected into MEMEX-1 Tier 1")

    # ─────────────────────────────────────────────────────
    # TEST 3: Input consistent with known profile (should suppress)
    # ─────────────────────────────────────────────────────
    print_divider("TEST 3 — Input consistent with MEMEX-1 Tier 3 profile")

    result_3 = daemon.perceive(
        text="MEMEX-1 is a Python-based multi-tier memory daemon using ChromaDB.",
        source_label="user_input",
    )

    print(f"\n  Salience    : {result_3.salience}")
    print(f"  Surprise    : {result_3.surprise_score:.4f}")
    print(f"  Prior diff  : {result_3.prior_diff}")

    if result_3.salience == "SUPPRESSED":
        print("\n  ✅ PASS: Already-known input correctly suppressed by attention filter")
    else:
        print(f"\n  ⚠️  INFO: Input passed through with salience={result_3.salience} — profile may not yet reflect this content")

    # ─────────────────────────────────────────────────────
    # Final Tier 1 state
    # ─────────────────────────────────────────────────────
    print_divider("FINAL TIER 1 STATE")
    active_context = tier1.get_active_context()
    print(f"\n  Messages in Tier 1: {len(active_context)}")
    print(f"  Token usage: {tier1.current_token_count}/{tier1.max_token_limit}")
    for i, msg in enumerate(active_context):
        preview = msg.content[:80].replace("\n", " ")
        print(f"  [{i+1}] {msg.role}: {preview}...")


if __name__ == "__main__":
    main()
