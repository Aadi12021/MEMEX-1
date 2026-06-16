"""
self_monitor_daemon.py

Week 3 of the Cognitive Digital Twin series.

The Self-Monitor Daemon wraps the full MEMEX-1 + PERCEPT-1 stack
and adds two layers of self-awareness:

    1. Cognitive Load Monitoring (Sweller's CLT)
       Tracks intrinsic, extraneous, and germane load in real time.
       Dynamically adjusts PERCEPT-1 attention thresholds and
       MEMEX-1 Tier 1 token limits based on current processing burden.

    2. Metacognitive Evaluation (Flavell's metacognition)
       Runs a second pass on every response before finalization.
       Scores confidence, completeness, and consistency.
       Triggers a revision if the response doesn't meet the bar.

One cycle:
    perceive → check load → adjust thresholds → compile context
    → generate response → evaluate → revise if needed → return report

Cognitive science grounding:
    - Sweller (1988): Cognitive Load Theory
    - Flavell (1979): Metacognitive monitoring and control
    - Both are functions of the prefrontal cortex — the brain's
      executive overseer of its own processing.
"""

import os
from typing import Optional
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from memory_tier1 import Tier1WorkingMemory
from memory_tier2 import Tier2EpisodicMemory
from retrieval_arbiter import RetrievalArbiter
from percept1_daemon import Percept1Daemon
from load_monitor import LoadMonitor
from meta_evaluator import MetaEvaluator
from monitor_schemas import MonitorReport


class SelfMonitorDaemon:
    """
    Main orchestrator for Week 3.

    Usage:
        daemon = SelfMonitorDaemon()
        report = daemon.run_cycle(
            user_input="Your query here",
            image_base64=None,       # optional
            structured_data=None,    # optional
        )
        print(report.final_response)
        print(report.summary())
    """

    def __init__(
        self,
        profile_path: str = "./semantic_profile.json",
        persist_directory: str = "./.memex_storage",
        base_token_limit: int = 2000,
        base_surprise_threshold: float = 0.6,
    ):
        self.profile_path = profile_path
        self.base_token_limit = base_token_limit
        self.base_surprise_threshold = base_surprise_threshold

        # Stack components
        self.tier1  = Tier1WorkingMemory(session_id="monitor_session", max_token_limit=base_token_limit)
        self.tier2  = Tier2EpisodicMemory(persist_directory=persist_directory)
        self.arbiter = RetrievalArbiter(profile_path=profile_path, persist_directory=persist_directory)
        self.percept = Percept1Daemon(
            profile_path=profile_path,
            high_threshold=base_surprise_threshold,
        )

        # Week 3 components
        self.load_monitor   = LoadMonitor()
        self.meta_evaluator = MetaEvaluator(profile_path=profile_path)

        # OpenAI for response generation
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Cycle tracking for load signals
        self._percept_count    = 0
        self._suppressed_count = 0
        self._tier3_updates    = 0

    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=False,
    )
    def _generate_response(self, compiled_context: str) -> str:
        """
        Generates a response from the compiled MEMEX-1 context.
        Retries up to 3 times with exponential backoff on transient API errors.
        Falls back to a safe default message if all retries fail.
        """
        try:
            result = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant operating within a cognitive architecture. "
                            "Use the provided memory context to answer accurately and concisely."
                        ),
                    },
                    {"role": "user", "content": compiled_context},
                ],
                temperature=0.2,
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  [MONITOR] Response generation failed after retries: {e}")
            return "I was unable to generate a response due to a temporary API issue. Please try again."

    def _adjust_thresholds(self, recommended_token_limit: int, recommended_surprise_threshold: float):
        """Applies load-driven threshold adjustments to PERCEPT-1 and Tier 1."""
        adjusted = False

        if self.tier1.max_token_limit != recommended_token_limit:
            print(f"  🔧 [MONITOR] Tier 1 token limit: {self.tier1.max_token_limit} → {recommended_token_limit}")
            self.tier1.max_token_limit = recommended_token_limit
            adjusted = True

        if self.percept.filter.high_threshold != recommended_surprise_threshold:
            print(f"  🔧 [MONITOR] PERCEPT-1 HIGH threshold: {self.percept.filter.high_threshold} → {recommended_surprise_threshold}")
            self.percept.filter.high_threshold = recommended_surprise_threshold
            adjusted = True

        return adjusted

    def run_cycle(
        self,
        user_input: str,
        structured_data: Optional[dict] = None,
        image_base64: Optional[str] = None,
        image_media_type: str = "image/jpeg",
        source_label: str = "user_input",
    ) -> MonitorReport:
        """
        Full self-monitoring cycle. Returns a MonitorReport.
        """
        print("\n" + "="*60)
        print("  SELF-MONITOR DAEMON — cycle initiated")
        print("="*60)

        # Start latency clock for load monitor
        self.load_monitor.start_cycle()

        modality_count = 1
        if structured_data: modality_count += 1
        if image_base64:    modality_count += 1

        # ── Step 1: Perception ──────────────────────────────────
        print("\n⚡ [STEP 1] Running PERCEPT-1...")
        salient = self.percept.perceive(
            text=user_input,
            structured=structured_data,
            image_base64=image_base64,
            image_media_type=image_media_type,
            source_label=source_label,
        )
        self._percept_count += 1
        if salient.salience == "SUPPRESSED":
            self._suppressed_count += 1

        # ── Step 2: Load monitoring ─────────────────────────────
        print("\n⚡ [STEP 2] Computing cognitive load...")
        load_state = self.load_monitor.compute(
            input_text=user_input,
            modality_count=modality_count,
            surprise_score=salient.surprise_score,
            suppressed_percept_count=self._suppressed_count,
            total_percept_count=self._percept_count,
            tier3_update_count=self._tier3_updates,
        )

        # ── Step 3: Threshold adjustment ────────────────────────
        print("\n⚡ [STEP 3] Adjusting stack thresholds...")
        adjusted = self._adjust_thresholds(
            load_state.recommended_token_limit,
            load_state.recommended_surprise_threshold,
        )
        if not adjusted:
            print("  ✅ Thresholds unchanged.")

        # ── Step 4: Feed percept into MEMEX-1 ───────────────────
        if salient.salience != "SUPPRESSED":
            active, evicted = self.tier1.add_message("user", salient.to_message_content())
            if evicted:
                self.tier2.process_eviction(
                    session_id=self.tier1.session_id,
                    evicted_messages=evicted,
                )

        # ── Step 5: Compile context via RetrievalArbiter ────────
        print("\n⚡ [STEP 4] Compiling MEMEX-1 context...")
        compiled_context = self.arbiter.compile_context(user_input, self.tier1)

        # ── Step 6: Generate response ───────────────────────────
        print("\n⚡ [STEP 5] Generating response...")
        raw_response = self._generate_response(compiled_context)
        print(f"  📤 Raw response: {raw_response[:100]}...")

        # ── Step 7: Metacognitive evaluation ────────────────────
        print("\n⚡ [STEP 6] Running metacognitive evaluation...")
        meta_score = self.meta_evaluator.evaluate(user_input, raw_response)

        # Use revised response if available, otherwise raw
        final_response = meta_score.revised_response or raw_response

        # ── Step 8: Log assistant response into Tier 1 ──────────
        active, evicted = self.tier1.add_message("assistant", final_response)
        if evicted:
            self.tier2.process_eviction(
                session_id=self.tier1.session_id,
                evicted_messages=evicted,
            )

        report = MonitorReport(
            query=user_input,
            raw_response=raw_response,
            final_response=final_response,
            load_state=load_state,
            meta_score=meta_score,
            thresholds_adjusted=adjusted,
        )

        print(f"\n{'='*60}")
        print(f"  CYCLE COMPLETE: {report.summary()}")
        print(f"{'='*60}\n")

        return report
