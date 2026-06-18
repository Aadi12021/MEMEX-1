"""
PERCEPT-1: Perception Daemon for the Cognitive Digital Twin

Architecture:
    RawPerceptInput
        → MultimodalBinder      (sensory integration)
        → PredictiveCoder       (prior retrieval + surprise scoring)
        → AttentionFilter       (salience gating)
        → SalientPerceptObject  → Tier1WorkingMemory.add_message()

Cognitive science grounding:
    - Karl Friston's Free Energy Principle / Predictive Coding
    - Broadbent's Filter Model + Treisman's Attenuation Theory
    - Binding Problem / Multisensory Integration

MEMEX-1 integration:
    - Reads Tier 3 global profile as the predictive prior
    - Outputs feed directly into Tier 1 working memory
    - Surprise score maps onto the same distance-gating logic
      used by RetrievalArbiter (threshold: 0.25 - 0.6)
"""

from typing import Optional
import config
from multimodal_binder import MultimodalBinder
from predictive_coder import PredictiveCoder
from attention_filter import AttentionFilter
from percept_schemas import RawPerceptInput, SalientPerceptObject


class Percept1Daemon:
    """
    Main orchestrator for the PERCEPT-1 perception pipeline.

    Usage:
        daemon = Percept1Daemon(profile_path="./semantic_profile.json")

        result = daemon.perceive(
            text="Some input text",
            structured={"key": "value"},
            image_base64="...",         # optional
        )

        # Feed into MEMEX-1 Tier 1:
        tier1.add_message("user", result.to_message_content())
    """

    def __init__(
        self,
        profile_path: str = config.PROFILE_PATH,
        high_threshold: float = config.PERCEPT_HIGH_THRESHOLD,
        mid_threshold: float = config.PERCEPT_MID_THRESHOLD,
        embedding_model: str = config.EMBEDDING_MODEL,
    ):
        self.binder = MultimodalBinder()
        self.coder = PredictiveCoder(
            profile_path=profile_path,
            embedding_model=embedding_model,
        )
        self.filter = AttentionFilter(
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
        )

    def perceive(
        self,
        text: Optional[str] = None,
        structured: Optional[dict] = None,
        image_base64: Optional[str] = None,
        image_media_type: str = "image/jpeg",
        source_label: str = "external",
    ) -> SalientPerceptObject:
        """
        Full PERCEPT-1 pipeline. Takes raw multimodal input,
        runs it through all three stages, and returns a
        SalientPerceptObject ready for MEMEX-1 ingestion.

        Returns a SUPPRESSED percept (salience='SUPPRESSED') if the
        input falls below the novelty threshold — the caller can
        choose to discard these rather than feeding them to Tier 1.
        """
        raw = RawPerceptInput(
            text=text,
            structured=structured,
            image_base64=image_base64,
            image_media_type=image_media_type,
            source_label=source_label,
        )

        print("\n⚡ [PERCEPT-1] ── Perception pipeline initiated")

        # Stage 1: Multimodal Binding
        print("  🔗 [STAGE 1] Multimodal Binder...")
        percept_obj = self.binder.bind(raw)
        print(f"     Modalities bound: {percept_obj.modalities_present}")

        # Stage 2: Predictive Coding
        print("  🔮 [STAGE 2] Predictive Coder...")
        surprise_score, prior_summary, prior_diff = self.coder.compute_surprise(percept_obj)

        # Stage 3: Attention Filter
        print("  🎯 [STAGE 3] Attention Filter...")
        salient_percept = self.filter.filter(
            percept_obj, surprise_score, prior_summary, prior_diff
        )

        print(f"⚡ [PERCEPT-1] ── Pipeline complete | salience={salient_percept.salience} | surprise={surprise_score:.4f}\n")
        return salient_percept