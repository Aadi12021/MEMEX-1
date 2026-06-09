from percept_schemas import PerceptObject, SalientPerceptObject


class AttentionFilter:
    """
    Stage 3 of PERCEPT-1.

    Implements selective attention as a surprise-gated routing mechanism.

    Cognitive analog: Broadbent's Filter Model + Treisman's Attenuation Theory.
    The brain doesn't process everything at full resolution — it promotes
    high-salience signals (novel, unexpected) and suppresses low-salience ones
    (predictable, already known). Surprise is the gating signal.

    Routing logic:
        surprise >= HIGH_THRESHOLD  → HIGH salience    (full pass to MEMEX-1 Tier 1)
        surprise >= MID_THRESHOLD   → MEDIUM salience  (compressed pass)
        surprise <  MID_THRESHOLD   → SUPPRESSED       (dropped or logged only)
    """

    # Thresholds tuned against cosine distance in [0, 2]
    # These mirror the distance gating in RetrievalArbiter (threshold 1.2 = low similarity)
    HIGH_THRESHOLD: float = 0.6    # Clearly novel — promote fully
    MID_THRESHOLD: float = 0.25    # Somewhat novel — compress and pass
    # Below MID_THRESHOLD = already known, suppress

    def __init__(
        self,
        high_threshold: float = HIGH_THRESHOLD,
        mid_threshold: float = MID_THRESHOLD,
    ):
        self.high_threshold = high_threshold
        self.mid_threshold = mid_threshold

    def _compress(self, fused_text: str) -> str:
        """
        Light compression for MEDIUM salience percepts.
        Truncates to first 300 characters to reduce token footprint
        before feeding to Tier 1. Production version would use
        the Tier 2 compactor pattern.
        """
        if len(fused_text) <= 300:
            return fused_text
        return fused_text[:300] + "... [attenuated]"

    def filter(
        self,
        percept: PerceptObject,
        surprise_score: float,
        prior_summary: str,
        prior_diff: str,
    ) -> SalientPerceptObject:
        """
        Applies the salience gate and returns a SalientPerceptObject
        ready for handoff to MEMEX-1 Tier 1.
        """
        if surprise_score >= self.high_threshold:
            salience = "HIGH"
            print(f"  🔥 [ATTENTION] HIGH salience — fully promoted to memory")

        elif surprise_score >= self.mid_threshold:
            salience = "MEDIUM"
            # Attenuate: compress fused text before passing through
            compressed = self._compress(percept.fused_text)
            percept = percept.model_copy(update={"fused_text": compressed})
            print(f"  🟡 [ATTENTION] MEDIUM salience — attenuated and passed")

        else:
            salience = "SUPPRESSED"
            print(f"  ⬇️  [ATTENTION] SUPPRESSED — below novelty threshold, discarded")

        return SalientPerceptObject(
            percept=percept,
            surprise_score=surprise_score,
            salience=salience,
            prior_summary=prior_summary,
            prior_diff=prior_diff,
        )
