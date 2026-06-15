"""
load_monitor.py

Cognitive Load Theory (Sweller, 1988) operationalized.

Three load types:
    Intrinsic   -- complexity inherent to the input itself
                   (token count, modality count, surprise score)
    Extraneous  -- noise and redundancy in the input
                   (low-salience percepts, suppressed signals still passing)
    Germane     -- schema-building demand
                   (how much new structure is being written to Tier 3)

The LoadMonitor computes all three, produces a composite LoadState,
and emits adjusted thresholds for PERCEPT-1 and MEMEX-1 to consume.
"""

from typing import Optional
import tiktoken
from monitor_schemas import LoadState


# Default Tier 1 token limits per load level
TOKEN_LIMITS = {
    "LOW":      2000,
    "MODERATE": 1500,
    "HIGH":     1000,
    "CRITICAL": 600,
}

# Adjusted PERCEPT-1 HIGH salience thresholds per load level
# Under high load, raise the bar for what gets promoted to memory
SURPRISE_THRESHOLDS = {
    "LOW":      0.6,
    "MODERATE": 0.65,
    "HIGH":     0.75,
    "CRITICAL": 0.85,
}


class LoadMonitor:
    """
    Computes cognitive load from live stack signals and
    emits a LoadState with recommended threshold adjustments.
    """

    def __init__(self, model_encoding: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(model_encoding)

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _compute_intrinsic(
        self,
        input_text: str,
        modality_count: int,
        surprise_score: float,
    ) -> float:
        """
        Intrinsic load = f(token complexity, modality count, surprise).
        Normalized to [0, 1].
        """
        token_score = min(self._token_count(input_text) / 1000, 1.0)
        modality_score = min((modality_count - 1) / 2, 1.0)  # 1 modal=0, 3 modal=1
        surprise_norm = min(surprise_score / 2.0, 1.0)

        # Weighted: tokens 40%, modalities 30%, surprise 30%
        return round(0.4 * token_score + 0.3 * modality_score + 0.3 * surprise_norm, 4)

    def _compute_extraneous(
        self,
        suppressed_count: int,
        total_percept_count: int,
    ) -> float:
        """
        Extraneous load = ratio of suppressed/low-salience signals
        that still consumed processing cycles.
        """
        if total_percept_count == 0:
            return 0.0
        return round(min(suppressed_count / total_percept_count, 1.0), 4)

    def _compute_germane(
        self,
        tier3_update_count: int,
    ) -> float:
        """
        Germane load = schema-building demand.
        Proxied by how many Tier 3 profile updates have fired recently.
        Caps at 5 updates = max germane load.
        """
        return round(min(tier3_update_count / 5, 1.0), 4)

    def _classify(self, total: float) -> str:
        if total < 0.3:
            return "LOW"
        elif total < 0.55:
            return "MODERATE"
        elif total < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"

    def compute(
        self,
        input_text: str,
        modality_count: int = 1,
        surprise_score: float = 0.5,
        suppressed_percept_count: int = 0,
        total_percept_count: int = 1,
        tier3_update_count: int = 0,
    ) -> LoadState:
        """
        Main entry point. Returns a LoadState with recommended
        threshold adjustments for the rest of the stack.
        """
        intrinsic  = self._compute_intrinsic(input_text, modality_count, surprise_score)
        extraneous = self._compute_extraneous(suppressed_percept_count, total_percept_count)
        germane    = self._compute_germane(tier3_update_count)

        # Weighted composite: intrinsic 50%, extraneous 30%, germane 20%
        total = round(0.5 * intrinsic + 0.3 * extraneous + 0.2 * germane, 4)
        level = self._classify(total)

        print(f"  ⚖️  [LOAD MONITOR] intrinsic={intrinsic:.3f} | extraneous={extraneous:.3f} | germane={germane:.3f} | total={total:.3f} → {level}")

        return LoadState(
            intrinsic=intrinsic,
            extraneous=extraneous,
            germane=germane,
            total=total,
            level=level,
            recommended_token_limit=TOKEN_LIMITS[level],
            recommended_surprise_threshold=SURPRISE_THRESHOLDS[level],
        )
