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

Improvement over v1:
    - Rolling window averaging (default N=5 cycles) smooths load spikes.
      A single complex input no longer flips the system to CRITICAL.
      Load builds and dissipates gradually, mirroring how cognitive fatigue
      actually accumulates in the brain.
    - Response latency tracking as an additional intrinsic load signal.
      If the system is taking longer to respond, that is a real indicator
      of processing burden independent of input complexity.

The LoadMonitor computes all three, produces a composite LoadState,
and emits adjusted thresholds for PERCEPT-1 and MEMEX-1 to consume.
"""

import time
from collections import deque
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

# Latency above this (seconds) contributes to intrinsic load
LATENCY_BASELINE_S = 1.0
LATENCY_MAX_S      = 10.0


class LoadMonitor:
    """
    Computes cognitive load from live stack signals and
    emits a LoadState with recommended threshold adjustments.

    Uses a rolling window of past N cycles to smooth load scores --
    prevents single spiky inputs from triggering aggressive threshold changes.
    """

    def __init__(
        self,
        model_encoding: str = "cl100k_base",
        window_size: int = 5,
    ):
        self.encoder = tiktoken.get_encoding(model_encoding)
        self.window_size = window_size

        # Rolling window of past composite load scores
        self._load_history: deque[float] = deque(maxlen=window_size)

        # Latency tracking: caller sets _cycle_start before compute()
        self._cycle_start: Optional[float] = None

    def start_cycle(self):
        """Call this at the start of a daemon cycle to begin latency tracking."""
        self._cycle_start = time.monotonic()

    def _get_latency(self) -> float:
        """Returns elapsed seconds since start_cycle(), or 0 if not set."""
        if self._cycle_start is None:
            return 0.0
        return time.monotonic() - self._cycle_start

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _compute_intrinsic(
        self,
        input_text: str,
        modality_count: int,
        surprise_score: float,
        latency_s: float,
    ) -> float:
        """
        Intrinsic load = f(token complexity, modality count, surprise, latency).
        Normalized to [0, 1].

        Latency contribution: scales linearly from LATENCY_BASELINE_S (no load)
        to LATENCY_MAX_S (max load). Reflects that slow responses signal burden.
        """
        token_score   = min(self._token_count(input_text) / 1000, 1.0)
        modality_score = min((modality_count - 1) / 2, 1.0)  # 1 modal=0, 3 modal=1
        surprise_norm = min(surprise_score / 2.0, 1.0)
        latency_norm  = min(
            max(latency_s - LATENCY_BASELINE_S, 0) / (LATENCY_MAX_S - LATENCY_BASELINE_S),
            1.0,
        )

        # Weighted: tokens 35%, modalities 25%, surprise 25%, latency 15%
        return round(
            0.35 * token_score
            + 0.25 * modality_score
            + 0.25 * surprise_norm
            + 0.15 * latency_norm,
            4,
        )

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

    def _rolling_average(self, current_total: float) -> float:
        """
        Appends current_total to the rolling window and returns
        the smoothed average across the last N cycles.

        On the first cycle the window has one entry so the raw score is used.
        As cycles accumulate the average stabilizes.
        """
        self._load_history.append(current_total)
        return round(sum(self._load_history) / len(self._load_history), 4)

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

        The composite load score is smoothed over the rolling window
        before classification -- single-cycle spikes are damped.
        Call start_cycle() before perceive() to enable latency tracking.
        """
        latency_s = self._get_latency()

        intrinsic  = self._compute_intrinsic(input_text, modality_count, surprise_score, latency_s)
        extraneous = self._compute_extraneous(suppressed_percept_count, total_percept_count)
        germane    = self._compute_germane(tier3_update_count)

        # Raw composite: intrinsic 50%, extraneous 30%, germane 20%
        raw_total = round(0.5 * intrinsic + 0.3 * extraneous + 0.2 * germane, 4)

        # Smoothed composite via rolling window
        smoothed_total = self._rolling_average(raw_total)
        level = self._classify(smoothed_total)

        print(
            f"  ⚖️  [LOAD MONITOR] intrinsic={intrinsic:.3f} | extraneous={extraneous:.3f} | "
            f"germane={germane:.3f} | latency={latency_s:.2f}s | "
            f"raw={raw_total:.3f} | smoothed={smoothed_total:.3f} (n={len(self._load_history)}) → {level}"
        )

        return LoadState(
            intrinsic=intrinsic,
            extraneous=extraneous,
            germane=germane,
            total=smoothed_total,
            level=level,
            recommended_token_limit=TOKEN_LIMITS[level],
            recommended_surprise_threshold=SURPRISE_THRESHOLDS[level],
        )