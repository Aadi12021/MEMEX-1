from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class LoadState(BaseModel):
    """
    Output of the Load Monitor.
    Captures all three Sweller cognitive load types
    and produces a unified load level for the daemon to act on.
    """
    intrinsic: float = Field(..., description="Input complexity score [0, 1]")
    extraneous: float = Field(..., description="Noise / redundancy score [0, 1]")
    germane: float = Field(..., description="Schema-building demand score [0, 1]")
    total: float = Field(..., description="Weighted composite load score [0, 1]")
    level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] = Field(
        ..., description="Bucketed load classification"
    )
    recommended_token_limit: int = Field(
        ..., description="Dynamically adjusted Tier 1 token limit based on load"
    )
    recommended_surprise_threshold: float = Field(
        ..., description="Adjusted PERCEPT-1 HIGH salience threshold based on load"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetaScore(BaseModel):
    """
    Output of the Meta-Evaluator.
    Scores a response on confidence, completeness,
    and consistency with the Tier 3 knowledge profile.
    """
    confidence: float = Field(..., description="How confident the response is [0, 1]")
    completeness: float = Field(..., description="How fully the query was addressed [0, 1]")
    consistency: float = Field(..., description="Alignment with Tier 3 profile [0, 1]")
    composite: float = Field(..., description="Weighted composite meta-score [0, 1]")
    verdict: Literal["PASS", "REVISE", "ESCALATE"] = Field(
        ..., description="Action flag for the daemon"
    )
    reasoning: str = Field(..., description="Natural language explanation of the verdict")
    revised_response: Optional[str] = Field(
        default=None, description="Revised response if verdict is REVISE"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MonitorReport(BaseModel):
    """
    Full output of one Self-Monitor Daemon cycle.
    Bundles load state, meta-score, and the final response
    into one structured report.
    """
    query: str
    raw_response: str
    final_response: str
    load_state: LoadState
    meta_score: MetaScore
    thresholds_adjusted: bool = Field(
        default=False, description="Whether PERCEPT-1/MEMEX-1 thresholds were modified this cycle"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        return (
            f"[MONITOR | load={self.load_state.level} | "
            f"verdict={self.meta_score.verdict} | "
            f"confidence={self.meta_score.confidence:.2f} | "
            f"composite_meta={self.meta_score.composite:.2f}]"
        )
