from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime


class RawPerceptInput(BaseModel):
    """Raw multimodal input before processing."""
    text: Optional[str] = Field(default=None, description="Raw text input")
    structured: Optional[Dict[str, Any]] = Field(default=None, description="JSON/dict structured data")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image")
    image_media_type: Optional[str] = Field(default="image/jpeg", description="MIME type of image")
    source_label: Optional[str] = Field(default="external", description="Semantic label for input origin")


class PerceptObject(BaseModel):
    """Fused multimodal representation after binding."""
    fused_text: str = Field(..., description="Unified text representation of all modalities")
    modalities_present: List[Literal["text", "structured", "image"]] = Field(default_factory=list)
    raw_input: RawPerceptInput
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SalientPerceptObject(BaseModel):
    """
    Output of PERCEPT-1. Carries the fused perception with
    surprise score, salience classification, and a prior diff
    showing what was expected vs. what was actually observed.
    """
    percept: PerceptObject
    surprise_score: float = Field(..., description="Cosine distance from prior expectation [0, 2]")
    salience: Literal["HIGH", "MEDIUM", "SUPPRESSED"] = Field(..., description="Attention routing decision")
    prior_summary: str = Field(..., description="What MEMEX-1 Tier 3 expected based on global profile")
    prior_diff: str = Field(..., description="Natural language description of what diverged from prior")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_message_content(self) -> str:
        """
        Serializes to a string suitable for Tier1WorkingMemory.add_message().
        Includes salience tag so MEMEX-1 can weight this perception appropriately.
        """
        return (
            f"[PERCEPT | salience={self.salience} | surprise={self.surprise_score:.4f}]\n"
            f"{self.percept.fused_text}\n"
            f"[PRIOR_DIFF] {self.prior_diff}"
        )
