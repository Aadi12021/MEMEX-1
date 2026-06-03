from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Message(BaseModel):
    role: str = Field(..., description="Either 'system', 'user', or 'assistant'")
    content: str = Field(..., description="The raw text content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: Optional[int] = Field(default=None, description="Pre-computed token length")

class SessionState(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvictionPayload(BaseModel):
    session_id: str
    evicted_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = Field(..., description="The sequence of raw messages being pushed out of Tier 1")