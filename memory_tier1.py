import tiktoken
from typing import List, Tuple
from schemas import Message, EvictionPayload

class Tier1WorkingMemory:
    def __init__(self, session_id: str, max_token_limit: int = 2000, model_encoding: str = "cl100k_base"):
        """
        Tier 1 Cache initialized per-session.
        Default encoding 'cl100k_base' is standard for gpt-3.5-turbo / gpt-4.
        """
        self.session_id = session_id
        self.max_token_limit = max_token_limit
        self.encoder = tiktoken.get_encoding(model_encoding)
        self.buffer: List[Message] = []
        self.current_token_count = 0

    def _calculate_tokens(self, text: str) -> int:
        """Helper to get exact token footprint of a string."""
        return len(self.encoder.encode(text))

    def add_message(self, role: str, content: str) -> Tuple[List[Message], List[Message]]:
        """
        Appends a message to the L1 cache buffer.
        Returns a tuple: (Current active buffer state, List of evicted messages if any)
        """
        # Calculate tokens for the incoming message
        msg_tokens = self._calculate_tokens(content) + self._calculate_tokens(role) + 4 # Padding for structural tokens
        
        new_message = Message(
            role=role,
            content=content,
            token_count=msg_tokens
        )
        
        self.buffer.append(new_message)
        self.current_token_count += msg_tokens
        
        eviected_messages = []
        
        # Eviction Engine: Check if we violated our hard memory bounds
        while self.current_token_count > self.max_token_limit and len(self.buffer) > 1:
            # Pop the oldest message (FIFO order)
            evicted = self.buffer.pop(0)
            self.current_token_count -= evicted.token_count
            eviected_messages.append(evicted)
            
        return self.buffer, eviected_messages

    def get_active_context(self) -> List[Message]:
        """Returns the current state of Tier 1 memory."""
        return self.buffer