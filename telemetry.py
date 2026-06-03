import tiktoken
from typing import List
from schemas import Message

class MemexTelemetry:
    def __init__(self, model_encoding: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(model_encoding)

    def _count_string_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def generate_report(self, full_historical_stream: List[tuple], compiled_context: str):
        """
        Computes system token allocation and prints a high-signal runtime efficiency dashboard.
        """
        # 1. Calculate what the unmanaged raw token size would be if we just stacked messages
        raw_text = "\n".join([f"{role.upper()}: {content}" for role, content in full_historical_stream])
        total_raw_tokens = self._count_string_tokens(raw_text)
        
        # 2. Calculate the size of the Arbiter's optimized context block
        optimized_tokens = self._count_string_tokens(compiled_context)
        
        # 3. Compute efficiency metrics
        tokens_saved = max(0, total_raw_tokens - optimized_tokens)
        reduction_percentage = (tokens_saved / total_raw_tokens) * 100 if total_raw_tokens > 0 else 0
        
        # Estimated financial savings factor per 1M tokens (based on standard input cache rates)
        financial_factor = (tokens_saved / 1_000_000) * 2.50 # Assume $2.50 per million input tokens

        print("\n" + "="*50)
        print(" 📊 MEMEX-1 SYSTEM TELEMETRY & PERFORMANCE METRICS")
        print("="*50)
        print(f" Raw Conversation Footprint : {total_raw_tokens:,} tokens")
        print(f" MEMEX-1 Injected Footprint  : {optimized_tokens:,} tokens")
        print(f" Hard Token Overlap Saved    : {tokens_saved:,} tokens")
        print(f" Context Window Compression  : {reduction_percentage:.2f}%")
        print(f" Simulated Cost Reduction    : ${financial_factor:.6f} per iteration")
        print("="*50 + "\n")