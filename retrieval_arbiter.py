import os
import json
from typing import List, Dict, Any
from schemas import Message
from memory_tier1 import Tier1WorkingMemory
from memory_tier2 import Tier2EpisodicMemory

class RetrievalArbiter:
    def __init__(self, profile_path: str = "./semantic_profile.json", persist_directory: str = "./.memex_storage"):
        self.profile_path = profile_path
        self.tier2_engine = Tier2EpisodicMemory(persist_directory=persist_directory)

    def _read_tier3_profile(self) -> Dict[str, Any]:
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                return json.load(f)
        return {}

    def compile_context(self, current_query: str, tier1_memory: Tier1WorkingMemory) -> str:
        """
        Gathers shards from all 3 memory tiers and applies distance gating to block semantic drift.
        """
        print(f"\n⚡ [ARBITER] Compiling context for query: '{current_query}'")
        
        # 1. Fetch Volatile L1 History
        active_history = tier1_memory.get_active_context()
        formatted_tier1 = "\n".join([f"{m.role.upper()}: {m.content}" for m in active_history])
        
        # 2. Fetch Relevant Tier 2 Episodes with SIMILARITY DISTANCE GATING
        # We access the raw collection query to extract internal distance lists
        query_results = self.tier2_engine.collection.query(
            query_texts=[current_query],
            n_results=2
        )
        
        valid_episodes = []
        if query_results['documents'] and query_results['distances']:
            documents = query_results['documents'][0]
            distances = query_results['distances'][0]
            
            # Gating Threshold: Closer to 0 means higher similarity. Drop anything > 1.2
            for doc, distance in zip(documents, distances):
                if distance <= 1.2:
                    valid_episodes.append(doc)
                else:
                    print(f"🛑 [ARBITER] Blocked low-confidence episodic leak! Distance: {distance:.4f} (Threshold: 1.2)")

        formatted_tier2 = "\n".join([f"- {ep}" for ep in valid_episodes])
        
        # 3. Fetch Tier 3 Global States
        global_profile = self._read_tier3_profile()
        pruned_profile = {k: v for k, v in global_profile.items() if v}
        formatted_tier3 = json.dumps(pruned_profile, indent=2)

        # 4. Synthesize Context Block
        context_payload = f"""=== MEMORY INJECTION ENGINE (MEMEX-1) ===

[TIER 3: GLOBAL PARAMETERS & PREFERENCES]
{formatted_tier3 if pruned_profile else "No global facts synchronized yet."}

[TIER 2: HISTORICAL EPISODES RETRIEVED]
{formatted_tier2 if valid_episodes else "No highly relevant historical contexts found for this domain."}

[TIER 1: VOLATILE ACTIVE CONVERSATION SESSION]
{formatted_tier1 if formatted_tier1 else "Session initiated."}
==========================================
USER QUERY: {current_query}
"""
        return context_payload