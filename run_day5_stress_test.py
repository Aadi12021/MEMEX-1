import os
import json
from memory_tier1 import Tier1WorkingMemory
from memory_tier2 import Tier2EpisodicMemory
from memory_tier3 import Tier3SemanticMemory
from retrieval_arbiter import RetrievalArbiter

def print_divider(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def main():
    # Initialize all active system nodes
    # We use a brand new session to track clean state mutations
    session_id = "session_stress_v5"
    tier1 = Tier1WorkingMemory(session_id=session_id, max_token_limit=100)
    tier2 = Tier2EpisodicMemory()
    tier3 = Tier3SemanticMemory(profile_path="./semantic_profile_v5.json")
    arbiter = RetrievalArbiter(profile_path="./semantic_profile_v5.json")

    # Clear old files if they exist to keep validation pristine
    if os.path.exists("./semantic_profile_v5.json"):
        os.remove("./semantic_profile_v5.json")
    tier3._write_profile({"user_profile": {}, "project_metadata": {}, "global_facts": []})

    # A chaotic conversational flow designed to test extreme context transitions
    stress_stream = [
        # Phrase Phase 1: Software Architecture
        ("user", "Let's capture our blueprint. We are creating an agentic system named MEMEX-1 running on Python."),
        ("user", "We selected ChromaDB for indexing because local binary execution reduces latency parameters."),
        
        # Phase 2: Complete Topic Divergence (Dietary Preferences)
        ("user", "Let's change focus. Note my profile requirement: I maintain a strict vegetarian lifestyle. No eggs, chicken, or seafood."),
        ("user", "Can you save my favorite recipe layout? I prefer making a quick eggless lasagna layered with mozzarella and marinara sauce."),
        
        # Phase 3: The Multi-Tier Push Trigger
        ("user", "Let's run a batch sweep to force the background engine to process everything we've talked about so far.")
    ]

    print_divider("STAGE 1: INGESTING HIGH-ENTROPY INTERACTION STREAM")
    
    for idx, (role, content) in enumerate(stress_stream):
        print(f"\n[Turn {idx+1}] Ingesting User Input...")
        active, evicted = tier1.add_message(role, content)
        if evicted:
            tier2.process_eviction(session_id=session_id, evicted_messages=evicted)

    print_divider("STAGE 2: FORCE BACKGROUND SEMANTIC BATCH RECONCILIATION")
    tier3.run_semantic_compaction()

    print_divider("STAGE 3: VERIFYING TARGETED RETRIEVAL & ANTI-DRIFT")
    
    # Target Query A: Should pull ONLY food profile data
    query_a = "What are my specific dietary requirements and favorite Italian meals?"
    context_a = arbiter.compile_context(query_a, tier1)
    
    print("\n--- [ARBITER COMPILATION FOR DIET QUERY] ---")
    print(context_a)
    
    # CRITICAL EVALUATION ASSERTION:
    # Check if the food query accidentally leaked software context into Tier 2 historical fragments
    assert "MEMEX-1" not in context_a.split("[TIER 2: HISTORICAL EPISODES RETRIEVED]")[1].split("[TIER 1:")[0], \
        "❌ EDGE CASE FAILURE: Tier 2 retrieval leaked irrelevant engineering metadata into a culinary request!"
    print("✅ PASS: Semantic boundary held. No engineering leak detected in culinary vector retrieval.")

    # Target Query B: Should pull ONLY tech architecture data
    query_b = "Remind me of the structural vector parameters we decided on for our project."
    context_b = arbiter.compile_context(query_b, tier1)
    
    print("\n--- [ARBITER COMPILATION FOR TECH QUERY] ---")
    print(context_b)
    
    assert "lasagna" not in context_b.split("[TIER 2: HISTORICAL EPISODES RETRIEVED]")[1].split("[TIER 1:")[0], \
        "❌ EDGE CASE FAILURE: Tier 2 retrieval leaked culinary preferences into an engineering context request!"
    print("✅ PASS: Semantic boundary held. No culinary leak detected in technology vector retrieval.")

if __name__ == "__main__":
    main()