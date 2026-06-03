import os
from memory_tier1 import Tier1WorkingMemory
from memory_tier2 import Tier2EpisodicMemory

def run_e2e_simulation():
    print("====================================================")
    print("TESTING COUPLING: TIER 1 CACHE -> TIER 2 VECTOR SPACE")
    print("====================================================\n")
    
    # Force a microscopic threshold to guarantee multiple evictions during testing
    # Change max_token_limit from 150 to 100
    tier1 = Tier1WorkingMemory(session_id="session_dev_002", max_token_limit=100)
    tier2 = Tier2EpisodicMemory()

    conversation = [
        ("user", "Let's track a project. This project is named MEMEX-1 and it's a multi-tier memory daemon written in Python."),
        ("assistant", "Understood. MEMEX-1 is an AI memory microservice built using Python, focusing on optimizing token usage via a tiered architecture."),
        ("user", "We are using ChromaDB for the Tier 2 memory layer because it's lightweight and runs locally."),
        ("assistant", "Excellent architecture selection. ChromaDB eliminates heavy system dependencies while maintaining vector performance."),
        ("user", "Quick switch of topics: What is a cool recipe for an easy vegetarian dinner? Maybe a quick lasagna without eggs?")
    ]

    for turn_idx, (role, content) in enumerate(conversation):
        print(f"\n--- Turn {turn_idx + 1} ---")
        active_buffer, evicted = tier1.add_message(role, content)
        
        print(f"L1 Status: {tier1.current_token_count}/{tier1.max_token_limit} tokens")
        
        if evicted:
            print(f"⚠️ [EVICTION BOUNDARY BREACHED] Forwarding payload to Tier 2 Pipeline...")
            tier2.process_eviction(session_id=tier1.session_id, evicted_messages=evicted)

    # Validate that we can query our vector RAM space for past facts that were evicted
    print("\n====================================================")
    print("VERIFYING SEMANTIC RETRIEVAL FROM TIER 2")
    print("====================================================")
    
    search_query = "What framework or database are we using for our secondary memory tier?"
    print(f"\nQuerying Tier 2 space with: '{search_query}'")
    
    retrieved_memories = tier2.query_episodes(search_query, n_results=1)
    
    for idx, memory in enumerate(retrieved_memories):
        print(f"\n[HIT #{idx + 1} RETRIEVED EPISODE]:\n{memory}")

if __name__ == "__main__":
    # Ensure you export your OPENAI_API_KEY in your terminal before execution
    run_e2e_simulation()