import os
from memory_tier1 import Tier1WorkingMemory
from memory_tier2 import Tier2EpisodicMemory
from memory_tier3 import Tier3SemanticMemory

def main():
    print("====================================================")
    print("RUNNING DAY 3: MULTI-TIER E2E INTERACTION PIPELINE")
    print("====================================================\n")
    
    # Initialize all three layers of MEMEX-1
    tier1 = Tier1WorkingMemory(session_id="session_stress_999", max_token_limit=100)
    tier2 = Tier2EpisodicMemory()
    tier3 = Tier3SemanticMemory()

    # Conversation stream designed to build two distinct memory clusters
    stream = [
        # Topic Cluster A: Software Architecture
        ("user", "Let's log our core architecture stack. We are developing an autonomous system named MEMEX-1 using Python and asynchronous microservices."),
        ("assistant", "Understood. Tracking project configuration: System title MEMEX-1, core backend engine leveraging Python patterns."),
        ("user", "We opted for ChromaDB for structural vector indexing because it gives us rapid local deployment capabilities without heavy cloud overhead."),
        
        # Topic Cluster B: Diet & Culinary Parameters
        ("user", "Switching subjects entirely. Please note my absolute dietary requirements for future food applications: I maintain a strict vegetarian profile. I never eat eggs, poultry, or fish."),
        ("assistant", "Profile parameters updated. Systems rules configured to filter out all meat products, poultry, seafood, and egg variants."),
        ("user", "Can you give me ideas for a fast dinner? Maybe a quick eggless vegetarian lasagna or a hot noodle soup with plant substitutions?")
    ]

    print("--- [STAGE 1] INGESTING CONVERSATION TRACKER ---")
    for role, content in stream:
        active, evicted = tier1.add_message(role, content)
        if evicted:
            tier2.process_eviction(session_id=tier1.session_id, evicted_messages=evicted)

    print("\n--- [STAGE 2] RUNNING TIER 3 BACKGROUND DEAMON ---")
    tier3.run_semantic_compaction()

    # Read back out the file to see the final results
    if os.path.exists("./semantic_profile.json"):
        print("\n====================================================")
        print("GENERATED SEMANTIC_PROFILE.JSON LAYER FROM DISK")
        print("====================================================")
        with open("./semantic_profile.json", "r") as f:
            print(f.read())

if __name__ == "__main__":
    main()