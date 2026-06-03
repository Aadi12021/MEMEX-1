from memory_tier1 import Tier1WorkingMemory
import json

def run_simulation():
    # Initialize Tier 1 with an intentionally small token limit to trigger early eviction
    print("Initializing MEMEX-1 Tier 1 Memory Daemon...")
    tier1 = Tier1WorkingMemory(session_id="session_dev_001", max_token_limit=300)
    
    # Simulated high-velocity conversation steps
    conversation_turns = [
        ("user", "Hello! Let's start designing an operating system from scratch."),
        ("assistant", "That sounds excellent. We will need to design the process scheduler, the memory management subsystem, and the virtual file system abstractions first."),
        ("user", "Can you explain how a multi-level feedback queue scheduler works in depth? Give me a massive paragraph detailing the priority transitions and starvation mitigation strategies."),
        ("assistant", "A Multi-Level Feedback Queue (MLFQ) scheduler breaks down execution priorities into separate discrete queues. High priority tasks land in Queue 0. If they consume their entire time slice without yielding, they are demoted to a lower priority queue. To prevent starvation of lower-priority tasks, a global priority boost occurs periodically, moving all tasks back up to the top tier queue."),
        ("user", "Awesome. Now let's talk about virtual memory and page translation lookaside buffers.")
    ]
    
    for turn_idx, (role, content) in enumerate(conversation_turns):
        print(f"\n--- Turn {turn_idx + 1} ---")
        print(f"Adding from [{role}]: '{content[:60]}...'")
        
        active_buffer, evicted = tier1.add_message(role, content)
        
        print(f"Active Buffer Count: {len(active_buffer)} messages | Current Tokens: {tier1.current_token_count}/{tier1.max_token_limit}")
        
        if evicted:
            print(f"⚠️ [EVICTION TRIGGERED] Pushing {len(evicted)} messages out of Tier 1 Cache:")
            for msg in evicted:
                print(f"  -> Evicted [{msg.role}]: {msg.content[:50]}... ({msg.token_count} tokens)")
                
if __name__ == "__main__":
    run_simulation()