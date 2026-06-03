import os
from memory_tier1 import Tier1WorkingMemory
from retrieval_arbiter import RetrievalArbiter
from openai import OpenAI

def main():
    print("====================================================")
    print("RUNNING DAY 4: ARBITER CONTEXT COMPILED AGENT QUERY")
    print("====================================================\n")

    # 1. Spin up active workspace components
    # We pass the same session ID used previously to inspect your current state
    tier1 = Tier1WorkingMemory(session_id="session_stress_999", max_token_limit=150)
    arbiter = RetrievalArbiter()
    ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key"))

    # Seed the active conversation with a quick turn (this stays in Tier 1)
    tier1.add_message("user", "Hey! Can you write an implementation draft for the next module?")
    tier1.add_message("assistant", "Sure! What are the exact system configuration constraints and technology selections we are building for?")

    # 2. The critical prompt where the user tests the agent's memory
    test_query = "Remind me what language and vector database we chose for MEMEX-1, and why we selected it."

    # 3. Let the Arbiter compile the multi-tier context payload
    compiled_context = arbiter.compile_context(test_query, tier1)
    
    print("\n--- [STAGE 1] ARBITER COMPILED CONTEXT INJECTION BLOCK ---")
    print(compiled_context)

    # 4. Pass the compiled payload directly to the LLM agent
    print("\n--- [STAGE 2] EXECUTING AGENT INFERENCE WITH MEMORY ---")
    try:
        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant. Rely completely on the provided MEMORY INJECTION ENGINE parameters to answer the user accurately."},
                {"role": "user", "content": compiled_context}
            ],
            temperature=0.2
        )
        print(f"\n[AGENT RESPONSE]:\n{response.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"⚠️ Inference step failed: {e}")

if __name__ == "__main__":
    main()