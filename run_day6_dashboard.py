from memory_tier1 import Tier1WorkingMemory
from retrieval_arbiter import RetrievalArbiter
from telemetry import MemexTelemetry

def main():
    print("====================================================")
    print("RUNNING HIGH-LOAD PRODUCTION SCALE TELEMETRY")
    print("====================================================\n")

    # A massive historical log stream representing a long, multi-turn technical development session
    production_log_stream = [
        ("user", "Let's capture our blueprint. We are creating an agentic system named MEMEX-1 running on Python."),
        ("assistant", "Understood. Tracking configuration details for MEMEX-1."),
        ("user", "We selected ChromaDB for indexing because local binary execution reduces latency parameters."),
        ("assistant", "ChromaDB selection recorded. This will serve as our high-performance volatile episodic memory layer, running locally without external network round-trip overhead."),
        ("user", "Note my profile requirement: I maintain a strict vegetarian lifestyle. No eggs, chicken, or seafood."),
        ("assistant", "Dietary restrictions updated. Filters applied to all food applications to ensure strict compliance with vegetarian constraints."),
        ("user", "I prefer making a quick eggless lasagna layered with mozzarella and marinara sauce."),
        ("assistant", "Recipe preference noted. Saving standard procedural instructions for lasagna formulations excluding egg binders."),
        ("user", "Let's expand the project framework to include an asynchronous queue manager in the future."),
        ("assistant", "Queue system mapped to future roadmap cycles."),
        # --- SCALING TRAFFIC EXTRA PACKETS ---
        ("user", "Let's append an extensive analysis of our system dependencies. We must ensure that our asynchronous loops handle concurrency perfectly under extreme I/O loads without blocking the runtime engine loop."),
        ("assistant", "Acknowledged. Concurrency profiling will be established using asyncio tasks, utilizing background worker pools to execute resource-heavy operations asynchronously while maintaining the core microservice architecture."),
        ("user", "Additionally, let's document our deployment topology. The entire MEMEX-1 daemon stack will be containerized using micro-containers and managed via localized composition configurations to allow rapid multi-node scaling across local environments."),
        ("assistant", "Topology map initialized. Local compositions will expose distinct networking channels to seamlessly pipe system telemetry out to our monitoring dashboards without interrupting active query flows.")
    ]

    # Initialize standard components with a tight limit to force massive evictions
    tier1 = Tier1WorkingMemory(session_id="telemetry_test_999", max_token_limit=30)
    arbiter = RetrievalArbiter()
    auditor = MemexTelemetry()

    # Ingest historical data streams through active memory channels
    for role, content in production_log_stream:
        tier1.add_message(role, content)

    # Compile context for an active engineering prompt
    active_query = "What database structures are we using for MEMEX-1?"
    compiled_context = arbiter.compile_context(active_query, tier1)

    # Generate telemetry analytics dashboard
    auditor.generate_report(production_log_stream, compiled_context)

if __name__ == "__main__":
    main()