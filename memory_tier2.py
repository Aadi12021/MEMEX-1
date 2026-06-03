import os
from typing import List
from datetime import datetime
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from schemas import Message, EvictionPayload

class Tier2EpisodicMemory:
    def __init__(self, collection_name: str = "memex_episodes", persist_directory: str = "./.memex_storage"):
        """
        Tier 2 Vector Storage layer using ChromaDB.
        """
        # Initialize local persistent storage client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create the vector collection
        # Note: We'll let Chroma handle embeddings natively using default models, 
        # or we can explicitly pass custom embedding vectors using OpenAI.
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # Initialize OpenAI Client (Ensure OPENAI_API_KEY is set in your environment variables)
        self.ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key-if-using-local-ollama"))

    def _compress_evicted_logs(self, messages: List[Message]) -> str:
        """
        Uses a fast model to synthesize raw dialogue logs into an episodic summary snapshot.
        """
        # Format the incoming messages into a single text block
        formatted_dialogue = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in messages])
        
        prompt = f"""
        You are the compaction sub-process of an advanced AI memory engine. 
        Your task is to take the following segment of raw conversation and distill it into a dense, objective summary snapshot.
        Focus heavily on extracting key technical decisions made, core questions asked, concepts explained, and explicit user preferences.
        Do not use fluff or conversational transitions.

        [RAW LOGS START]
        {formatted_dialogue}
        [RAW LOGS END]

        Output the summary as a cohesive paragraph of dense facts.
        """
        
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini", # Using a highly cost-efficient, high-velocity model for system subroutines
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback mitigation if API fails or isn't configured yet
            print(f"⚠️ Compactor API error: {e}. Falling back to basic concatenation.")
            return f"Evicted fragment fallback summary: {messages[-1].content[:200]}"

    def process_eviction(self, session_id: str, evicted_messages: List[Message]):
        """
        The downstream receiver endpoint. Compresses the logs, generates metadata, and updates the vector space.
        """
        if not evicted_messages:
            return

        print(f"\n⚡ [TIER 2] Compressing {len(evicted_messages)} evicted messages...")
        compressed_summary = self._compress_evicted_logs(evicted_messages)
        print(f"📄 [TIER 2] Generated Episode Summary:\n   \"{compressed_summary}\"")

        # Generate unique memory signature
        timestamp_str = datetime.utcnow().isoformat()
        memory_id = f"epi_{session_id}_{int(datetime.utcnow().timestamp())}"

        # Write data to Chroma
        # Chroma will auto-generate an embedding for the document text if no explicit vector is passed.
        self.collection.add(
            documents=[compressed_summary],
            metadatas=[{
                "session_id": session_id,
                "timestamp": timestamp_str,
                "msg_count": len(evicted_messages)
            }],
            ids=[memory_id]
        )
        print(f"💾 [TIER 2] Episode successfully indexed into vector space. ID: {memory_id}")

    def query_episodes(self, query_text: str, n_results: int = 2) -> List[str]:
        """
        Performs a semantic similarity lookup over stored long-term episodic summaries.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        # Flatten documents structure returned by Chroma
        return results['documents'][0] if results['documents'] else []