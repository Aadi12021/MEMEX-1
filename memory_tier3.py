import os
import json
from typing import List, Dict, Any
import chromadb
from openai import OpenAI

class Tier3SemanticMemory:
    def __init__(self, profile_path: str = "./semantic_profile.json", persist_directory: str = "./.memex_storage"):
        self.profile_path = profile_path
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="memex_episodes")
        self.ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key"))
        
        # Initialize an empty profile structure if it doesn't exist yet
        if not os.path.exists(self.profile_path):
            self._write_profile({"user_profile": {}, "project_metadata": {}, "global_facts": []})

    def _read_profile(self) -> Dict[str, Any]:
        with open(self.profile_path, "r") as f:
            return json.load(f)

    def _write_profile(self, data: Dict[str, Any]):
        with open(self.profile_path, "w") as f:
            json.dump(data, f, indent=2)

    def _cluster_episodes_via_llm(self, documents: List[str]) -> Dict[str, List[str]]:
        """
        Replaces K-Means with a fast, deterministic semantic sorting pass.
        Groups documents into logical conceptual themes natively.
        """
        formatted_docs = "\n".join([f"[{idx}]: {doc}" for idx, doc in enumerate(documents)])
        
        prompt = f"""
        You are a data-sorting subroutine. Group the following episodic memories into distinct conceptual themes (maximum 3 themes).
        
        [EPISODES]
        {formatted_docs}
        
        Return your response strictly as a JSON object mapping category names to lists of item indexes.
        Example: {{"Software Architecture": [0, 1], "Dietary Preferences": [2]}}
        Do not include markdown codeblocks or extra text.
        """
        
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_output = response.choices[0].message.content.strip()
            if raw_output.startswith("```"):
                raw_output = raw_output.strip("```json").strip("```").strip()
                
            group_mapping = json.loads(raw_output)
            
            # Reconstruct the grouped text categories
            clustered_groups = {}
            for category, indices in group_mapping.items():
                clustered_groups[category] = [documents[int(idx)] for idx in indices if int(idx) < len(documents)]
            return clustered_groups
            
        except Exception as e:
            print(f"⚠️ [TIER 3] LLM Sorting pass failed: {e}. Falling back to single-bucket consolidation.")
            return {"Global Consolidated Context": documents}

    def run_semantic_compaction(self):
        """
        Extracts episodic memories, groups them semantically,
        synthesizes high-level schemas, and updates the profile on disk.
        """
        # Pull documents from Chroma
        results = self.collection.get(include=["documents"])
        documents = results.get("documents", [])
        
        if len(documents) < 2:
            print(f"ℹ️ [TIER 3] Only {len(documents)} episodes found in Chroma. Skipping compaction until history fills up.")
            return

        print(f"\n🔮 [TIER 3] Initiating Semantic Consolidation on {len(documents)} items...")
        
        # Group related items using our clean sorting pass
        clustered_groups = self._cluster_episodes_via_llm(documents)
        current_profile = self._read_profile()
        
        # Consolidate each group into our central profile
        for category, texts in clustered_groups.items():
            if not texts:
                continue
            print(f"🧠 [TIER 3] Distilling insights from Semantic Category: '{category}'...")
            cluster_corpus = "\n".join([f"- {t}" for t in texts])
            
            prompt = f"""
            You are the high-level Tier 3 Memory Compactor subroutine of an agentic state manager.
            Your role is to analyze a group of related interaction summaries and extract permanent, abstract, global rules, user properties, or project configurations.

            [CURRENT PROFILE STATE]
            {json.dumps(current_profile, indent=2)}

            [NEW RELATED INSIGHTS FOR CATEGORY: {category}]
            {cluster_corpus}

            Reconcile these updates. Inject new fields or update existing properties inside the JSON layout.
            Return ONLY the valid, raw updated JSON object string. Do not include markdown codeblocks (like ```json) or any meta prose.
            """
            
            try:
                response = self.ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                raw_output = response.choices[0].message.content.strip()
                if raw_output.startswith("```"):
                    raw_output = raw_output.strip("```json").strip("```").strip()
                    
                current_profile = json.loads(raw_output)
            except Exception as e:
                print(f"⚠️ [TIER 3] Processing engine encountered a parsing error: {e}")
                
        # Save structural JSON back to disk
        self._write_profile(current_profile)
        print("💾 [TIER 3] Global semantic profile successfully reconciled and saved to disk.")