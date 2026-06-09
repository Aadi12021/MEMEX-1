import json
import os
from typing import Dict, Any

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from percept_schemas import PerceptObject


class PredictiveCoder:
    """
    Stage 2 of PERCEPT-1.

    Implements Karl Friston's predictive coding principle:
    the brain doesn't passively receive input — it constantly generates
    predictions and only processes the *prediction error* (surprise).

    This module:
    1. Reads the Tier 3 global profile from MEMEX-1 as the brain's "prior"
    2. Generates an expected embedding from that prior
    3. Computes surprise = cosine distance between expected and actual
    4. Returns the surprise score + a natural language prior diff
    """

    def __init__(
        self,
        profile_path: str = "./semantic_profile.json",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.profile_path = profile_path
        self.encoder = SentenceTransformer(embedding_model)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _read_tier3_profile(self) -> Dict[str, Any]:
        """Reads the MEMEX-1 Tier 3 semantic profile from disk."""
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                return json.load(f)
        return {}

    def _summarize_prior(self, profile: Dict[str, Any]) -> str:
        """
        Converts the Tier 3 JSON profile into a dense natural language
        summary. This is the 'what the system expects to see' signal.
        """
        if not profile or all(not v for v in profile.values()):
            return "No prior context established. All inputs treated as maximally novel."

        pruned = {k: v for k, v in profile.items() if v}
        return f"Known global context: {json.dumps(pruned, indent=None)}"

    def _cosine_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine distance in [0, 2]. 0 = identical, 2 = opposite."""
        dot = np.dot(vec_a, vec_b)
        norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if norm == 0:
            return 1.0  # Treat zero vectors as maximally uncertain
        cosine_similarity = dot / norm
        return float(1.0 - cosine_similarity)

    def _generate_prior_diff(
        self, prior_summary: str, actual_fused_text: str, surprise_score: float
    ) -> str:
        """
        Uses Claude to articulate *why* the input diverged from the prior.
        This is the prediction error signal in natural language — useful for
        logging, debugging, and feeding back into Tier 3 updates.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You are a prediction-error analyzer in a cognitive architecture.\n\n"
                        f"PRIOR (what was expected):\n{prior_summary}\n\n"
                        f"ACTUAL PERCEPTION:\n{actual_fused_text}\n\n"
                        f"Surprise score: {surprise_score:.4f} (scale 0-2, higher = more novel)\n\n"
                        "In one sentence, describe specifically what diverged from the prior. "
                        "Be concrete. No filler phrases."
                    ),
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def compute_surprise(self, percept: PerceptObject) -> tuple[float, str, str]:
        """
        Main entry point. Returns (surprise_score, prior_summary, prior_diff).
        """
        profile = self._read_tier3_profile()
        prior_summary = self._summarize_prior(profile)

        print(f"  🔮 [PREDICTIVE CODER] Prior loaded: {prior_summary[:80]}...")

        # Embed both prior expectation and actual perception
        prior_vec = self.encoder.encode(prior_summary, normalize_embeddings=True)
        actual_vec = self.encoder.encode(percept.fused_text, normalize_embeddings=True)

        surprise_score = self._cosine_distance(prior_vec, actual_vec)
        print(f"  📡 [PREDICTIVE CODER] Surprise score: {surprise_score:.4f}")

        prior_diff = self._generate_prior_diff(
            prior_summary, percept.fused_text, surprise_score
        )
        print(f"  📝 [PREDICTIVE CODER] Prior diff: {prior_diff}")

        return surprise_score, prior_summary, prior_diff