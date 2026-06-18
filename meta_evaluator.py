"""
meta_evaluator.py

Metacognition (Flavell, 1979) operationalized.

The brain doesn't just produce outputs — it monitors them.
Metacognitive monitoring asks: is what I'm about to say actually good?
Metacognitive control asks: if not, what do I do about it?

The MetaEvaluator runs a second pass on every response before
it's finalized, scoring it on three dimensions:
    Confidence   -- how certain is the response?
    Completeness -- does it actually address the query?
    Consistency  -- does it align with what's in the Tier 3 profile?

Verdict:
    PASS      -- response is good, finalize as-is
    REVISE    -- response has fixable issues, rewrite and retry once
    ESCALATE  -- response is too uncertain to trust, flag for human review

Improvement over v1:
    MetaScores are now stored in ChromaDB (Tier 2) after every evaluation.
    Before scoring a new response, the evaluator retrieves past scores for
    semantically similar queries. If similar queries historically scored low
    on a dimension, that context is fed into the scoring prompt so the LLM
    can calibrate early rather than discovering the issue after the fact.
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List

import chromadb
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from monitor_schemas import MetaScore


# Retry on transient OpenAI errors only
RETRYABLE = (APIError, APITimeoutError, RateLimitError)

FALLBACK_SCORES = {
    "confidence": 0.5,
    "completeness": 0.5,
    "consistency": 0.5,
    "reasoning": "Fallback scores applied — API call failed after retries.",
}


def _parse_json_safe(raw: str) -> dict:
    """
    Robustly extracts a JSON object from a string.
    Handles markdown fences, leading/trailing text, and malformed output.
    Falls back to FALLBACK_SCORES if nothing can be parsed.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print("  ⚠️  [META-EVAL] JSON parsing failed — applying fallback scores.")
    return FALLBACK_SCORES


class MetaEvaluator:
    """
    Runs metacognitive monitoring and control on a candidate response.

    All OpenAI calls retry up to 3 times with exponential backoff.
    JSON parsing is safe — never raises on malformed output.
    MetaScores are persisted to ChromaDB and retrieved for similar
    queries to calibrate future evaluations.
    """

    def __init__(
        self,
        profile_path: str = "./semantic_profile.json",
        persist_directory: str = "./.memex_storage",
        history_n_results: int = 3,
    ):
        self.profile_path = profile_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.history_n_results = history_n_results

        # ChromaDB collection for MetaScore history
        self._chroma = chromadb.PersistentClient(path=persist_directory)
        self._history = self._chroma.get_or_create_collection(
            name="meta_score_history"
        )

    def _read_profile(self) -> Dict[str, Any]:
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                return json.load(f)
        return {}

    def _store_score(self, query: str, meta: MetaScore):
        """
        Persists a MetaScore to ChromaDB after evaluation.
        Stored as a summary string with score metadata so future
        evaluations can retrieve and learn from similar past queries.
        """
        doc = (
            f"Query: {query} | "
            f"confidence={meta.confidence:.2f} | "
            f"completeness={meta.completeness:.2f} | "
            f"consistency={meta.consistency:.2f} | "
            f"verdict={meta.verdict} | "
            f"reasoning={meta.reasoning}"
        )
        score_id = f"meta_{int(datetime.utcnow().timestamp() * 1000)}"
        try:
            self._history.add(
                documents=[doc],
                metadatas=[{
                    "query": query[:200],
                    "verdict": meta.verdict,
                    "composite": meta.composite,
                    "timestamp": datetime.utcnow().isoformat(),
                }],
                ids=[score_id],
            )
        except Exception as e:
            print(f"  ⚠️  [META-EVAL] Failed to store score history: {e}")

    def _retrieve_similar_scores(self, query: str) -> List[str]:
        """
        Retrieves MetaScores from past evaluations on semantically
        similar queries. Returns empty list if history is too small.
        """
        try:
            count = self._history.count()
            if count == 0:
                return []
            n = min(self.history_n_results, count)
            results = self._history.query(
                query_texts=[query],
                n_results=n,
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"  ⚠️  [META-EVAL] History retrieval failed: {e}")
            return []

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=False,
    )
    def _call_score_api(self, prompt: str) -> str:
        """Isolated API call so retry decorator only wraps the network hop."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=False,
    )
    def _call_revise_api(self, prompt: str) -> str:
        """Isolated API call for the revision pass."""
        result = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return result.choices[0].message.content.strip()

    def _score(
        self,
        query: str,
        response: str,
        profile: Dict[str, Any],
        similar_history: List[str],
    ) -> dict:
        """
        Calls GPT-4o-mini to score the response on all three dimensions.
        Injects similar past MetaScores into the prompt for calibration.
        Returns a dict with scores and reasoning. Never raises.
        """
        profile_str = json.dumps({k: v for k, v in profile.items() if v}, indent=None)

        history_block = ""
        if similar_history:
            history_block = (
                "\nPAST EVALUATIONS ON SIMILAR QUERIES (use to calibrate your scores):\n"
                + "\n".join(f"- {h}" for h in similar_history)
                + "\n"
            )

        prompt = f"""You are a metacognitive evaluator in a cognitive architecture.
Score the following response on three dimensions. Return ONLY valid JSON, no markdown.

QUERY: {query}

RESPONSE: {response}

KNOWN PROFILE CONTEXT: {profile_str if profile_str else "No profile established yet."}
{history_block}
Score each dimension from 0.0 to 1.0:
- confidence: how certain and well-grounded is the response?
- completeness: does it fully address the query?
- consistency: does it align with or contradict the known profile?

Then produce a verdict:
- "PASS" if composite >= 0.7
- "REVISE" if composite is 0.4-0.69 (fixable issues)
- "ESCALATE" if composite < 0.4 (too uncertain)

Return this exact JSON structure:
{{
  "confidence": 0.0,
  "completeness": 0.0,
  "consistency": 0.0,
  "reasoning": "one sentence explanation of the verdict"
}}"""

        try:
            raw = self._call_score_api(prompt)
            return _parse_json_safe(raw)
        except Exception as e:
            print(f"  ⚠️  [META-EVAL] Scoring API failed after retries: {e}")
            return FALLBACK_SCORES

    def _revise(self, query: str, response: str, reasoning: str) -> str:
        """
        Attempts one revision pass if verdict is REVISE.
        Returns original response if the API call fails.
        """
        prompt = f"""A response was flagged for revision by a metacognitive evaluator.

ORIGINAL QUERY: {query}

ORIGINAL RESPONSE: {response}

ISSUE IDENTIFIED: {reasoning}

Rewrite the response to fix the identified issue. Be concise. Return only the revised response text."""

        try:
            return self._call_revise_api(prompt)
        except Exception as e:
            print(f"  ⚠️  [META-EVAL] Revision API failed after retries: {e}. Keeping original response.")
            return response

    def evaluate(self, query: str, response: str) -> MetaScore:
        """
        Main entry point. Scores a response and returns a MetaScore
        with verdict and optional revision. Never raises.

        Flow:
            1. Retrieve similar past MetaScores from ChromaDB
            2. Score response with history context injected
            3. Classify verdict, trigger revision if needed
            4. Store this MetaScore for future calibration
        """
        profile = self._read_profile()

        # Step 1: retrieve history for calibration
        similar_history = self._retrieve_similar_scores(query)
        if similar_history:
            print(f"  📚 [META-EVAL] Retrieved {len(similar_history)} similar past score(s) for calibration.")

        # Step 2: score with history context
        scores = self._score(query, response, profile, similar_history)

        confidence   = float(scores.get("confidence", 0.5))
        completeness = float(scores.get("completeness", 0.5))
        consistency  = float(scores.get("consistency", 0.5))
        reasoning    = scores.get("reasoning", "")

        # Weighted composite: confidence 40%, completeness 35%, consistency 25%
        composite = round(0.4 * confidence + 0.35 * completeness + 0.25 * consistency, 4)

        if composite >= 0.7:
            verdict = "PASS"
        elif composite >= 0.4:
            verdict = "REVISE"
        else:
            verdict = "ESCALATE"

        print(f"  🧠 [META-EVAL] confidence={confidence:.2f} | completeness={completeness:.2f} | consistency={consistency:.2f} | composite={composite:.2f} → {verdict}")
        print(f"  📝 [META-EVAL] {reasoning}")

        revised_response = None
        if verdict == "REVISE":
            print("  🔄 [META-EVAL] Triggering revision pass...")
            revised_response = self._revise(query, response, reasoning)
            print("  ✅ [META-EVAL] Revision complete.")

        meta = MetaScore(
            confidence=confidence,
            completeness=completeness,
            consistency=consistency,
            composite=composite,
            verdict=verdict,
            reasoning=reasoning,
            revised_response=revised_response,
        )

        # Step 4: persist for future calibration
        self._store_score(query, meta)

        return meta