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
"""

import json
import os
from typing import Dict, Any
from openai import OpenAI
from monitor_schemas import MetaScore


class MetaEvaluator:
    """
    Runs metacognitive monitoring and control on a candidate response.
    """

    def __init__(self, profile_path: str = "./semantic_profile.json"):
        self.profile_path = profile_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _read_profile(self) -> Dict[str, Any]:
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                return json.load(f)
        return {}

    def _score(self, query: str, response: str, profile: Dict[str, Any]) -> dict:
        """
        Calls GPT-4o-mini to score the response on all three dimensions.
        Returns a dict with scores and reasoning.
        """
        profile_str = json.dumps({k: v for k, v in profile.items() if v}, indent=None)

        prompt = f"""You are a metacognitive evaluator in a cognitive architecture.
Score the following response on three dimensions. Return ONLY valid JSON, no markdown.

QUERY: {query}

RESPONSE: {response}

KNOWN PROFILE CONTEXT: {profile_str if profile_str else "No profile established yet."}

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

        response_obj = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response_obj.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("```json").strip("```").strip()
        return json.loads(raw)

    def _revise(self, query: str, response: str, reasoning: str) -> str:
        """
        Attempts one revision pass if verdict is REVISE.
        """
        prompt = f"""A response was flagged for revision by a metacognitive evaluator.

ORIGINAL QUERY: {query}

ORIGINAL RESPONSE: {response}

ISSUE IDENTIFIED: {reasoning}

Rewrite the response to fix the identified issue. Be concise. Return only the revised response text."""

        result = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return result.choices[0].message.content.strip()

    def evaluate(self, query: str, response: str) -> MetaScore:
        """
        Main entry point. Scores a response and returns a MetaScore
        with verdict and optional revision.
        """
        profile = self._read_profile()
        scores = self._score(query, response, profile)

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
            print(f"  ✅ [META-EVAL] Revision complete.")

        return MetaScore(
            confidence=confidence,
            completeness=completeness,
            consistency=consistency,
            composite=composite,
            verdict=verdict,
            reasoning=reasoning,
            revised_response=revised_response,
        )
