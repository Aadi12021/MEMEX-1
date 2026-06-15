# PERCEPT-1 — Perception Daemon

**Week 2 of the Cognitive Digital Twin series.**  
Picks up where MEMEX-1 left off: if MEMEX-1 is how the twin *remembers*, PERCEPT-1 is how it *sees*.

---

## Architecture

```
RawPerceptInput  (text / JSON / image)
      ↓
MultimodalBinder      — fuses all modalities into one PerceptObject
      ↓
PredictiveCoder       — reads MEMEX-1 Tier 3 as prior, computes surprise score
      ↓
AttentionFilter       — gates salience: HIGH / MEDIUM / SUPPRESSED
      ↓
SalientPerceptObject  → Tier1WorkingMemory.add_message()
```

## Cognitive Science Grounding

| Module | Brain analog |
|---|---|
| MultimodalBinder | Multisensory integration / binding problem |
| PredictiveCoder | Friston's Free Energy Principle / predictive coding |
| AttentionFilter | Broadbent's filter model + Treisman's attenuation theory |

The key insight: the brain doesn't just receive input — it *predicts* input and only routes the
prediction error (surprise) forward. Low-surprise signals (things you already know) get suppressed
before they ever reach conscious attention.

## MEMEX-1 Integration

- **Prior source**: Tier 3 `semantic_profile.json` (same file your existing system writes)
- **Output destination**: `Tier1WorkingMemory.add_message()` via `SalientPerceptObject.to_message_content()`
- **Distance alignment**: Salience thresholds (0.25 / 0.6) mirror the RetrievalArbiter's `1.2` gate, both using cosine distance

## Files

```
percept_schemas.py      — RawPerceptInput, PerceptObject, SalientPerceptObject
multimodal_binder.py    — Stage 1: fuse text + structured + image
predictive_coder.py     — Stage 2: MEMEX-1 prior + surprise scoring
attention_filter.py     — Stage 3: salience gating
percept1_daemon.py      — Main orchestrator
run_percept1_test.py    — E2E integration test with MEMEX-1
```

## Setup

```bash
pip install anthropic sentence-transformers numpy pydantic
export ANTHROPIC_API_KEY=your_key
```

Place all PERCEPT-1 files in the same directory as your MEMEX-1 files, then:

```bash
python run_percept1_test.py
```

## Usage

```python
from percept1_daemon import Percept1Daemon
from memory_tier1 import Tier1WorkingMemory

daemon = Percept1Daemon(profile_path="./semantic_profile.json")
tier1  = Tier1WorkingMemory(session_id="my_session", max_token_limit=2000)

result = daemon.perceive(
    text="Some new information",
    structured={"context": "value"},   # optional
    image_base64="...",                # optional
)

# Only inject high-signal percepts into working memory
if result.salience != "SUPPRESSED":
    tier1.add_message("user", result.to_message_content())
```

## Salience Thresholds

| Surprise score | Salience | Action |
|---|---|---|
| >= 0.6 | HIGH | Full pass to Tier 1 |
| 0.25 – 0.6 | MEDIUM | Attenuated (truncated) pass |
| < 0.25 | SUPPRESSED | Dropped — already known |

Thresholds are configurable via `Percept1Daemon(high_threshold=..., mid_threshold=...)`.
