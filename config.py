"""
config.py

Single source of truth for all tunable parameters across the
Cognitive Digital Twin stack (MEMEX-1 + PERCEPT-1 + Self-Monitor).

To change any setting, edit this file only.
All modules import from here -- nothing is hardcoded elsewhere.
"""

import os

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROFILE_PATH      = os.getenv("CDT_PROFILE_PATH", "./semantic_profile.json")
PERSIST_DIRECTORY = os.getenv("CDT_PERSIST_DIR",  "./.memex_storage")

# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Models
MODEL_FAST    = "gpt-4o-mini"   # used for scoring, revision, prior diff
MODEL_VISION  = "gpt-4o"        # used for image description in binder

# ─────────────────────────────────────────────────────────────
# MEMEX-1 Tier 1
# ─────────────────────────────────────────────────────────────
TIER1_BASE_TOKEN_LIMIT = int(os.getenv("CDT_TIER1_TOKENS", 2000))

# ─────────────────────────────────────────────────────────────
# PERCEPT-1 attention thresholds
# ─────────────────────────────────────────────────────────────
PERCEPT_HIGH_THRESHOLD = float(os.getenv("CDT_PERCEPT_HIGH", 0.6))
PERCEPT_MID_THRESHOLD  = float(os.getenv("CDT_PERCEPT_MID",  0.25))

# ─────────────────────────────────────────────────────────────
# Load Monitor
# ─────────────────────────────────────────────────────────────
LOAD_WINDOW_SIZE      = int(os.getenv("CDT_LOAD_WINDOW", 5))
LOAD_LATENCY_BASELINE = float(os.getenv("CDT_LATENCY_BASELINE", 1.0))   # seconds
LOAD_LATENCY_MAX      = float(os.getenv("CDT_LATENCY_MAX",      10.0))  # seconds

# Token limits per load level
LOAD_TOKEN_LIMITS = {
    "LOW":      2000,
    "MODERATE": 1500,
    "HIGH":     1000,
    "CRITICAL": 600,
}

# PERCEPT-1 HIGH threshold per load level
LOAD_SURPRISE_THRESHOLDS = {
    "LOW":      0.60,
    "MODERATE": 0.65,
    "HIGH":     0.75,
    "CRITICAL": 0.85,
}

# ─────────────────────────────────────────────────────────────
# Meta-Evaluator
# ─────────────────────────────────────────────────────────────
META_HISTORY_N_RESULTS = int(os.getenv("CDT_META_HISTORY_N", 3))

# Composite score thresholds
META_PASS_THRESHOLD    = float(os.getenv("CDT_META_PASS",     0.7))
META_REVISE_THRESHOLD  = float(os.getenv("CDT_META_REVISE",   0.4))

# Composite weights: must sum to 1.0
META_WEIGHT_CONFIDENCE   = 0.40
META_WEIGHT_COMPLETENESS = 0.35
META_WEIGHT_CONSISTENCY  = 0.25

# ─────────────────────────────────────────────────────────────
# Retry policy (tenacity)
# ─────────────────────────────────────────────────────────────
RETRY_MAX_ATTEMPTS  = 3
RETRY_WAIT_MIN_S    = 2
RETRY_WAIT_MAX_S    = 10

# ─────────────────────────────────────────────────────────────
# Embedding model (sentence-transformers)
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
