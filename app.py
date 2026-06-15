"""
app.py — PERCEPT-1 Web Interface

Run with:
    streamlit run app.py

Place this file in your MEMEX-1 directory alongside all PERCEPT-1 modules.
"""

import base64
import json
import os
import io
import time

import streamlit as st
import pandas as pd

from percept1_daemon import Percept1Daemon
from memory_tier1 import Tier1WorkingMemory
from memory_tier2 import Tier2EpisodicMemory

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PERCEPT-1",
    page_icon="👁️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Minimal custom styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighten sidebar */
    section[data-testid="stSidebar"] { min-width: 320px; max-width: 320px; }

    /* Salience badges */
    .badge-HIGH     { background:#1a472a; color:#6fcf97; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .badge-MEDIUM   { background:#3d2e00; color:#f2c94c; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
    .badge-SUPPRESSED { background:#2d1b1b; color:#eb5757; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }

    /* Stage cards */
    .stage-card { background:#1e1e2e; border:1px solid #313244; border-radius:8px; padding:14px 18px; margin-bottom:10px; }
    .stage-title { font-size:0.78rem; font-weight:700; letter-spacing:0.08em; color:#7f849c; text-transform:uppercase; margin-bottom:4px; }
    .stage-body  { font-size:0.92rem; color:#cdd6f4; }

    /* Surprise meter label */
    .surprise-label { font-size:0.78rem; color:#7f849c; letter-spacing:0.06em; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────
if "tier1" not in st.session_state:
    st.session_state.tier1 = Tier1WorkingMemory(
        session_id="percept_ui_session", max_token_limit=2000
    )
if "tier2" not in st.session_state:
    st.session_state.tier2 = Tier2EpisodicMemory()
if "history" not in st.session_state:
    st.session_state.history = []  # list of SalientPerceptObject results
if "daemon" not in st.session_state:
    st.session_state.daemon = None


# ─────────────────────────────────────────────────────────────
# Sidebar — config
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required for prior diff generation and image description.",
    )

    profile_path = st.text_input(
        "MEMEX-1 Tier 3 Profile Path",
        value="./semantic_profile.json",
        help="Path to your semantic_profile.json from MEMEX-1.",
    )

    st.markdown("---")
    st.markdown("### Attention Thresholds")

    high_thresh = st.slider(
        "HIGH salience threshold", 0.0, 2.0, 0.6, 0.05,
        help="Surprise ≥ this → fully promoted to Tier 1."
    )
    mid_thresh = st.slider(
        "MEDIUM salience threshold", 0.0, 2.0, 0.25, 0.05,
        help="Surprise ≥ this → attenuated and passed."
    )

    st.markdown("---")
    tier1_limit = st.number_input(
        "Tier 1 Token Limit", min_value=100, max_value=8000, value=2000, step=100
    )

    init_btn = st.button("Initialize / Reset Pipeline", type="primary", use_container_width=True)
    if init_btn:
        if not openai_key:
            st.error("Please enter your OpenAI API key.")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.session_state.daemon = Percept1Daemon(
                profile_path=profile_path,
                high_threshold=high_thresh,
                mid_threshold=mid_thresh,
            )
            st.session_state.tier1 = Tier1WorkingMemory(
                session_id="percept_ui_session", max_token_limit=tier1_limit
            )
            st.session_state.tier2 = Tier2EpisodicMemory()
            st.session_state.history = []
            st.success("Pipeline initialized.")

    # Auto-init if key is already in env
    if st.session_state.daemon is None and os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        st.session_state.daemon = Percept1Daemon(
            profile_path=profile_path,
            high_threshold=high_thresh,
            mid_threshold=mid_thresh,
        )

    st.markdown("---")
    st.markdown("### Tier 1 Memory")
    tier1 = st.session_state.tier1
    used = tier1.current_token_count
    total = tier1.max_token_limit
    st.progress(min(used / total, 1.0))
    st.caption(f"{used} / {total} tokens — {len(tier1.buffer)} messages")


# ─────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────
st.markdown("# 👁️ PERCEPT-1")
st.caption("Perception Daemon — Cognitive Digital Twin · Week 2")
st.markdown("---")

input_col, output_col = st.columns([1, 1], gap="large")

# ── LEFT: Input panel ──────────────────────────────────────
with input_col:
    st.markdown("### Input")

    text_input = st.text_area(
        "Text",
        placeholder="Type or paste text input here...",
        height=140,
    )

    st.markdown("**Structured Data** (JSON or CSV)")
    structured_file = st.file_uploader(
        "Drop a JSON or CSV file",
        type=["json", "csv"],
        label_visibility="collapsed",
    )
    structured_preview = None
    structured_data = None
    if structured_file:
        try:
            if structured_file.name.endswith(".json"):
                structured_data = json.load(structured_file)
                structured_preview = json.dumps(structured_data, indent=2)[:400]
            else:
                df = pd.read_csv(structured_file)
                structured_data = df.to_dict(orient="records")
                structured_preview = df.head(3).to_string()
            st.code(structured_preview, language="json")
        except Exception as e:
            st.error(f"Could not parse file: {e}")

    st.markdown("**Image**")
    image_file = st.file_uploader(
        "Drop an image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )
    image_b64 = None
    image_media_type = None
    if image_file:
        st.image(image_file, use_container_width=True)
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        image_media_type = image_file.type

    st.markdown("")
    source_label = st.text_input("Source label", value="user_input", help="Tag for where this input came from.")

    run_btn = st.button("▶ Run Perception Pipeline", type="primary", use_container_width=True)


# ── RIGHT: Output panel ────────────────────────────────────
with output_col:
    st.markdown("### Pipeline Output")

    if run_btn:
        if st.session_state.daemon is None:
            st.error("Pipeline not initialized. Enter your OpenAI API key in the sidebar and click Initialize.")
        elif not any([text_input, structured_data, image_b64]):
            st.warning("Please provide at least one input (text, file, or image).")
        else:
            with st.spinner("Running PERCEPT-1..."):
                try:
                    result = st.session_state.daemon.perceive(
                        text=text_input or None,
                        structured=structured_data or None,
                        image_base64=image_b64,
                        image_media_type=image_media_type or "image/jpeg",
                        source_label=source_label,
                    )

                    # Store in history
                    st.session_state.history.append(result)

                    # Feed into MEMEX-1 if not suppressed
                    if result.salience != "SUPPRESSED":
                        active, evicted = st.session_state.tier1.add_message(
                            "user", result.to_message_content()
                        )
                        if evicted:
                            st.session_state.tier2.process_eviction(
                                session_id=st.session_state.tier1.session_id,
                                evicted_messages=evicted,
                            )

                    # ── Stage cards ──
                    st.markdown(f"""
                    <div class="stage-card">
                        <div class="stage-title">🔗 Stage 1 — Multimodal Binder</div>
                        <div class="stage-body">Modalities bound: <b>{', '.join(result.percept.modalities_present)}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="stage-card">
                        <div class="stage-title">🔮 Stage 2 — Predictive Coder</div>
                        <div class="stage-body">
                            <span class="surprise-label">Surprise score</span><br/>
                            <b style="font-size:1.6rem;">{result.surprise_score:.4f}</b>
                            <span style="color:#7f849c;font-size:0.8rem;"> / 2.0</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="stage-card">
                        <div class="stage-title">🎯 Stage 3 — Attention Filter</div>
                        <div class="stage-body">
                            Salience: <span class="badge-{result.salience}">{result.salience}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Prior Diff")
                    st.info(result.prior_diff)

                    st.markdown("#### Fused Percept")
                    st.code(result.percept.fused_text, language=None)

                    if result.salience == "SUPPRESSED":
                        st.warning("Percept suppressed — below novelty threshold. Not injected into MEMEX-1.")
                    else:
                        st.success(f"Percept injected into MEMEX-1 Tier 1 (salience: {result.salience})")

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)

    elif not st.session_state.history:
        st.markdown("""
        <div style="color:#7f849c; margin-top:40px; text-align:center;">
            Configure the pipeline in the sidebar,<br/>add your inputs on the left,<br/>then click <b>Run Perception Pipeline</b>.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Perception history log
# ─────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Perception History")

    for i, r in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        badge = f'<span class="badge-{r.salience}">{r.salience}</span>'
        modalities = ", ".join(r.percept.modalities_present)
        with st.expander(f"Percept #{idx} — surprise {r.surprise_score:.4f} · {modalities}", expanded=(i == 0)):
            st.markdown(f"**Salience:** {badge}", unsafe_allow_html=True)
            st.markdown(f"**Prior diff:** {r.prior_diff}")
            st.code(r.percept.fused_text[:600] + ("..." if len(r.percept.fused_text) > 600 else ""), language=None)
