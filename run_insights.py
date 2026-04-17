"""
FlocBot Run Insights – Streamlit application
Operator-friendly run summaries and comparisons for RoboJar/FlocBot exports.
"""

import io
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from metric_help import METRIC_HELP, classify_flag

# ─── Load environment variables (.env in local dev, Render env in prod) ───
load_dotenv()

# ─── Keep-alive (lightweight, no heavy deps) ─────────────────────────────
import keep_alive
keep_alive.start()

# ─── Page config (must come before any other st.* calls) ─────────────────
st.set_page_config(
    page_title="FlocBot Run Insights",
    page_icon="🔬",
    layout="wide",
)

# ─── UptimeRobot keyword marker (hidden HTML, renders on every load) ─────
st.markdown("<!-- APP_READY_FLOCBOT -->", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Lazy imports — deferred so the page shell renders immediately on cold boot
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_heavy_deps():
    """Import heavy libraries once and cache across reruns."""
    _CACHE_VERSION = 2  # bump to invalidate st.cache_resource after dep changes
    import pandas as _pd
    import numpy as _np
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots
    from fpdf import FPDF as _FPDF
    from flocbot_parser import parse_file as _parse_file, parse_file_all_sheets as _parse_file_all_sheets, RunMetadata as _RunMetadata
    from flocbot_metrics import (
        detect_phases as _detect_phases,
        compute_kpis as _compute_kpis,
        compute_score as _compute_score,
        Phase as _Phase,
        RunKPIs as _RunKPIs,
        DEFAULT_WEIGHTS as _DEFAULT_WEIGHTS,
        phase_by_name as _phase_by_name,
    )
    return {
        "pd": _pd,
        "np": _np,
        "go": _go,
        "make_subplots": _make_subplots,
        "FPDF": _FPDF,
        "parse_file": _parse_file,
        "parse_file_all_sheets": _parse_file_all_sheets,
        "RunMetadata": _RunMetadata,
        "detect_phases": _detect_phases,
        "compute_kpis": _compute_kpis,
        "compute_score": _compute_score,
        "Phase": _Phase,
        "RunKPIs": _RunKPIs,
        "DEFAULT_WEIGHTS": _DEFAULT_WEIGHTS,
        "phase_by_name": _phase_by_name,
    }


def _deps():
    """Shorthand accessor — always returns the cached dict."""
    return _load_heavy_deps()


# ─── Design system: single CSS injection ─────────────────────────────────
# Accent: #0284C7 (sky-600, clean water blue)
# Surface: #FFFFFF   Background: #F8FAFB   Tinted: #EFF6FA
# Text: #1B2A3D (primary)  #526580 (secondary)  #8C9BB0 (muted)
# Border: rgba(2,132,199,0.10)  Shadow: 0 1px 3px rgba(27,42,61,0.06)
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp {
        background: linear-gradient(168deg, #F8FAFB 0%, #EFF6FA 50%, #F4F8FB 100%);
    }

    /* ── Main content area ── */
    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Typography ── */
    h1, h2 {
        color: #1B2A3D !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h3, h4 {
        color: #1B2A3D !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    p, li, span, .stMarkdown {
        color: #334155;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EFF6FA 0%, #E8F1F7 100%);
        border-right: 1px solid rgba(2,132,199,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #526580;
        margin-top: 0.75rem;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }

    /* ── File uploader dropzone ── */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div:first-child {
        border: 2px dashed rgba(2,132,199,0.25) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.6) !important;
        transition: border-color 0.2s, background 0.2s;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div:first-child:hover {
        border-color: rgba(2,132,199,0.50) !important;
        background: rgba(255,255,255,0.85) !important;
    }

    /* ── Tabs styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 2px solid rgba(2,132,199,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #526580;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        transition: color 0.2s, border-color 0.2s;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0284C7;
    }
    .stTabs [aria-selected="true"] {
        color: #0284C7 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #0284C7 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #0284C7 !important;
    }

    /* ── Metric row labels ── */
    .metrics-row-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #526580;
        margin: 12px 0 1px 0;
        line-height: 1.2;
    }
    .metrics-row-desc {
        font-size: 0.68rem;
        color: #8C9BB0;
        margin: 0 0 6px 0;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.10);
        border-top: 3px solid #0284C7;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(27,42,61,0.06);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(27,42,61,0.10);
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] label {
        color: #526580 !important;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: none;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1B2A3D !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: anywhere;
        word-break: break-word;
        line-height: 1.3;
        min-width: 0;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: anywhere;
        min-width: 0;
    }

    /* ── Score progress bar ── */
    .score-bar-track {
        background: #E2E8F0;
        border-radius: 6px;
        height: 6px;
        margin-top: -6px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }

    /* ── Chart card wrappers ── */
    .chart-card {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.08);
        border-radius: 16px;
        padding: 24px 20px 12px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(27,42,61,0.06);
    }
    .chart-card h5 {
        margin: 0 0 2px 0;
        font-size: 0.92rem;
        font-weight: 600;
        color: #1B2A3D;
        letter-spacing: -0.01em;
    }
    .chart-card .chart-subtitle {
        margin: 0 0 12px 0;
        font-size: 0.76rem;
        color: #8C9BB0;
    }

    /* ── Containers / bordered panels ── */
    [data-testid="stContainer"] > div:has(> [data-testid="stVerticalBlock"]) {
        border-radius: 16px !important;
        border-color: rgba(2,132,199,0.10) !important;
        background: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(27,42,61,0.06);
    }

    /* ── Section dividers ── */
    hr {
        border-color: rgba(2,132,199,0.08);
        margin: 1.8rem 0;
    }

    /* ── Branded header accent ── */
    .header-accent {
        height: 3px;
        background: linear-gradient(90deg, #0284C7 0%, #38BDF8 40%, #BAE6FD 70%, transparent 100%);
        border-radius: 2px;
        margin-bottom: 16px;
    }

    /* ── Success / info / warning banners ── */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border-left-width: 4px;
    }

    /* ── Buttons ── */
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.85rem;
        border: 1px solid rgba(2,132,199,0.15);
        transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        border-color: #0284C7;
        box-shadow: 0 2px 8px rgba(2,132,199,0.15);
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(2,132,199,0.08);
        background: #FFFFFF;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(2,132,199,0.08);
    }

    /* ── Select boxes / dropdowns ── */
    [data-testid="stSelectbox"] > div > div {
        background: #FFFFFF;
        border: 1.5px solid rgba(2,132,199,0.20);
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(27,42,61,0.06);
    }
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: #0284C7;
    }

    /* ── Welcome page ── */
    .welcome-card {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.10);
        border-radius: 20px;
        padding: 40px 36px;
        box-shadow: 0 2px 8px rgba(27,42,61,0.06);
        max-width: 640px;
        margin: 2rem auto;
    }
    .welcome-card h2 {
        margin-top: 0;
    }
    /* Consistent input styling */
    .stTextInput [data-testid="stTextInputRootElement"] {
        border: 1px solid #c0c8d0 !important;
        border-radius: 6px !important;
        background-color: #f8f9fb !important;
        overflow: hidden !important;
    }
    .stTextInput [data-testid="stTextInputRootElement"]:focus-within {
        border-color: #0284c7 !important;
        background-color: #ffffff !important;
    }
    .stTextInput [data-testid="stTextInputRootElement"] * {
        border: none !important;
        background-color: inherit !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Authentication gate — must log in before accessing the app
# ═══════════════════════════════════════════════════════════════════════════

def _show_auth_page():
    """Render login / signup form. Returns True if user is now authenticated."""
    import supabase_client as auth

    # Already logged in?
    if st.session_state.get("user"):
        return True

    st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)
    st.markdown("## FlocBot Run Insights")
    st.caption("Sign in to analyze your RoboJar data.")

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In", use_container_width=True, type="primary"):
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                try:
                    data = auth.sign_in(email, password)
                    st.session_state["user"] = {
                        "id": data["user"]["id"],
                        "email": data["user"]["email"],
                        "access_token": data["access_token"],
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")

    with tab_signup:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.button("Create Account", use_container_width=True):
            if not new_email or not new_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                try:
                    data = auth.sign_up(new_email, new_password)
                    if data.get("user") and data["user"].get("identities"):
                        st.success("Account created! You can now sign in.")
                    else:
                        st.info("Check your email to confirm your account, then sign in.")
                except Exception as e:
                    st.error(f"Signup failed: {e}")

    return False


if not _show_auth_page():
    st.stop()

# ─── Logout button (shown in sidebar when logged in) ─────────────────────
def _add_logout_button():
    """Add logout to the bottom of the sidebar."""
    with st.sidebar:
        st.markdown("---")
        user_email = st.session_state["user"]["email"]
        st.caption(f"Signed in as **{user_email}**")
        if st.button("Sign Out", use_container_width=True):
            import supabase_client as auth
            try:
                auth.sign_out(st.session_state["user"]["access_token"])
            except Exception:
                pass
            st.session_state.pop("user", None)
            st.rerun()

_add_logout_button()

PHASE_COLORS = {
    "rapid_mix": "rgba(239, 68, 68, 0.08)",
    "flocculation": "rgba(16, 185, 129, 0.08)",
    "settling": "rgba(2, 132, 199, 0.08)",
}
PHASE_LABELS = {
    "rapid_mix": "Rapid Mix",
    "flocculation": "Flocculation",
    "settling": "Settling",
}

CHART_HEIGHT = 500
CHART_LAYOUT = dict(
    template="plotly_white",
    height=CHART_HEIGHT,
    margin=dict(t=100, b=48, l=56, r=24),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.06,
        xanchor="center",
        x=0.5,
        font=dict(size=11, color="#526580"),
    ),
    font=dict(family="Inter, -apple-system, system-ui, sans-serif", size=12, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    xaxis=dict(gridcolor="rgba(2,132,199,0.06)", zerolinecolor="rgba(2,132,199,0.10)"),
    yaxis=dict(gridcolor="rgba(2,132,199,0.06)", zerolinecolor="rgba(2,132,199,0.10)"),
)

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("FlocBot Run Insights")
    st.caption("Upload RoboJar exports to analyze flocculation performance.")

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    uploaded_files = st.file_uploader(
        "Upload RoboJar exports",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state['uploader_key']}",
    )

    # Fetch past runs list (metadata only) — cached in session_state
    import supabase_client as _db
    from ui_operator import PlantBaseline, save_baseline_from_kpi, load_baseline

    if "past_runs_meta" not in st.session_state:
        try:
            st.session_state["past_runs_meta"] = _db.get_runs_list(
                st.session_state["user"]["access_token"]
            )
        except Exception:
            st.session_state["past_runs_meta"] = []

    # Load saved user preferences (scoring weights) once per session
    if "prefs_loaded" not in st.session_state:
        st.session_state["prefs_loaded"] = True
        try:
            prefs = _db.get_preferences(st.session_state["user"]["access_token"])
            if prefs:
                weights_pref = prefs.get("scoring_weights", {})
                if "w1" in weights_pref:
                    st.session_state["w1"] = weights_pref["w1"]
                if "w2" in weights_pref:
                    st.session_state["w2"] = weights_pref["w2"]
                if "w3" in weights_pref:
                    st.session_state["w3"] = weights_pref["w3"]
                if "w4" in weights_pref:
                    st.session_state["w4"] = weights_pref["w4"]
        except Exception:
            pass  # preferences table may not exist yet; use defaults

    _past_meta = st.session_state["past_runs_meta"]

    # ── Plant Baseline ──
    st.markdown("---")
    st.subheader("Plant Baseline")

    # Helper: save a run as baseline
    def _save_baseline_from_run(run_meta):
        sj = run_meta.get("summary_json", {})
        proto = sj.get("protocol_title", run_meta.get("protocol", ""))
        if not proto:
            st.warning("This run has no protocol — cannot set as baseline.")
            return False
        bl = PlantBaseline(
            protocol=proto,
            growth_rate_um_per_min=sj.get("growth_rate_um_per_min"),
            time_to_300_min=(sj.get("time_to_thresholds_min") or {}).get("300") or (sj.get("time_to_thresholds_min") or {}).get("300.0"),
            pre_settle_diameter_um=sj.get("pre_settle_diameter_um"),
            t50_min=sj.get("t50_min"),
        )
        from ui_operator import save_baseline as _save_bl
        _save_bl(proto, bl)
        return True

    # Look for any existing baseline across all protocols
    _all_protocols = set(
        rm.get("protocol") or rm.get("summary_json", {}).get("protocol_title", "")
        for rm in _past_meta
    ) if _past_meta else set()
    _all_protocols.discard("")

    existing_bl = None
    _display_protocol = ""
    for _p in _all_protocols:
        existing_bl = load_baseline(_p)
        if existing_bl:
            _display_protocol = _p
            break

    if existing_bl:
        # ── STATE 2: Baseline set — compact read-only display ──
        st.caption(f"**{_display_protocol}**")
        bl_items = []
        if existing_bl.growth_rate_um_per_min is not None:
            bl_items.append(f"Growth: {existing_bl.growth_rate_um_per_min:.0f} µm/min")
        if existing_bl.pre_settle_diameter_um is not None:
            bl_items.append(f"Pre-settle: {existing_bl.pre_settle_diameter_um:.0f} µm")
        if existing_bl.t50_min is not None:
            bl_items.append(f"t50: {existing_bl.t50_min:.1f} min")
        if bl_items:
            st.caption(" · ".join(bl_items))

        # ── STATE 3: Change baseline (toggle) ──
        if not st.session_state.get("_bl_changing"):
            if st.button("Change Baseline", key="bl_change_toggle"):
                st.session_state["_bl_changing"] = True
                st.session_state.pop("_confirm_baseline", None)
                st.rerun()
        else:
            if _past_meta:
                _bl_options = {
                    rm["id"]: f"{rm.get('file_name', 'Run')} ({rm.get('created_at', '')[:10]})"
                    for rm in _past_meta
                }
                _bl_run_id = st.selectbox(
                    "Select a run",
                    options=list(_bl_options.keys()),
                    format_func=lambda x: _bl_options[x],
                    key="bl_run_select",
                )

                if not st.session_state.get("_confirm_baseline"):
                    _col_save, _col_cancel = st.columns(2)
                    with _col_save:
                        if st.button("Save", key="bl_save_new", type="primary"):
                            st.session_state["_confirm_baseline"] = _bl_run_id
                            st.rerun()
                    with _col_cancel:
                        if st.button("Cancel", key="bl_cancel_change"):
                            st.session_state.pop("_bl_changing", None)
                            st.rerun()
                else:
                    st.warning("Are you sure you want to replace the current baseline?")
                    _col_yes, _col_no = st.columns(2)
                    with _col_yes:
                        if st.button("Yes, replace", key="bl_confirm_yes", type="primary"):
                            _bl_rm = next((r for r in _past_meta if r["id"] == st.session_state["_confirm_baseline"]), None)
                            if _bl_rm and _save_baseline_from_run(_bl_rm):
                                st.session_state.pop("_confirm_baseline", None)
                                st.session_state.pop("_bl_changing", None)
                                st.success("Baseline updated!")
                                st.rerun()
                    with _col_no:
                        if st.button("Cancel", key="bl_confirm_no"):
                            st.session_state.pop("_confirm_baseline", None)
                            st.session_state.pop("_bl_changing", None)
                            st.rerun()

    elif _past_meta:
        # ── STATE 1: No baseline yet — prompt to set one ──
        st.caption("No baseline set yet. Select a run to establish your plant baseline.")
        _bl_options = {
            rm["id"]: f"{rm.get('file_name', 'Run')} ({rm.get('created_at', '')[:10]})"
            for rm in _past_meta
        }
        _bl_run_id = st.selectbox(
            "Select a run",
            options=list(_bl_options.keys()),
            format_func=lambda x: _bl_options[x],
            key="bl_run_select_initial",
        )
        if st.button("Set as Baseline", key="bl_set_initial", type="primary"):
            _bl_rm = next((r for r in _past_meta if r["id"] == _bl_run_id), None)
            if _bl_rm and _save_baseline_from_run(_bl_rm):
                st.success("Baseline saved!")
                st.rerun()

    else:
        # No runs at all yet
        st.caption("Upload and save runs first, then set one as your plant baseline.")

    # ── Run Library ──
    st.markdown("---")
    st.subheader("Run Library")

    # Initialize selected set — newly uploaded runs get added here automatically
    if "selected_run_ids" not in st.session_state:
        st.session_state["selected_run_ids"] = set()

    if _past_meta:
        if st.button("Clear All", use_container_width=True, key="clear_all_runs"):
            st.session_state["selected_run_ids"].clear()
            st.rerun()

        # Scrollable run list
        _run_list_container = st.container(height=300)
        with _run_list_container:
            for _rm in _past_meta:
                _rid = _rm["id"]
                _date = _rm.get("created_at", "")[:10]
                _label = f"{_rm.get('file_name', 'Run')} ({_date})"
                _default = _rid in st.session_state["selected_run_ids"]

                _col_cb, _col_del = st.columns([5, 1])
                with _col_cb:
                    if st.checkbox(_label, value=_default, key=f"run_{_rid}"):
                        st.session_state["selected_run_ids"].add(_rid)
                    else:
                        st.session_state["selected_run_ids"].discard(_rid)
                with _col_del:
                    if st.button("🗑", key=f"del_{_rid}", help="Delete this run"):
                        st.session_state["_confirm_delete_run"] = _rid

        # Confirmation dialog for delete (shown outside the scroll container)
        _del_rid = st.session_state.get("_confirm_delete_run")
        if _del_rid:
            _del_name = next(
                (r.get("file_name", "this run") for r in _past_meta if r["id"] == _del_rid),
                "this run",
            )
            st.warning(f"Are you sure you want to delete **{_del_name}**?")
            _col_yes, _col_no = st.columns(2)
            with _col_yes:
                if st.button("Yes, delete", key="confirm_del_yes", type="primary"):
                    try:
                        _db.delete_run(
                            st.session_state["user"]["access_token"], _del_rid
                        )
                        st.session_state["selected_run_ids"].discard(_del_rid)
                        st.session_state.get("loaded_runs_cache", {}).pop(_del_rid, None)
                        st.session_state.pop("past_runs_meta", None)
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                    st.session_state.pop("_confirm_delete_run", None)
                    st.rerun()
            with _col_no:
                if st.button("Cancel", key="confirm_del_no"):
                    st.session_state.pop("_confirm_delete_run", None)
                    st.rerun()

    else:
        st.caption("No past runs saved yet.")

    # ── Mode selector ──
    # Handle pending mode-switch request (from Summary Mode button)
    if st.session_state.get("_switch_to_advanced"):
        st.session_state["app_mode"] = "Detailed"
        del st.session_state["_switch_to_advanced"]
    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = "Summary"
    st.markdown("---")
    st.subheader("Dashboard Mode")
    app_mode = st.radio(
        "Dashboard Mode",
        ["Summary", "Detailed"],
        key="app_mode",
        label_visibility="collapsed",
        help="Summary shows a simplified traffic-light view. Detailed shows full controls and tables.",
    )

    # ── Advanced-only sidebar controls ──
    thresholds = [250, 300, 350, 400, 450]
    weights = None
    score_threshold = 300.0

    if app_mode == "Detailed":
        st.markdown("---")

        # ── Thresholds ──
        st.subheader("Diameter Thresholds")
        default_thresholds = "250, 300, 350, 400, 450"
        threshold_str = st.text_input(
            "Thresholds (μm)",
            default_thresholds,
            help="Comma-separated diameter values for time-to-threshold metrics.",
        )
        st.caption("Comma-separated values in μm.")
        try:
            thresholds = sorted(set(
                float(t.strip()) for t in threshold_str.split(",") if t.strip()
            ))
        except ValueError:
            thresholds = [250, 300, 350, 400, 450]
            st.warning("Invalid thresholds – using defaults.")

        if len(thresholds) > 5:
            thresholds = thresholds[:5]
            st.caption("Showing first 5 thresholds (to keep the dashboard readable).")

        st.markdown("---")

        # ── Scoring weights (collapsed) ──
        with st.expander("Scoring Weights", expanded=False):
            st.caption("Adjust relative importance of each metric.")
            w_time = st.slider("Time to threshold", 0, 100, 30, key="w1")
            w_diam = st.slider("Pre-settle diameter", 0, 100, 30, key="w2")
            w_cv = st.slider("Signal noise", 0, 100, 20, key="w3")
            w_t50 = st.slider("Settling t50", 0, 100, 20, key="w4")

            score_threshold = 300.0
            score_thr_options = [t for t in thresholds if t > 0]
            if score_thr_options:
                score_threshold = st.selectbox(
                    "Threshold for scoring (μm)",
                    score_thr_options,
                    index=min(1, len(score_thr_options) - 1),
                    help="Which diameter threshold to use for the time-to-threshold score component.",
                )

        # Persist scoring weights to Supabase when they change
        _current_weights = {"w1": w_time, "w2": w_diam, "w3": w_cv, "w4": w_t50}
        if st.session_state.get("_saved_weights") != _current_weights:
            st.session_state["_saved_weights"] = _current_weights
            try:
                user = st.session_state["user"]
                _db.save_preferences(
                    user["access_token"], user["id"],
                    {"scoring_weights": _current_weights},
                )
            except Exception:
                pass  # silent — don't break UX if save fails

        total_w = w_time + w_diam + w_cv + w_t50
        if total_w == 0:
            total_w = 1  # avoid div-by-zero
        weights = {
            "time_to_300": w_time / total_w,
            "pre_settle_diameter": w_diam / total_w,
            "floc_noise_mad": w_cv / total_w,
            "settling_t50": w_t50 / total_w,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Upload ingestion — parse, save to DB, auto-select, then let Run Library
# be the single source of truth for what's displayed.
# ═══════════════════════════════════════════════════════════════════════════

if uploaded_files:
    # Only process files we haven't already ingested this session
    _current_file_set = frozenset(
        getattr(f, "file_id", f"{getattr(f, 'name', '')}_{getattr(f, 'size', 0)}")
        for f in uploaded_files
    )
    if _current_file_set != st.session_state.get("_last_file_set"):
        st.session_state["_last_file_set"] = _current_file_set

        # Load heavy deps for parsing
        with st.spinner("Loading analysis libraries..."):
            D = _deps()
        _parse_all = D["parse_file_all_sheets"]
        _detect = D["detect_phases"]
        _compute = D["compute_kpis"]
        _score = D["compute_score"]

        _ingest_errors = []
        _saved_count = 0
        for f in uploaded_files:
            try:
                f.seek(0)
                for df, meta in _parse_all(f):
                    phases = _detect(df)
                    kpi = _compute(df, phases, thresholds=[250, 300, 350, 400, 450])
                    _score(kpi)
                    # Build summary for DB
                    summary = {
                        "filename": meta.filename,
                        "protocol_title": meta.protocol_title,
                        "run_chemistry": meta.run_chemistry,
                        "run_dosage": meta.run_dosage,
                        "comments": meta.comments,
                        "generated_timestamp": meta.generated_timestamp,
                        "rapid_mix_duration_min": kpi.rapid_mix_duration_min,
                        "flocculation_duration_min": kpi.flocculation_duration_min,
                        "settling_duration_min": kpi.settling_duration_min,
                        "growth_rate_um_per_min": kpi.growth_rate_um_per_min,
                        "growth_rate_r2": kpi.growth_rate_r2,
                        "growth_rate_window": kpi.growth_rate_window,
                        "pre_settle_diameter_um": kpi.pre_settle_diameter_um,
                        "plateau_mean_um": kpi.plateau_mean_um,
                        "floc_noise_mad": kpi.floc_noise_mad,
                        "settle_baseline_vol_conc": kpi.settle_baseline_vol_conc,
                        "t50_min": kpi.t50_min,
                        "t10_min": kpi.t10_min,
                        "time_to_thresholds_min": {str(k): v for k, v in kpi.time_to_thresholds_min.items()},
                        "score": kpi.score,
                        "score_components": kpi.score_components,
                        "score_reason": kpi.score_reason,
                        "quality_flags": kpi.quality_flags,
                    }
                    run_data_json = df.to_json(orient="split")
                    user = st.session_state["user"]
                    row = _db.save_run(
                        access_token=user["access_token"],
                        user_id=user["id"],
                        file_name=meta.filename,
                        protocol=meta.protocol_title or "",
                        chemistry=meta.run_chemistry or "",
                        dosage=meta.run_dosage or "",
                        summary_json=summary,
                        run_data_json=run_data_json,
                    )
                    st.session_state["selected_run_ids"].add(row["id"])
                    _saved_count += 1
            except Exception as e:
                _ingest_errors.append((getattr(f, "name", "?"), str(e)))

        # Refresh library, clear uploader, and show feedback
        st.session_state.pop("past_runs_meta", None)
        for fname, err in _ingest_errors:
            st.error(f"**{fname}:** {err}")
        if _saved_count:
            st.session_state["uploader_key"] += 1  # reset uploader to empty drop zone
            st.session_state.pop("_last_file_set", None)
            st.success(f"{_saved_count} run{'s' if _saved_count != 1 else ''} saved to library.")
            st.rerun()  # rerun so library list updates


# ═══════════════════════════════════════════════════════════════════════════
# Welcome screen (no runs selected in library)
# ═══════════════════════════════════════════════════════════════════════════

if not st.session_state.get("selected_run_ids"):
    st.markdown("""<div class="welcome-card">
        <h2>Welcome to FlocBot Run Insights</h2>
        <p style="color:#526580; margin-bottom:20px;">
            Upload one or more RoboJar Excel exports using the sidebar,
            or select past runs from your Run Library.
        </p>
        <ul style="color:#334155; line-height:2;">
            <li>Automatic phase detection (Rapid Mix / Flocculation / Settling)</li>
            <li>Key performance indicators and an overall run score</li>
            <li>Interactive Plotly charts with phase shading</li>
            <li>Multi-run comparison with side-by-side metrics</li>
            <li>CSV / JSON / PDF export for reporting</li>
        </ul>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Heavy deps loaded only when runs are selected for display
# ═══════════════════════════════════════════════════════════════════════════

with st.spinner("Loading analysis libraries..."):
    try:
        D = _deps()
    except Exception as e:
        st.error(f"Failed to load required libraries: {e}")
        st.info("If this persists, check that all packages in requirements.txt are installable.")
        st.stop()

pd = D["pd"]
np = D["np"]
go = D["go"]
FPDF = D["FPDF"]
parse_file = D["parse_file"]
parse_file_all_sheets = D["parse_file_all_sheets"]
detect_phases = D["detect_phases"]
compute_kpis = D["compute_kpis"]
compute_score = D["compute_score"]
Phase = D["Phase"]
RunKPIs = D["RunKPIs"]
phase_by_name = D["phase_by_name"]
RunMetadata = D["RunMetadata"]


# ═══════════════════════════════════════════════════════════════════════════
# Processing — single source: load selected runs from Run Library
# ═══════════════════════════════════════════════════════════════════════════

def kpi_to_dict(meta, kpi):
    """Full KPI dict for JSON export and DB storage."""
    return {
        "filename": meta.filename,
        "protocol_title": meta.protocol_title,
        "run_chemistry": meta.run_chemistry,
        "run_dosage": meta.run_dosage,
        "comments": meta.comments,
        "generated_timestamp": meta.generated_timestamp,
        "rapid_mix_duration_min": kpi.rapid_mix_duration_min,
        "flocculation_duration_min": kpi.flocculation_duration_min,
        "settling_duration_min": kpi.settling_duration_min,
        "growth_rate_um_per_min": kpi.growth_rate_um_per_min,
        "growth_rate_r2": kpi.growth_rate_r2,
        "growth_rate_window": kpi.growth_rate_window,
        "pre_settle_diameter_um": kpi.pre_settle_diameter_um,
        "plateau_mean_um": kpi.plateau_mean_um,
        "floc_noise_mad": kpi.floc_noise_mad,
        "settle_baseline_vol_conc": kpi.settle_baseline_vol_conc,
        "t50_min": kpi.t50_min,
        "t10_min": kpi.t10_min,
        "time_to_thresholds_min": {str(k): v for k, v in kpi.time_to_thresholds_min.items()},
        "score": kpi.score,
        "score_components": kpi.score_components,
        "score_reason": kpi.score_reason,
        "quality_flags": kpi.quality_flags,
    }


def _load_run_from_db(run_meta: dict) -> dict | None:
    """Load a run from DB and reconstruct the run dict {df, meta, phases}."""
    try:
        user = st.session_state["user"]
        run_data_json = _db.get_run_data(user["access_token"], run_meta["id"])
        df = pd.read_json(run_data_json, orient="split")
        sj = run_meta.get("summary_json", {})
        meta = RunMetadata(
            filename=sj.get("filename", run_meta.get("file_name", "")),
            generated_timestamp=sj.get("generated_timestamp", ""),
            protocol_title=sj.get("protocol_title", run_meta.get("protocol", "")),
            run_chemistry=sj.get("run_chemistry", run_meta.get("chemistry", "")),
            run_dosage=sj.get("run_dosage", run_meta.get("dosage", "")),
            comments=sj.get("comments", ""),
        )
        phases = detect_phases(df)
        return {"df": df, "meta": meta, "phases": phases}
    except Exception:
        return None


# --- Build runs list from selected library runs only ---
runs = []

if "loaded_runs_cache" not in st.session_state:
    st.session_state["loaded_runs_cache"] = {}

_lib_meta = st.session_state.get("past_runs_meta", [])
_selected = st.session_state["selected_run_ids"]
_cache = st.session_state["loaded_runs_cache"]

for rm in _lib_meta:
    rid = rm["id"]
    if rid not in _selected:
        continue

    # Load from cache or fetch from DB
    if rid not in _cache:
        loaded = _load_run_from_db(rm)
        if loaded:
            _cache[rid] = loaded
        else:
            st.warning(f"Could not load run: {rm.get('file_name', rid)}")
            continue

    base = _cache[rid]
    # Re-compute KPIs with current sidebar settings (thresholds/weights)
    kpi = compute_kpis(base["df"], base["phases"], thresholds=thresholds)
    score_weights = dict(weights) if weights is not None else None
    compute_score(kpi, weights=score_weights, threshold_for_time=score_threshold)
    runs.append({
        "df": base["df"],
        "meta": base["meta"],
        "phases": base["phases"],
        "kpi": kpi,
    })

if not runs:
    st.warning("Selected runs could not be loaded. Try refreshing the Run Library.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Root container – cleared on every rerun so mode-switch has no ghost UI
# ═══════════════════════════════════════════════════════════════════════════
root = st.empty()

# ═══════════════════════════════════════════════════════════════════════════
# Summary Mode (early exit – renders its own layout)
# ═══════════════════════════════════════════════════════════════════════════

if app_mode == "Summary":
    with root.container():
        from ui_operator import show_operator_mode
        show_operator_mode(
            st, runs, go,
            phase_colors=PHASE_COLORS,
            phase_labels=PHASE_LABELS,
            chart_layout=CHART_LAYOUT,
            phase_by_name=phase_by_name,
        )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: plotting
# ═══════════════════════════════════════════════════════════════════════════

def add_phase_shading(fig, phases: list, row=None, col=None):
    for p in phases:
        fig.add_vrect(
            x0=p.start_min, x1=p.end_min,
            fillcolor=PHASE_COLORS.get(p.name, "rgba(200,200,200,0.1)"),
            layer="below", line_width=0,
            row=row, col=col,
        )
        fig.add_annotation(
            x=(p.start_min + p.end_min) / 2,
            y=1.01,
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            text=PHASE_LABELS.get(p.name, p.name),
            showarrow=False,
            font=dict(size=10, color="gray"),
        )


def plot_diameter(df, phases, kpi, meta_label, thresholds_to_show):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_min"], y=df.get("diameter_um"),
        mode="lines+markers", marker=dict(size=3),
        name=meta_label,
        hovertemplate="Time: %{x:.1f} min<br>Diameter: %{y:.0f} μm<extra></extra>",
    ))
    add_phase_shading(fig, phases)

    # Threshold crossing markers
    for thr, t_val in kpi.time_to_thresholds_min.items():
        if thr in thresholds_to_show and t_val is not None:
            fig.add_trace(go.Scatter(
                x=[t_val], y=[thr], mode="markers",
                marker=dict(size=10, symbol="star", color="orange"),
                name=f"{thr} μm @ {t_val:.1f} min",
                hovertemplate=f"{thr} μm reached at {t_val:.1f} min<extra></extra>",
            ))

    # Settling start marker
    se = phase_by_name(phases, "settling")
    if se:
        fig.add_vline(x=se.start_min, line_dash="dash", line_color="#0284C7")
        fig.add_annotation(
            x=se.start_min, y=1, yref="paper",
            xanchor="right", yanchor="top",
            text="Settle start", showarrow=False,
            xshift=-6,
            font=dict(size=11, color="#0284C7"),
        )

    fig.update_layout(
        title=dict(text="Mean Floc Diameter vs Time", y=0.97, yanchor="top", x=0.5, xanchor="center"),
        xaxis_title="Time (min)", yaxis_title="Mean Diameter (μm)",
        **CHART_LAYOUT,
    )
    return fig


def plot_vol_conc(df, phases, kpi, meta_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_min"], y=df.get("vol_conc_mm3_L"),
        mode="lines+markers", marker=dict(size=3),
        name=meta_label,
        hovertemplate="Time: %{x:.1f} min<br>Vol. Conc: %{y:.1f} mm³/L<extra></extra>",
    ))
    add_phase_shading(fig, phases)

    se = phase_by_name(phases, "settling")
    if se and kpi.settle_baseline_vol_conc is not None:
        bl = kpi.settle_baseline_vol_conc
        fig.add_hline(y=bl, line_dash="dot", line_color="gray",
                      annotation_text=f"Baseline {bl:.1f}")
        if kpi.t50_min is not None:
            fig.add_trace(go.Scatter(
                x=[se.start_min + kpi.t50_min], y=[bl * 0.5],
                mode="markers", marker=dict(size=10, symbol="diamond", color="red"),
                name=f"t50 = {kpi.t50_min:.1f} min",
            ))
        if kpi.t10_min is not None:
            fig.add_trace(go.Scatter(
                x=[se.start_min + kpi.t10_min], y=[bl * 0.1],
                mode="markers", marker=dict(size=10, symbol="diamond", color="darkred"),
                name=f"t10 = {kpi.t10_min:.1f} min",
            ))

    fig.update_layout(
        title=dict(text="Vol. Concentration vs Time", y=0.97, yanchor="top", x=0.5, xanchor="center"),
        xaxis_title="Time (min)", yaxis_title="Vol. Concentration (mm³/L)",
        **CHART_LAYOUT,
    )
    return fig


def plot_floc_count(df, phases, meta_label):
    if "floc_count_ml" not in df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_min"], y=df["floc_count_ml"],
        mode="lines+markers", marker=dict(size=3),
        name=meta_label,
    ))
    add_phase_shading(fig, phases)
    fig.update_layout(
        title=dict(text="Floc Count vs Time", y=0.97, yanchor="top", x=0.5, xanchor="center"),
        xaxis_title="Time (min)", yaxis_title="Floc Count (per mL)",
        **CHART_LAYOUT,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Summary table builder
# ═══════════════════════════════════════════════════════════════════════════

def build_summary_row(meta, kpi):
    row = {
        "File": meta.filename,
        "Protocol": meta.protocol_title,
        "Chemistry": meta.run_chemistry,
        "Dosage": meta.run_dosage,
        "Comments": meta.comments,
        "Rapid Mix (min)": kpi.rapid_mix_duration_min,
        "Flocculation (min)": kpi.flocculation_duration_min,
        "Settling (min)": kpi.settling_duration_min,
        "Growth Rate (μm/min)": kpi.growth_rate_um_per_min,
        "Pre-settle Ø (μm)": kpi.pre_settle_diameter_um,
        "Plateau Mean (μm)": kpi.plateau_mean_um,
        "Signal Noise (μm)": kpi.floc_noise_mad,
        "t50 (min)": kpi.t50_min,
        "t10 (min)": kpi.t10_min,
        "Score": kpi.score,
    }
    for thr, val in sorted(kpi.time_to_thresholds_min.items()):
        row[f"t_{int(thr)}μm (min)"] = val
    return row


def _fig_to_png(fig, width=900, height=400):
    """Render a Plotly figure to PNG bytes via kaleido. Returns None on failure."""
    import sys
    if sys.platform == "win32":
        return None
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _pdf_safe(text):
    """Replace Unicode chars unsupported by Helvetica with ASCII equivalents."""
    return (
        str(text)
        .replace("\u03bc", "u")   # μ → u
        .replace("\u00d8", "O")   # Ø → O
        .replace("\u00f8", "o")   # ø → o
        .replace("\u2014", "-")   # — → -
        .replace("\u2013", "-")   # – → -
        .replace("\u2026", "...")  # … → ...
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )


def _pdf_add_run(pdf, run, thresholds_to_show, chart_w=190):
    """Add a single run's metadata, KPIs, and charts to the PDF."""
    meta = run["meta"]
    kpi = run["kpi"]
    df = run["df"]
    phases = run["phases"]

    # ── Run heading ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)  # #1E293B
    pdf.cell(0, 6, _pdf_safe(meta.label), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    meta_rows = [
        ("Filename", meta.filename),
        ("Timestamp", meta.generated_timestamp or "-"),
        ("Protocol", meta.protocol_title or "-"),
        ("Chemistry", meta.run_chemistry or "-"),
        ("Dosage", meta.run_dosage or "-"),
    ]
    pdf.set_font("Helvetica", "", 9)
    for label, value in meta_rows:
        pdf.set_text_color(100, 116, 139)
        pdf.cell(32, 5, label, new_x="RIGHT")
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 5, _pdf_safe(value), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── KPI table ──
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "Key Performance Indicators", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    kpi_rows = [
        ("Score", f"{kpi.score}/100" if kpi.score is not None else "N/A"),
        ("Growth Rate", f"{kpi.growth_rate_um_per_min} \u00b5m/min" if kpi.growth_rate_um_per_min else "N/A"),
        ("Pre-settle O", f"{kpi.pre_settle_diameter_um} \u00b5m" if kpi.pre_settle_diameter_um else "N/A"),
        ("Plateau Mean", f"{kpi.plateau_mean_um} \u00b5m" if kpi.plateau_mean_um else "N/A"),
        ("Signal Noise", f"{kpi.floc_noise_mad} \u00b5m" if kpi.floc_noise_mad else "N/A"),
        ("Settling t50", f"{kpi.t50_min} min" if kpi.t50_min else "N/A"),
        ("Settling t10", f"{kpi.t10_min} min" if kpi.t10_min else "N/A"),
        ("Rapid Mix", f"{kpi.rapid_mix_duration_min:.1f} min" if kpi.rapid_mix_duration_min else "N/A"),
        ("Flocculation", f"{kpi.flocculation_duration_min:.1f} min" if kpi.flocculation_duration_min else "N/A"),
        ("Settling", f"{kpi.settling_duration_min:.1f} min" if kpi.settling_duration_min else "N/A"),
    ]
    # Add threshold times
    for thr in sorted(thresholds_to_show):
        val = kpi.time_to_thresholds_min.get(thr)
        kpi_rows.append((f"t {int(thr)} \u00b5m", f"{val} min" if val is not None else "Not reached"))

    pdf.set_font("Helvetica", "", 9)
    col_w = 45
    val_w = 45
    fill = False
    for label, value in kpi_rows:
        if fill:
            pdf.set_fill_color(248, 250, 252)  # #F8FAFC
        pdf.set_text_color(100, 116, 139)
        pdf.cell(col_w, 5.5, _pdf_safe(label), fill=fill, new_x="RIGHT")
        pdf.set_text_color(30, 41, 59)
        pdf.cell(val_w, 5.5, _pdf_safe(value), fill=fill, new_x="LMARGIN", new_y="NEXT")
        fill = not fill
    pdf.ln(4)

    # ── Charts ──
    has_charts = False

    # Diameter chart
    fig_diam = plot_diameter(df, phases, kpi, meta.short_label, thresholds_to_show)
    fig_diam.update_layout(height=380, margin=dict(t=80, b=40, l=50, r=20))
    png_diam = _fig_to_png(fig_diam, width=900, height=380)
    if png_diam:
        if not has_charts:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 7, "Charts", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            has_charts = True
        img_diam = io.BytesIO(png_diam)
        img_diam.name = "diameter.png"
        pdf.image(img_diam, x=10, w=chart_w)
        pdf.ln(3)

    # Vol conc chart (new page if needed)
    if "vol_conc_mm3_L" in df.columns:
        fig_vc = plot_vol_conc(df, phases, kpi, meta.short_label)
        fig_vc.update_layout(height=380, margin=dict(t=80, b=40, l=50, r=20))
        png_vc = _fig_to_png(fig_vc, width=900, height=380)
        if png_vc:
            if not has_charts:
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(30, 41, 59)
                pdf.cell(0, 7, "Charts", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
                has_charts = True
            if pdf.get_y() > 200:
                pdf.add_page()
            img_vc = io.BytesIO(png_vc)
            img_vc.name = "volconc.png"
            pdf.image(img_vc, x=10, w=chart_w)


class _ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(148, 163, 184)
        self.cell(0, 5, "FlocBot Run Insights  |  Research / sample-based output. For evaluation purposes only.", align="C")


def generate_pdf(all_runs, thresholds_to_show):
    """Build a multi-page PDF report for all uploaded runs and return bytes."""

    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title block ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(37, 99, 235)  # #2563EB
    pdf.cell(0, 10, "FlocBot Run Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 120, pdf.get_y())
    pdf.ln(4)

    # ── Generated timestamp ──
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)  # #64748B
    n_runs = len(all_runs)
    pdf.cell(0, 5, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {n_runs} run{'s' if n_runs != 1 else ''}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    chart_w = 190  # mm, page width minus margins

    for run_i, run in enumerate(all_runs):
        if run_i > 0:
            pdf.add_page()
        _pdf_add_run(pdf, run, thresholds_to_show, chart_w)

    return bytes(pdf.output())


def _chart_card_open(title, subtitle=""):
    """Emit opening HTML for a chart card wrapper."""
    sub = f'<p class="chart-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="chart-card"><h5>{title}</h5>{sub}', unsafe_allow_html=True)

def _chart_card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════════


with root.container():
    st.markdown(f'## FlocBot Run Insights')
    st.caption(f'{len(runs)} run{"s" if len(runs) != 1 else ""} loaded')
    st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)

    with st.expander("About This Tool", expanded=False):
        st.markdown(
            "FlocBot Run Insights is a research tool under active development. "
            "Results are based on limited sample data and should be treated as "
            "exploratory, not as standalone guidance. All outputs should be "
            "independently verified before informing operational decisions."
        )

    run_labels = [r["meta"].label for r in runs]
    chart_labels = [r["meta"].short_label for r in runs]

    # ─── Best run ranking (used by Summary + Export tabs) ─────────────────────
    scored_runs = [(i, r) for i, r in enumerate(runs) if r["kpi"].score is not None]
    best_idx = 0
    if scored_runs:
        scored_runs.sort(key=lambda x: x[1]["kpi"].score, reverse=True)
        best_idx = scored_runs[0][0]

    # ─── Top-level navigation tabs ────────────────────────────────────────────
    nav_summary, nav_charts, nav_diagnostics, nav_export = st.tabs(
        ["Summary", "Charts", "Diagnostics", "Export"]
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB: Summary
    # ═══════════════════════════════════════════════════════════════════════════
    with nav_summary:

        with st.expander("How to read this page", expanded=False):
            st.markdown(
                "Each uploaded run is shown as a card with key performance metrics. "
                "Hover over any metric's **?** icon for a quick explanation.\n\n"
                "| Metric | What it tells you | Good value |\n"
                "|:--|:--|:--|\n"
                f"| **Score** | {METRIC_HELP['score']['short']} | {METRIC_HELP['score']['good']} |\n"
                f"| **Growth Rate** | {METRIC_HELP['growth_rate']['short']} | {METRIC_HELP['growth_rate']['good']} |\n"
                f"| **Pre-settle Ø** | {METRIC_HELP['pre_settle_diameter']['short']} | {METRIC_HELP['pre_settle_diameter']['good']} |\n"
                f"| **Settling t50** | {METRIC_HELP['settling_t50']['short']} | {METRIC_HELP['settling_t50']['good']} |\n"
                f"| **Signal Noise** | {METRIC_HELP['floc_noise_mad']['short']} | {METRIC_HELP['floc_noise_mad']['good']} |\n"
                f"| **Threshold times** | {METRIC_HELP['time_to_threshold']['short']} | {METRIC_HELP['time_to_threshold']['good']} |\n"
            )

        # Best run ranking
        if scored_runs:
            best_run = runs[best_idx]
            st.success(
                f"**Best run:** {best_run['meta'].label} — Score **{best_run['kpi'].score}**/100"
            )

        # ── Per-run KPI cards ──
        for run_idx, run in enumerate(runs):
            kpi = run["kpi"]
            meta = run["meta"]

            with st.container(border=True):
                st.markdown(f"#### {meta.label}")

                # Row 1: Performance
                st.markdown(
                    '<p class="metrics-row-label">Performance</p>'
                    '<p class="metrics-row-desc">Overall floc quality indicators</p>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Score", f"{kpi.score}/100" if kpi.score is not None else "N/A",
                               help=METRIC_HELP["score"]["short"])
                    pct = kpi.score if kpi.score is not None else 0
                    if pct >= 70:
                        bar_color = "#059669"
                    elif pct >= 40:
                        bar_color = "#D97706"
                    else:
                        bar_color = "#DC2626"
                    st.markdown(
                        f'<div class="score-bar-track">'
                        f'<div class="score-bar-fill" style="width:{pct}%;background:{bar_color};"></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                c2.metric("Growth Rate", f"{kpi.growth_rate_um_per_min} μm/min" if kpi.growth_rate_um_per_min else "N/A",
                           help=METRIC_HELP["growth_rate"]["short"])
                c3.metric("Pre-settle Ø", f"{kpi.pre_settle_diameter_um} μm" if kpi.pre_settle_diameter_um else "N/A",
                           help=METRIC_HELP["pre_settle_diameter"]["short"])
                c4.metric("Settling t50", f"{kpi.t50_min} min" if kpi.t50_min else "N/A",
                           help=METRIC_HELP["settling_t50"]["short"])

                # Row 2: Process Timing
                st.markdown(
                    '<p class="metrics-row-label">Process Timing</p>'
                    '<p class="metrics-row-desc">Duration of each mixing stage</p>',
                    unsafe_allow_html=True,
                )
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Rapid Mix", f"{kpi.rapid_mix_duration_min:.1f} min" if kpi.rapid_mix_duration_min else "N/A",
                           help=METRIC_HELP["rapid_mix_duration"]["short"])
                c6.metric("Flocculation", f"{kpi.flocculation_duration_min:.1f} min" if kpi.flocculation_duration_min else "N/A",
                           help=METRIC_HELP["flocculation_duration"]["short"])
                c7.metric("Settling", f"{kpi.settling_duration_min:.1f} min" if kpi.settling_duration_min else "N/A",
                           help=METRIC_HELP["settling_duration"]["short"])
                c8.metric("Signal Noise", f"{kpi.floc_noise_mad} μm" if kpi.floc_noise_mad else "N/A",
                           help=METRIC_HELP["floc_noise_mad"]["short"])

                # Row 3: Floc Growth Milestones
                if thresholds:
                    st.markdown(
                        '<p class="metrics-row-label">Floc Growth Milestones</p>'
                        '<p class="metrics-row-desc">Time to reach key floc sizes</p>',
                        unsafe_allow_html=True,
                    )
                    thr_display = thresholds[:5]
                    thr_cols = st.columns(len(thr_display))
                    for tc, thr in zip(thr_cols, thr_display):
                        val = kpi.time_to_thresholds_min.get(thr)
                        tc.metric(f"{int(thr)} μm", f"{val} min" if val is not None else "Not reached",
                                  help=METRIC_HELP["time_to_threshold"]["short"])


                if kpi.score_reason:
                    st.caption(kpi.score_reason)

        # ── Summary table ──
        st.markdown("---")
        st.markdown("#### Run Summary Table")
        summary_rows = [build_summary_row(r["meta"], r["kpi"]) for r in runs]
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "File": st.column_config.TextColumn(width="medium"),
                "Protocol": st.column_config.TextColumn(width="medium"),
                "Chemistry": st.column_config.TextColumn(width="small"),
                "Dosage": st.column_config.TextColumn(width="small"),
                "Plateau Mean (μm)": st.column_config.NumberColumn(width="small"),
                "Signal Noise (μm)": st.column_config.NumberColumn(width="small"),
                "Pre-settle Ø (μm)": st.column_config.NumberColumn(width="small"),
            },
        )

        # ── Multi-run comparison (if >= 2) ──
        if len(runs) >= 2:
            st.markdown("---")
            st.markdown("#### Run Comparison")
            col_a, col_b = st.columns(2)
            with col_a:
                idx_a = st.selectbox("Run A", range(len(runs)), format_func=lambda i: run_labels[i], key="cmp_a")
            with col_b:
                default_b = 1 if len(runs) > 1 else 0
                idx_b = st.selectbox("Run B", range(len(runs)), format_func=lambda i: run_labels[i], index=default_b, key="cmp_b")

            ra, rb = runs[idx_a], runs[idx_b]
            ka, kb = ra["kpi"], rb["kpi"]

            def _fmt(val, unit=""):
                if val is None:
                    return "N/A"
                return f"{val}{unit}"

            def _winner(va, vb, higher_better=True):
                if va is None or vb is None:
                    return "", ""
                if higher_better:
                    return ("**✓**" if va > vb else "", "**✓**" if vb > va else "")
                return ("**✓**" if va < vb else "", "**✓**" if vb < va else "")

            comp_metrics = [
                ("Score", ka.score, kb.score, "/100", True),
                ("Growth Rate", ka.growth_rate_um_per_min, kb.growth_rate_um_per_min, " μm/min", True),
                ("Pre-settle Ø", ka.pre_settle_diameter_um, kb.pre_settle_diameter_um, " μm", True),
                ("Signal Noise", ka.floc_noise_mad, kb.floc_noise_mad, " μm", False),
                ("Settling t50", ka.t50_min, kb.t50_min, " min", False),
                ("Settling t90", ka.t10_min, kb.t10_min, " min", False),
            ]
            for thr in thresholds:
                va = ka.time_to_thresholds_min.get(thr)
                vb = kb.time_to_thresholds_min.get(thr)
                comp_metrics.append((f"t_{int(thr)}μm", va, vb, " min", False))

            comp_rows = []
            for label, va, vb, unit, hb in comp_metrics:
                wa, wb = _winner(va, vb, hb)
                comp_rows.append({
                    "Metric": label,
                    f"Run A ({run_labels[idx_a]})": f"{_fmt(va, unit)} {wa}",
                    f"Run B ({run_labels[idx_b]})": f"{_fmt(vb, unit)} {wb}",
                })
            st.table(pd.DataFrame(comp_rows))

            # Natural language summary
            summaries = []
            if ka.growth_rate_um_per_min is not None and kb.growth_rate_um_per_min is not None:
                if ka.growth_rate_um_per_min > kb.growth_rate_um_per_min:
                    summaries.append("Run A forms floc faster")
                else:
                    summaries.append("Run B forms floc faster")
            if ka.pre_settle_diameter_um is not None and kb.pre_settle_diameter_um is not None:
                if ka.pre_settle_diameter_um > kb.pre_settle_diameter_um:
                    summaries.append("Run A achieves higher pre-settle size")
                else:
                    summaries.append("Run B achieves higher pre-settle size")
            if ka.t50_min is not None and kb.t50_min is not None:
                if ka.t50_min < kb.t50_min:
                    summaries.append("Run A settles faster")
                else:
                    summaries.append("Run B settles faster")
            if ka.score is not None and kb.score is not None:
                if ka.score > kb.score:
                    summaries.append(f"overall score favors Run A ({ka.score} vs {kb.score})")
                elif kb.score > ka.score:
                    summaries.append(f"overall score favors Run B ({kb.score} vs {ka.score})")
                else:
                    summaries.append("overall scores are tied")
            if summaries:
                st.info("; ".join(summaries) + ".")


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB: Charts
    # ═══════════════════════════════════════════════════════════════════════════
    with nav_charts:

        # ── Per-run charts ──
        if len(runs) == 1:
            run = runs[0]
            df, meta, phases, kpi = run["df"], run["meta"], run["phases"], run["kpi"]
            _chart_card_open("Floc Diameter", "Mean particle diameter over the run duration")
            st.plotly_chart(plot_diameter(df, phases, kpi, meta.short_label, thresholds), use_container_width=True)
            _chart_card_close()
            if "vol_conc_mm3_L" in df.columns:
                _chart_card_open("Volume Concentration", "Particle volume concentration over time")
                st.plotly_chart(plot_vol_conc(df, phases, kpi, meta.short_label), use_container_width=True)
                _chart_card_close()
        else:
            chart_run_tabs = st.tabs(run_labels + ["Overlay"])

            for tab, run in zip(chart_run_tabs[:-1], runs):
                df, meta, phases, kpi = run["df"], run["meta"], run["phases"], run["kpi"]
                with tab:
                    _chart_card_open("Floc Diameter", "Mean particle diameter over the run duration")
                    st.plotly_chart(plot_diameter(df, phases, kpi, meta.short_label, thresholds), use_container_width=True)
                    _chart_card_close()
                    if "vol_conc_mm3_L" in df.columns:
                        _chart_card_open("Volume Concentration", "Particle volume concentration over time")
                        st.plotly_chart(plot_vol_conc(df, phases, kpi, meta.short_label), use_container_width=True)
                        _chart_card_close()

            # Overlay tab
            with chart_run_tabs[-1]:
                _chart_card_open("Overlay: Diameter", "All runs compared on a single axis")
                fig_cmp = go.Figure()
                colors = ["#0284C7", "#E11D48", "#059669", "#D97706", "#7C3AED", "#DB2777"]
                for i, run in enumerate(runs):
                    color = colors[i % len(colors)]
                    fig_cmp.add_trace(go.Scatter(
                        x=run["df"]["time_min"], y=run["df"].get("diameter_um"),
                        mode="lines+markers", marker=dict(size=3, color=color),
                        name=chart_labels[i],
                    ))
                add_phase_shading(fig_cmp, runs[0]["phases"])
                fig_cmp.update_layout(
                    xaxis_title="Time (min)", yaxis_title="Mean Diameter (μm)",
                    **CHART_LAYOUT,
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
                _chart_card_close()

                # Vol conc overlay
                vol_runs = [r for r in runs if "vol_conc_mm3_L" in r["df"].columns]
                if len(vol_runs) >= 2:
                    _chart_card_open("Overlay: Vol. Concentration", "Volume concentration comparison across runs")
                    fig_vc = go.Figure()
                    for i, run in enumerate(vol_runs):
                        color = colors[i % len(colors)]
                        fig_vc.add_trace(go.Scatter(
                            x=run["df"]["time_min"], y=run["df"]["vol_conc_mm3_L"],
                            mode="lines+markers", marker=dict(size=3, color=color),
                            name=run["meta"].short_label,
                        ))
                    add_phase_shading(fig_vc, vol_runs[0]["phases"])
                    fig_vc.update_layout(
                        xaxis_title="Time (min)", yaxis_title="Vol. Concentration (mm³/L)",
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig_vc, use_container_width=True)
                    _chart_card_close()


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB: Diagnostics
    # ═══════════════════════════════════════════════════════════════════════════
    with nav_diagnostics:

        for run in runs:
            df, meta, phases, kpi = run["df"], run["meta"], run["phases"], run["kpi"]

            with st.container(border=True):
                st.markdown(f"#### {meta.label}")

                # Classify flags by severity
                if kpi.quality_flags:
                    classified = [classify_flag(f) for f in kpi.quality_flags]

                    # Group by severity
                    severity_order = ["warning", "caution", "info"]
                    severity_labels = {"warning": "Warning", "caution": "Caution", "info": "Info"}
                    severity_st = {"warning": st.error, "caution": st.warning, "info": st.info}

                    for sev in severity_order:
                        items = [c for c in classified if c["severity"] == sev]
                        if not items:
                            continue
                        for item in items:
                            severity_st[sev](
                                f"{item['icon']} **{severity_labels[sev]}:** {item['summary']}"
                            )
                            st.caption(f"**What to do:** {item['action']}")
                else:
                    st.success("No data quality issues detected — all checks passed.")

                # Floc count chart
                fc_fig = plot_floc_count(df, phases, meta.short_label)
                if fc_fig:
                    _chart_card_open("Floc Count", "Particle count per mL — useful for detecting low-count artifacts")
                    st.plotly_chart(fc_fig, use_container_width=True)
                    _chart_card_close()

                # Metadata
                with st.expander("Metadata"):
                    md_rows = [
                        ("File", meta.filename),
                        ("Generated", meta.generated_timestamp or "—"),
                        ("Protocol", meta.protocol_title or "—"),
                        ("Chemistry", meta.run_chemistry or "—"),
                        ("Dosage", meta.run_dosage or "—"),
                        ("Comments", meta.comments or "—"),
                    ]
                    md_table = "| Field | Value |\n|:--|:--|\n"
                    md_table += "\n".join(f"| {k} | {v} |" for k, v in md_rows)
                    st.markdown(md_table)
                    if meta.warnings:
                        st.caption("**Warnings:** " + "; ".join(meta.warnings))

                    # Import debug — sheet selection details (collapsible)
                    dbg = getattr(meta, "import_debug", None)
                    if dbg:
                        with st.expander("Import details", expanded=False):
                            st.caption(f"Engine: `{dbg.get('engine', '?')}`")
                            st.caption(f"Sheets found: {dbg.get('sheets_found', [])}")
                            st.caption(f"Chosen sheet: `{dbg.get('chosen_sheet', '?')}`")
                            scores = dbg.get("sheet_scores", {})
                            for sn, info in scores.items():
                                marker = "✅" if info.get("valid") else "❌"
                                st.caption(
                                    f"{marker} **{sn}** — "
                                    f"required: {info.get('required_matched', [])} | "
                                    f"missing: {info.get('missing_required', [])} | "
                                    f"desired: {len(info.get('desired_matched', []))} cols | "
                                    f"numeric rows: {info.get('numeric_rows', 0)}"
                                )


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB: Export
    # ═══════════════════════════════════════════════════════════════════════════
    with nav_export:

        st.markdown("#### Download Results")
        st.caption("Export the summary table or full KPI data for reporting.")

        col_csv, col_json, col_pdf = st.columns(3)

        with col_csv:
            csv_buf = summary_df.to_csv(index=False)
            st.download_button(
                "Download summary table.csv",
                csv_buf,
                file_name="summary_table.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_json:
            all_kpis = [kpi_to_dict(r["meta"], r["kpi"]) for r in runs]
            json_buf = json.dumps(all_kpis, indent=2, default=str)
            st.download_button(
                "Download all runs.json",
                json_buf,
                file_name="all_runs.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_pdf:
            try:
                pdf_bytes = generate_pdf(runs, thresholds)
            except Exception as e:
                pdf_bytes = None
                st.warning(f"PDF generation failed: {e}")
            if pdf_bytes:
                st.download_button(
                    "Download report pdf",
                    pdf_bytes,
                    file_name="report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
