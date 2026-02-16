"""
FlocBot Run Insights – Streamlit application
Operator-friendly run summaries and comparisons for RoboJar/FlocBot exports.
"""

import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flocbot_parser import parse_file, RunMetadata
from flocbot_metrics import (
    detect_phases,
    compute_kpis,
    compute_score,
    Phase,
    RunKPIs,
    DEFAULT_WEIGHTS,
    phase_by_name,
)

# ─── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlocBot Run Insights",
    page_icon="🔬",
    layout="wide",
)

PHASE_COLORS = {
    "rapid_mix": "rgba(255, 99, 71, 0.12)",
    "flocculation": "rgba(60, 179, 113, 0.12)",
    "settling": "rgba(100, 149, 237, 0.12)",
}
PHASE_LABELS = {
    "rapid_mix": "Rapid Mix",
    "flocculation": "Flocculation",
    "settling": "Settling",
}

# ─── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("FlocBot Run Insights")
uploaded_files = st.sidebar.file_uploader(
    "Upload RoboJar exports",
    type=["xls", "xlsx"],
    accept_multiple_files=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")

# Thresholds
default_thresholds = "250, 300, 350, 400, 450"
threshold_str = st.sidebar.text_input("Diameter thresholds (μm)", default_thresholds)
try:
    thresholds = [float(t.strip()) for t in threshold_str.split(",") if t.strip()]
except ValueError:
    thresholds = [250, 300, 350, 400, 450]
    st.sidebar.warning("Invalid thresholds – using defaults.")

# Scoring weights
st.sidebar.subheader("Score weights")
w_time = st.sidebar.slider("Time to threshold", 0, 100, 30, key="w1")
w_diam = st.sidebar.slider("Pre-settle diameter", 0, 100, 30, key="w2")
w_cv = st.sidebar.slider("Plateau stability (CV)", 0, 100, 20, key="w3")
w_t50 = st.sidebar.slider("Settling t50", 0, 100, 20, key="w4")

total_w = w_time + w_diam + w_cv + w_t50
if total_w == 0:
    total_w = 1  # avoid div-by-zero
weights = {
    "time_to_300": w_time / total_w,
    "pre_settle_diameter": w_diam / total_w,
    "plateau_cv": w_cv / total_w,
    "settling_t50": w_t50 / total_w,
}

score_threshold = 300.0
score_thr_options = [t for t in thresholds if t > 0]
if score_thr_options:
    score_threshold = st.sidebar.selectbox(
        "Threshold for scoring (μm)", score_thr_options, index=min(1, len(score_thr_options) - 1)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Processing
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Parsing files …")
def process_uploads(_files, _thresholds, _weights, _score_thr):
    """Parse all uploaded files and compute KPIs.
    Returns list of dicts with keys: df, meta, phases, kpi."""
    runs = []
    errors = []
    for f in _files:
        try:
            f.seek(0)
            df, meta = parse_file(f)
            phases = detect_phases(df)
            kpi = compute_kpis(df, phases, thresholds=_thresholds)
            compute_score(kpi, weights=dict(_weights), threshold_for_time=_score_thr)
            runs.append({"df": df, "meta": meta, "phases": phases, "kpi": kpi})
        except Exception as e:
            errors.append((getattr(f, "name", "?"), str(e)))
    return runs, errors


if not uploaded_files:
    st.markdown(
        """
        ## Welcome to FlocBot Run Insights
        Upload one or more RoboJar Excel exports using the sidebar to get started.

        **Features:**
        - Automatic phase detection (Rapid Mix → Flocculation → Settling)
        - Operator-relevant KPIs and an overall run score
        - Interactive Plotly charts with phase shading
        - Multi-run comparison with side-by-side metrics
        - CSV / JSON export for reporting
        """
    )
    st.stop()

# --- Process ---
runs, errors = process_uploads(
    uploaded_files,
    tuple(thresholds),
    tuple(sorted(weights.items())),
    score_threshold,
)

for fname, err in errors:
    st.error(f"**{fname}:** {err}")

if not runs:
    st.warning("No valid runs were parsed.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# Helpers: plotting
# ═══════════════════════════════════════════════════════════════════════════

def add_phase_shading(fig, phases: list[Phase], row=None, col=None):
    for p in phases:
        fig.add_vrect(
            x0=p.start_min, x1=p.end_min,
            fillcolor=PHASE_COLORS.get(p.name, "rgba(200,200,200,0.1)"),
            layer="below", line_width=0,
            annotation_text=PHASE_LABELS.get(p.name, p.name),
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="gray",
            row=row, col=col,
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
        fig.add_vline(x=se.start_min, line_dash="dash", line_color="blue",
                      annotation_text="Settle start", annotation_position="top right")

    fig.update_layout(
        title="Mean Floc Diameter vs Time",
        xaxis_title="Time (min)", yaxis_title="Mean Diameter (μm)",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
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
        title="Vol. Concentration vs Time",
        xaxis_title="Time (min)", yaxis_title="Vol. Concentration (mm³/L)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
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
        title="Floc Count vs Time",
        xaxis_title="Time (min)", yaxis_title="Floc Count (per mL)",
        template="plotly_white", height=350,
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
        "Plateau CV (%)": kpi.plateau_cv,
        "t50 (min)": kpi.t50_min,
        "t10 (min)": kpi.t10_min,
        "Score": kpi.score,
    }
    for thr, val in sorted(kpi.time_to_thresholds_min.items()):
        row[f"t_{int(thr)}μm (min)"] = val
    return row


def kpi_to_dict(meta, kpi):
    """Full KPI dict for JSON export."""
    d = {
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
        "plateau_cv": kpi.plateau_cv,
        "settle_baseline_vol_conc": kpi.settle_baseline_vol_conc,
        "t50_min": kpi.t50_min,
        "t10_min": kpi.t10_min,
        "time_to_thresholds_min": {str(k): v for k, v in kpi.time_to_thresholds_min.items()},
        "score": kpi.score,
        "score_components": kpi.score_components,
        "score_reason": kpi.score_reason,
        "quality_flags": kpi.quality_flags,
    }
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════════

st.title("FlocBot Run Insights")

# ─── Summary table ────────────────────────────────────────────────────────
st.header("Run Summary Table")
summary_rows = [build_summary_row(r["meta"], r["kpi"]) for r in runs]
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

# Best run ranking
scored_runs = [(i, r) for i, r in enumerate(runs) if r["kpi"].score is not None]
if scored_runs:
    scored_runs.sort(key=lambda x: x[1]["kpi"].score, reverse=True)
    best_idx, best_run = scored_runs[0]
    st.success(
        f"**Best run:** {best_run['meta'].label} — Score **{best_run['kpi'].score}**/100"
    )

# ─── Tabs: individual runs ───────────────────────────────────────────────
st.header("Individual Runs")
run_labels = [r["meta"].label for r in runs]
tabs = st.tabs(run_labels)

for tab, run in zip(tabs, runs):
    df, meta, phases, kpi = run["df"], run["meta"], run["phases"], run["kpi"]
    with tab:
        # --- KPI summary card ---
        st.subheader("Run Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{kpi.score}/100" if kpi.score is not None else "N/A")
        c2.metric("Growth Rate", f"{kpi.growth_rate_um_per_min} μm/min" if kpi.growth_rate_um_per_min else "N/A")
        c3.metric("Pre-settle Ø", f"{kpi.pre_settle_diameter_um} μm" if kpi.pre_settle_diameter_um else "N/A")
        c4.metric("Settling t50", f"{kpi.t50_min} min" if kpi.t50_min else "N/A")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Rapid Mix", f"{kpi.rapid_mix_duration_min:.1f} min" if kpi.rapid_mix_duration_min else "N/A")
        c6.metric("Flocculation", f"{kpi.flocculation_duration_min:.1f} min" if kpi.flocculation_duration_min else "N/A")
        c7.metric("Settling", f"{kpi.settling_duration_min:.1f} min" if kpi.settling_duration_min else "N/A")
        c8.metric("Plateau CV", f"{kpi.plateau_cv}%" if kpi.plateau_cv else "N/A")

        # Threshold times
        st.markdown("**Time to thresholds:**")
        thr_cols = st.columns(len(thresholds))
        for tc, thr in zip(thr_cols, thresholds):
            val = kpi.time_to_thresholds_min.get(thr)
            tc.metric(f"{int(thr)} μm", f"{val} min" if val is not None else "Not reached")

        # Quality flags
        if kpi.quality_flags:
            with st.expander("Data quality flags"):
                for flag in kpi.quality_flags:
                    st.warning(flag)
        if kpi.score_reason:
            st.caption(kpi.score_reason)

        # --- Plots ---
        st.plotly_chart(plot_diameter(df, phases, kpi, meta.label, thresholds), use_container_width=True)

        if "vol_conc_mm3_L" in df.columns:
            st.plotly_chart(plot_vol_conc(df, phases, kpi, meta.label), use_container_width=True)

        fc_fig = plot_floc_count(df, phases, meta.label)
        if fc_fig:
            with st.expander("Floc Count (diagnostic)"):
                st.plotly_chart(fc_fig, use_container_width=True)

        # Metadata
        with st.expander("Metadata"):
            st.json({
                "filename": meta.filename,
                "generated": meta.generated_timestamp,
                "protocol": meta.protocol_title,
                "chemistry": meta.run_chemistry,
                "dosage": meta.run_dosage,
                "comments": meta.comments,
                "warnings": meta.warnings,
            })

# ═══════════════════════════════════════════════════════════════════════════
# Multi-run comparison
# ═══════════════════════════════════════════════════════════════════════════

if len(runs) >= 2:
    st.header("Run Comparison")
    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = st.selectbox("Run A", range(len(runs)), format_func=lambda i: run_labels[i], key="cmp_a")
    with col_b:
        default_b = 1 if len(runs) > 1 else 0
        idx_b = st.selectbox("Run B", range(len(runs)), format_func=lambda i: run_labels[i], index=default_b, key="cmp_b")

    ra, rb = runs[idx_a], runs[idx_b]
    ka, kb = ra["kpi"], rb["kpi"]

    # ── Side-by-side KPIs ──
    st.subheader("KPI Comparison")

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
        ("Plateau CV", ka.plateau_cv, kb.plateau_cv, "%", False),
        ("Settling t50", ka.t50_min, kb.t50_min, " min", False),
        ("Settling t10", ka.t10_min, kb.t10_min, " min", False),
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

    # ── Natural language summary ──
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

    # ── Overlay plots ──
    st.subheader("Overlay: Diameter")
    fig_cmp = go.Figure()
    for idx, run, color in [(idx_a, ra, "royalblue"), (idx_b, rb, "tomato")]:
        fig_cmp.add_trace(go.Scatter(
            x=run["df"]["time_min"], y=run["df"].get("diameter_um"),
            mode="lines+markers", marker=dict(size=3, color=color),
            name=run_labels[idx],
        ))
    # Phase shading from run A
    add_phase_shading(fig_cmp, ra["phases"])
    fig_cmp.update_layout(
        xaxis_title="Time (min)", yaxis_title="Mean Diameter (μm)",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    if "vol_conc_mm3_L" in ra["df"].columns and "vol_conc_mm3_L" in rb["df"].columns:
        st.subheader("Overlay: Vol. Concentration")
        fig_vc = go.Figure()
        for idx, run, color in [(idx_a, ra, "royalblue"), (idx_b, rb, "tomato")]:
            fig_vc.add_trace(go.Scatter(
                x=run["df"]["time_min"], y=run["df"]["vol_conc_mm3_L"],
                mode="lines+markers", marker=dict(size=3, color=color),
                name=run_labels[idx],
            ))
        add_phase_shading(fig_vc, ra["phases"])
        fig_vc.update_layout(
            xaxis_title="Time (min)", yaxis_title="Vol. Concentration (mm³/L)",
            template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_vc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

st.header("Export")

col_csv, col_json = st.columns(2)

with col_csv:
    csv_buf = summary_df.to_csv(index=False)
    st.download_button(
        "Download summary_table.csv",
        csv_buf,
        file_name="summary_table.csv",
        mime="text/csv",
    )

with col_json:
    all_kpis = [kpi_to_dict(r["meta"], r["kpi"]) for r in runs]
    json_buf = json.dumps(all_kpis, indent=2, default=str)
    st.download_button(
        "Download all_runs.json",
        json_buf,
        file_name="all_runs.json",
        mime="application/json",
    )
