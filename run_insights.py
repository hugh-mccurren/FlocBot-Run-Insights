"""
FlocBot Run Insights – Streamlit application
Operator-friendly run summaries and comparisons for RoboJar/FlocBot exports.
"""

import io
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF

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

# ─── Custom CSS for card styling ──────────────────────────────────────────
st.markdown("""
<style>
    /* Compact header */
    .block-container { padding-top: 2.2rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #64748B;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E293B;
    }

    /* Score progress bar – sits inside the Score column, right under the metric */
    .score-bar-track {
        background: #E2E8F0;
        border-radius: 6px;
        height: 8px;
        margin-top: -8px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    /* Chart card wrappers */
    .chart-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px 16px 8px 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .chart-card h5 {
        margin: 0 0 2px 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E293B;
    }
    .chart-card .chart-subtitle {
        margin: 0 0 8px 0;
        font-size: 0.78rem;
        color: #94A3B8;
    }

    /* Section dividers */
    hr { border-color: #E2E8F0; margin: 1.5rem 0; }

    /* Branded header accent */
    .header-accent {
        height: 3px;
        background: linear-gradient(90deg, #2563EB 0%, #60A5FA 50%, transparent 100%);
        border-radius: 2px;
        margin-bottom: 12px;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: #F8FAFC;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-top: 0.75rem;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

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

CHART_HEIGHT = 520
CHART_LAYOUT = dict(
    template="plotly_white",
    height=CHART_HEIGHT,
    margin=dict(t=110, b=48, l=56, r=24),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.06,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
    ),
    font=dict(family="Inter, system-ui, sans-serif", size=12),
)

# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("FlocBot Run Insights")
    st.caption("Upload RoboJar exports to analyze flocculation performance.")

    uploaded_files = st.file_uploader(
        "Upload RoboJar exports",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
    )

    st.markdown("---")

    # ── Thresholds ──
    st.markdown("### Diameter Thresholds")
    default_thresholds = "250, 300, 350, 400, 450"
    threshold_str = st.text_input(
        "Thresholds (μm)",
        default_thresholds,
        help="Comma-separated diameter values for time-to-threshold metrics.",
    )
    st.caption("Comma-separated values in μm.")
    try:
        thresholds = [float(t.strip()) for t in threshold_str.split(",") if t.strip()]
    except ValueError:
        thresholds = [250, 300, 350, 400, 450]
        st.warning("Invalid thresholds – using defaults.")

    st.markdown("---")

    # ── Scoring weights (collapsed) ──
    with st.expander("Advanced: Scoring Weights", expanded=False):
        st.caption("Adjust relative importance of each metric.")
        w_time = st.slider("Time to threshold", 0, 100, 30, key="w1")
        w_diam = st.slider("Pre-settle diameter", 0, 100, 30, key="w2")
        w_cv = st.slider("Plateau stability (CV)", 0, 100, 20, key="w3")
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

        if st.button("Reset defaults", use_container_width=True):
            for key in ["w1", "w2", "w3", "w4"]:
                st.session_state.pop(key, None)
            st.rerun()

    total_w = w_time + w_diam + w_cv + w_t50
    if total_w == 0:
        total_w = 1  # avoid div-by-zero
    weights = {
        "time_to_300": w_time / total_w,
        "pre_settle_diameter": w_diam / total_w,
        "plateau_cv": w_cv / total_w,
        "settling_t50": w_t50 / total_w,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Processing
# ═══════════════════════════════════════════════════════════════════════════

def parse_uploads(files):
    """Parse all uploaded files (I/O + phase detection).
    Returns list of dicts with keys: df, meta, phases; and a list of errors."""
    parsed = []
    errors = []
    for f in files:
        try:
            f.seek(0)
            df, meta = parse_file(f)
            phases = detect_phases(df)
            parsed.append({"df": df, "meta": meta, "phases": phases})
        except Exception as e:
            errors.append((getattr(f, "name", "?"), str(e)))
    return parsed, errors


if not uploaded_files:
    st.markdown("## Welcome to FlocBot Run Insights")
    st.caption("Upload one or more RoboJar Excel exports using the sidebar to get started.")
    st.markdown(
        """
        - Automatic phase detection (Rapid Mix / Flocculation / Settling)
        - Operator-relevant KPIs and an overall run score
        - Interactive Plotly charts with phase shading
        - Multi-run comparison with side-by-side metrics
        - CSV / JSON export for reporting
        """
    )
    st.stop()

# --- Parse then compute KPIs with current sidebar values ---
parsed, errors = parse_uploads(uploaded_files)

runs = []
for p in parsed:
    kpi = compute_kpis(p["df"], p["phases"], thresholds=thresholds)
    compute_score(kpi, weights=dict(weights), threshold_for_time=score_threshold)
    runs.append({"df": p["df"], "meta": p["meta"], "phases": p["phases"], "kpi": kpi})

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
        fig.add_vline(x=se.start_min, line_dash="dash", line_color="blue")
        fig.add_annotation(
            x=se.start_min, y=1, yref="paper",
            xanchor="right", yanchor="top",
            text="Settle start", showarrow=False,
            xshift=-6,
            font=dict(size=11, color="blue"),
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


def _fig_to_png(fig, width=900, height=400):
    """Render a Plotly figure to PNG bytes via kaleido."""
    return fig.to_image(format="png", width=width, height=height, scale=2)


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
        ("Growth Rate", f"{kpi.growth_rate_um_per_min} um/min" if kpi.growth_rate_um_per_min else "N/A"),
        ("Pre-settle O", f"{kpi.pre_settle_diameter_um} um" if kpi.pre_settle_diameter_um else "N/A"),
        ("Plateau Mean", f"{kpi.plateau_mean_um} um" if kpi.plateau_mean_um else "N/A"),
        ("Plateau CV", f"{kpi.plateau_cv}%" if kpi.plateau_cv else "N/A"),
        ("Settling t50", f"{kpi.t50_min} min" if kpi.t50_min else "N/A"),
        ("Settling t10", f"{kpi.t10_min} min" if kpi.t10_min else "N/A"),
        ("Rapid Mix", f"{kpi.rapid_mix_duration_min:.1f} min" if kpi.rapid_mix_duration_min else "N/A"),
        ("Flocculation", f"{kpi.flocculation_duration_min:.1f} min" if kpi.flocculation_duration_min else "N/A"),
        ("Settling", f"{kpi.settling_duration_min:.1f} min" if kpi.settling_duration_min else "N/A"),
    ]
    # Add threshold times
    for thr in sorted(thresholds_to_show):
        val = kpi.time_to_thresholds_min.get(thr)
        kpi_rows.append((f"t {int(thr)} um", f"{val} min" if val is not None else "Not reached"))

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
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "Charts", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Diameter chart
    fig_diam = plot_diameter(df, phases, kpi, meta.short_label, thresholds_to_show)
    fig_diam.update_layout(height=380, margin=dict(t=80, b=40, l=50, r=20))
    png_diam = _fig_to_png(fig_diam, width=900, height=380)
    img_diam = io.BytesIO(png_diam)
    img_diam.name = "diameter.png"
    pdf.image(img_diam, x=10, w=chart_w)
    pdf.ln(3)

    # Vol conc chart (new page if needed)
    if "vol_conc_mm3_L" in df.columns:
        if pdf.get_y() > 200:
            pdf.add_page()
        fig_vc = plot_vol_conc(df, phases, kpi, meta.short_label)
        fig_vc.update_layout(height=380, margin=dict(t=80, b=40, l=50, r=20))
        png_vc = _fig_to_png(fig_vc, width=900, height=380)
        img_vc = io.BytesIO(png_vc)
        img_vc.name = "volconc.png"
        pdf.image(img_vc, x=10, w=chart_w)


def generate_pdf(all_runs, thresholds_to_show):
    """Build a multi-page PDF report for all uploaded runs and return bytes."""

    pdf = FPDF(orientation="P", unit="mm", format="A4")
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

    # Footer on last page
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(148, 163, 184)  # #94A3B8
    pdf.cell(0, 5, "FlocBot Run Insights", align="C")

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

st.markdown(f'## FlocBot Run Insights')
st.caption(f'{len(runs)} run{"s" if len(runs) != 1 else ""} loaded')
st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)

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

            # Row 1: Score, Growth Rate, Pre-settle Ø, Settling t50
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Score", f"{kpi.score}/100" if kpi.score is not None else "N/A")
                pct = kpi.score if kpi.score is not None else 0
                if pct >= 70:
                    bar_color = "#10B981"
                elif pct >= 40:
                    bar_color = "#F59E0B"
                else:
                    bar_color = "#EF4444"
                st.markdown(
                    f'<div class="score-bar-track">'
                    f'<div class="score-bar-fill" style="width:{pct}%;background:{bar_color};"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            c2.metric("Growth Rate", f"{kpi.growth_rate_um_per_min} μm/min" if kpi.growth_rate_um_per_min else "N/A")
            c3.metric("Pre-settle Ø", f"{kpi.pre_settle_diameter_um} μm" if kpi.pre_settle_diameter_um else "N/A")
            c4.metric("Settling t50", f"{kpi.t50_min} min" if kpi.t50_min else "N/A")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Rapid Mix", f"{kpi.rapid_mix_duration_min:.1f} min" if kpi.rapid_mix_duration_min else "N/A")
            c6.metric("Flocculation", f"{kpi.flocculation_duration_min:.1f} min" if kpi.flocculation_duration_min else "N/A")
            c7.metric("Settling", f"{kpi.settling_duration_min:.1f} min" if kpi.settling_duration_min else "N/A")
            c8.metric("Plateau CV", f"{kpi.plateau_cv}%" if kpi.plateau_cv else "N/A")

            # Threshold times
            if thresholds:
                thr_cols = st.columns(len(thresholds))
                for tc, thr in zip(thr_cols, thresholds):
                    val = kpi.time_to_thresholds_min.get(thr)
                    tc.metric(f"{int(thr)} μm", f"{val} min" if val is not None else "Not reached")

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
            "Plateau CV (%)": st.column_config.NumberColumn(width="small"),
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
            colors = ["#2563EB", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
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

            # Quality flags
            if kpi.quality_flags:
                for flag in kpi.quality_flags:
                    st.warning(flag)
            else:
                st.caption("No data quality issues detected.")

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
            "Download summary_table.csv",
            csv_buf,
            file_name="summary_table.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_json:
        all_kpis = [kpi_to_dict(r["meta"], r["kpi"]) for r in runs]
        json_buf = json.dumps(all_kpis, indent=2, default=str)
        st.download_button(
            "Download all_runs.json",
            json_buf,
            file_name="all_runs.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_pdf:
        pdf_bytes = generate_pdf(runs, thresholds)
        st.download_button(
            "Download report.pdf",
            pdf_bytes,
            file_name="report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
