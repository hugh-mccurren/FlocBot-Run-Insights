"""
ui_operator.py – Operator Mode dashboard for FlocBot Run Insights.

Provides a simplified, traffic-light-based view of run quality
with recommended actions. Reuses existing KPI pipeline outputs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Metric selector configuration – single source of truth
# ═══════════════════════════════════════════════════════════════════════════

OPERATOR_METRICS = {
    "diameter": {
        "display_name": "Mean Floc Diameter",
        "column": "diameter_um",
        "y_axis_label": "Mean Diameter (\u00b5m)",
        "chart_title": "Mean Floc Diameter vs Time",
        "tooltip_template": "Time: %{x:.1f} min<br>Diameter: %{y:.0f} \u00b5m<extra></extra>",
        "legend_label_suffix": "",
    },
    "vol_conc": {
        "display_name": "Volume Concentration",
        "column": "vol_conc_mm3_L",
        "y_axis_label": "Vol. Concentration (mm\u00b3/L)",
        "chart_title": "Volume Concentration vs Time",
        "tooltip_template": "Time: %{x:.1f} min<br>Vol. Conc: %{y:.1f} mm\u00b3/L<extra></extra>",
        "legend_label_suffix": "",
    },
}

_DEFAULT_METRIC = "diameter"


# ═══════════════════════════════════════════════════════════════════════════
# Thresholds for traffic-light evaluation
# ═══════════════════════════════════════════════════════════════════════════

_FORMATION_THRESHOLDS = {
    "growth_good": 20.0,       # μm/min – green if >=
    "growth_poor": 5.0,        # μm/min – red if <
    "time_good": 8.0,          # min to 300 μm – green if <=
    "time_poor": 15.0,         # min to 300 μm – red if >
}

_SIZE_THRESHOLDS = {
    "good": 400.0,   # μm – green if >=
    "poor": 200.0,   # μm – red if <
}

_SETTLING_THRESHOLDS = {
    "good": 3.0,     # min t50 – green if <=
    "poor": 8.0,     # min t50 – red if >
}


# ═══════════════════════════════════════════════════════════════════════════
# Stage evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StageResult:
    status: str          # "good", "mixed", "poor"
    icon: str            # traffic-light emoji
    value_text: str      # formatted key number
    message: str         # short explanation


def _status_icon(status: str) -> str:
    return {"good": "\U0001f7e2", "mixed": "\U0001f7e1", "poor": "\U0001f534"}[status]


def evaluate_formation(kpi) -> StageResult:
    gr = kpi.growth_rate_um_per_min
    t300 = kpi.time_to_thresholds_min.get(300)

    if gr is None:
        return StageResult("poor", _status_icon("poor"), "N/A",
                           "Growth rate could not be measured")

    gr_good = gr >= _FORMATION_THRESHOLDS["growth_good"]
    gr_poor = gr < _FORMATION_THRESHOLDS["growth_poor"]

    if t300 is not None:
        t_good = t300 <= _FORMATION_THRESHOLDS["time_good"]
        t_poor = t300 > _FORMATION_THRESHOLDS["time_poor"]
    else:
        t_good = False
        t_poor = True  # threshold never reached

    if gr_good and t_good:
        status = "good"
        msg = f"Fast growth ({gr:.0f} \u00b5m/min) and quick threshold reach ({t300:.1f} min)"
    elif gr_poor or t_poor:
        status = "poor"
        if gr_poor:
            msg = f"Slow growth rate ({gr:.0f} \u00b5m/min)"
        else:
            msg = "300 \u00b5m threshold not reached in time" if t300 is None else f"Slow to reach 300 \u00b5m ({t300:.1f} min)"
    else:
        status = "mixed"
        parts = [f"Growth {gr:.0f} \u00b5m/min"]
        if t300 is not None:
            parts.append(f"300 \u00b5m at {t300:.1f} min")
        msg = ", ".join(parts)

    val = f"{gr:.0f} \u00b5m/min"
    return StageResult(status, _status_icon(status), val, msg)


def evaluate_floc_size(kpi) -> StageResult:
    d = kpi.pre_settle_diameter_um
    if d is None:
        return StageResult("poor", _status_icon("poor"), "N/A",
                           "Pre-settle diameter not available")

    if d >= _SIZE_THRESHOLDS["good"]:
        status = "good"
        msg = f"Large flocs ({d:.0f} \u00b5m) before settling"
    elif d < _SIZE_THRESHOLDS["poor"]:
        status = "poor"
        msg = f"Small flocs ({d:.0f} \u00b5m) \u2013 may settle slowly"
    else:
        status = "mixed"
        msg = f"Moderate floc size ({d:.0f} \u00b5m)"

    return StageResult(status, _status_icon(status), f"{d:.0f} \u00b5m", msg)


def evaluate_settling(kpi) -> StageResult:
    t50 = kpi.t50_min
    if t50 is None:
        # Check quality flags for extra context
        has_low_count = any("unreliable" in f.lower() or "low" in f.lower()
                           for f in kpi.quality_flags)
        msg = "Settling t50 not measurable"
        if has_low_count:
            msg += " (low floc count during settling)"
        return StageResult("poor", _status_icon("poor"), "N/A", msg)

    if t50 <= _SETTLING_THRESHOLDS["good"]:
        status = "good"
        msg = f"Fast settling (t50 = {t50:.1f} min)"
    elif t50 > _SETTLING_THRESHOLDS["poor"]:
        status = "poor"
        msg = f"Slow settling (t50 = {t50:.1f} min)"
    else:
        status = "mixed"
        msg = f"Moderate settling (t50 = {t50:.1f} min)"

    return StageResult(status, _status_icon(status), f"{t50:.1f} min", msg)


def evaluate_overall(formation: StageResult, size: StageResult,
                     settling: StageResult, kpi) -> StageResult:
    statuses = [formation.status, size.status, settling.status]
    n_good = statuses.count("good")
    n_poor = statuses.count("poor")

    # Check for severe confidence issues
    severe_flags = any("unreliable" in f.lower() for f in kpi.quality_flags)

    if n_poor >= 2 or (n_poor >= 1 and severe_flags):
        status = "poor"
        msg = "Multiple stages underperforming"
    elif n_good >= 2 and n_poor == 0:
        status = "good"
        msg = "Strong performance across stages"
    else:
        status = "mixed"
        msg = "Mixed results \u2013 some stages need attention"

    return StageResult(status, _status_icon(status), "", msg)


def recommend_action(formation: StageResult, size: StageResult,
                     settling: StageResult,
                     overall: StageResult) -> tuple[str, str]:
    """Returns (action_text, explanation)."""
    if overall.status == "good":
        return "Keep dose", "Run quality is good across all stages."

    if formation.status == "poor":
        return "Re-test", "Verify mixing conditions and coagulant feed."

    if settling.status == "poor" and formation.status == "good":
        return "Consider slight dose increase", "Flocs form well but settle slowly \u2013 may need more polymer or higher dose."

    if overall.status == "mixed":
        # Check if data is noisy
        if size.status == "poor" and formation.status != "poor":
            return "Consider dose increase", "Flocs are forming but staying small."
        return "Re-test", "Results are mixed \u2013 another run may clarify."

    return "Re-test (data noisy)", "Insufficient clarity to make a dosing recommendation."


# ═══════════════════════════════════════════════════════════════════════════
# Multi-run comparison (simplified)
# ═══════════════════════════════════════════════════════════════════════════

def compare_runs(runs: list[dict]) -> dict:
    """Pick the best run and explain why in 2-3 bullets."""
    scored = [(i, r) for i, r in enumerate(runs) if r["kpi"].score is not None]
    if not scored:
        return {"best_idx": 0, "best_label": runs[0]["meta"].label,
                "status": "mixed", "icon": _status_icon("mixed"),
                "bullets": ["No scores available for comparison"]}

    scored.sort(key=lambda x: x[1]["kpi"].score, reverse=True)
    best_i, best_run = scored[0]
    best_kpi = best_run["kpi"]

    formation = evaluate_formation(best_kpi)
    size = evaluate_floc_size(best_kpi)
    settling = evaluate_settling(best_kpi)
    overall = evaluate_overall(formation, size, settling, best_kpi)

    bullets = []
    if best_kpi.score is not None:
        bullets.append(f"Highest overall score: {best_kpi.score}/100")
    if best_kpi.pre_settle_diameter_um is not None:
        bullets.append(f"Largest pre-settle flocs: {best_kpi.pre_settle_diameter_um:.0f} \u00b5m")
    if best_kpi.t50_min is not None:
        bullets.append(f"Settling t50: {best_kpi.t50_min:.1f} min")
    elif best_kpi.growth_rate_um_per_min is not None:
        bullets.append(f"Growth rate: {best_kpi.growth_rate_um_per_min:.0f} \u00b5m/min")

    return {
        "best_idx": best_i,
        "best_label": best_run["meta"].label,
        "status": overall.status,
        "icon": overall.icon,
        "bullets": bullets[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Generic metric plot (configurable via OPERATOR_METRICS)
# ═══════════════════════════════════════════════════════════════════════════

def plot_metric_simple(df, phases, meta_label, go, phase_colors, phase_labels,
                       chart_layout, phase_by_name, metric_key=_DEFAULT_METRIC):
    """Plot the selected metric vs Time with phase shading – clean operator style."""
    cfg = OPERATOR_METRICS[metric_key]
    col = cfg["column"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time_min"], y=df.get(col),
        mode="lines+markers",
        marker=dict(size=3, color="#0284C7"),
        line=dict(color="#0284C7", width=2),
        name=meta_label,
        hovertemplate=cfg["tooltip_template"],
    ))

    # Phase shading
    for p in phases:
        fig.add_vrect(
            x0=p.start_min, x1=p.end_min,
            fillcolor=phase_colors.get(p.name, "rgba(200,200,200,0.1)"),
            layer="below", line_width=0,
        )
        fig.add_annotation(
            x=(p.start_min + p.end_min) / 2,
            y=1.01, yref="paper",
            xanchor="center", yanchor="bottom",
            text=phase_labels.get(p.name, p.name),
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

    # Settling start line
    se = phase_by_name(phases, "settling")
    if se:
        fig.add_vline(x=se.start_min, line_dash="dash", line_color="#0284C7",
                      line_width=1, opacity=0.5)

    fig.update_layout(
        title=dict(text=cfg["chart_title"], y=0.97,
                   yanchor="top", x=0.5, xanchor="center"),
        xaxis_title="Time (min)",
        yaxis_title=cfg["y_axis_label"],
        **chart_layout,
    )
    return fig


# Keep backward-compatible alias
def plot_diameter_simple(df, phases, meta_label, go, phase_colors, phase_labels,
                         chart_layout, phase_by_name):
    return plot_metric_simple(df, phases, meta_label, go, phase_colors,
                              phase_labels, chart_layout, phase_by_name,
                              metric_key="diameter")


# ═══════════════════════════════════════════════════════════════════════════
# Operator Mode CSS (additional styles)
# ═══════════════════════════════════════════════════════════════════════════

OPERATOR_CSS = """
<style>
    .op-result-card {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.10);
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 2px 8px rgba(27,42,61,0.06);
        margin-bottom: 16px;
        text-align: center;
    }
    .op-result-card.good { border-top: 4px solid #059669; }
    .op-result-card.mixed { border-top: 4px solid #D97706; }
    .op-result-card.poor { border-top: 4px solid #DC2626; }

    .op-result-icon { font-size: 2.5rem; margin-bottom: 4px; }
    .op-result-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #526580;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .op-result-status {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1B2A3D;
        margin-bottom: 4px;
    }
    .op-result-msg {
        font-size: 0.88rem;
        color: #526580;
        margin: 0;
    }

    .op-action-card {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.10);
        border-radius: 16px;
        border-left: 5px solid #0284C7;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(27,42,61,0.06);
        margin-bottom: 16px;
    }
    .op-action-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #526580;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .op-action-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1B2A3D;
        margin-bottom: 4px;
    }
    .op-action-reason {
        font-size: 0.85rem;
        color: #526580;
        margin: 0;
    }

    .op-stage-card {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.10);
        border-radius: 14px;
        padding: 18px 16px;
        box-shadow: 0 1px 3px rgba(27,42,61,0.06);
        text-align: center;
        height: 100%;
    }
    .op-stage-icon { font-size: 2rem; margin-bottom: 2px; }
    .op-stage-name {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #526580;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .op-stage-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B2A3D;
        margin-bottom: 4px;
    }
    .op-stage-msg {
        font-size: 0.78rem;
        color: #526580;
        margin: 0;
    }

    .op-compare-card {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.10);
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 2px 8px rgba(27,42,61,0.06);
        margin-bottom: 16px;
    }
    .op-compare-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #526580;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .op-compare-best {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1B2A3D;
        margin-bottom: 8px;
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════
# Main Operator Mode renderer
# ═══════════════════════════════════════════════════════════════════════════

def show_operator_mode(st, runs, go, phase_colors, phase_labels, chart_layout, phase_by_name):
    """Render the Operator Mode dashboard."""

    st.markdown(OPERATOR_CSS, unsafe_allow_html=True)
    st.markdown("## FlocBot Run Insights")
    st.caption(f'{len(runs)} run{"s" if len(runs) != 1 else ""} loaded \u2022 Operator Mode')
    st.markdown('<div class="header-accent"></div>', unsafe_allow_html=True)

    # ── Multi-run comparison box ──
    if len(runs) >= 2:
        cmp = compare_runs(runs)
        st.markdown(
            f'<div class="op-compare-card">'
            f'<p class="op-compare-title">Which run is better?</p>'
            f'<p class="op-compare-best">{cmp["icon"]} Best Run: {cmp["best_label"]}</p>'
            + "".join(f'<p class="op-result-msg">\u2022 {b}</p>' for b in cmp["bullets"])
            + '</div>',
            unsafe_allow_html=True,
        )
        def _go_advanced():
            st.session_state["_switch_to_advanced"] = True
        st.button("Show advanced comparison details \u2192",
                  key="op_switch_advanced", on_click=_go_advanced)

    # Show details for the best/only run
    if len(runs) >= 2:
        cmp = compare_runs(runs)
        display_run = runs[cmp["best_idx"]]
    else:
        display_run = runs[0]

    kpi = display_run["kpi"]
    meta = display_run["meta"]
    df = display_run["df"]
    phases = display_run["phases"]

    # If multiple runs, let user pick which to view
    if len(runs) >= 2:
        run_labels = [r["meta"].label for r in runs]
        cmp_data = compare_runs(runs)
        selected_idx = st.selectbox(
            "View run details",
            range(len(runs)),
            index=cmp_data["best_idx"],
            format_func=lambda i: run_labels[i],
            key="op_run_select",
        )
        display_run = runs[selected_idx]
        kpi = display_run["kpi"]
        meta = display_run["meta"]
        df = display_run["df"]
        phases = display_run["phases"]

    # ── Evaluate stages ──
    formation = evaluate_formation(kpi)
    size = evaluate_floc_size(kpi)
    settling = evaluate_settling(kpi)
    overall = evaluate_overall(formation, size, settling, kpi)
    action, action_reason = recommend_action(formation, size, settling, overall)

    status_word = {"good": "Good", "mixed": "Mixed", "poor": "Poor"}[overall.status]

    # ── A) RUN RESULT card ──
    st.markdown(
        f'<div class="op-result-card {overall.status}">'
        f'<p class="op-result-label">Run Result</p>'
        f'<p class="op-result-icon">{overall.icon}</p>'
        f'<p class="op-result-status">{status_word}</p>'
        f'<p class="op-result-msg">{overall.message}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── B) RECOMMENDED ACTION card ──
    st.markdown(
        f'<div class="op-action-card">'
        f'<p class="op-action-title">Recommended Action</p>'
        f'<p class="op-action-text">{action}</p>'
        f'<p class="op-action-reason">{action_reason}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── C) Three stage cards ──
    cols = st.columns(3)
    stages = [
        ("Floc Formation", formation),
        ("Floc Size", size),
        ("Settling", settling),
    ]
    for col, (name, stage) in zip(cols, stages):
        with col:
            st.markdown(
                f'<div class="op-stage-card">'
                f'<p class="op-stage-icon">{stage.icon}</p>'
                f'<p class="op-stage-name">{name}</p>'
                f'<p class="op-stage-value">{stage.value_text}</p>'
                f'<p class="op-stage-msg">{stage.message}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── D) Plot with metric selector ──
    st.markdown("---")

    # Determine which metrics are available for the current run
    available_metrics = {}
    for key, cfg in OPERATOR_METRICS.items():
        if cfg["column"] in df.columns:
            available_metrics[key] = cfg["display_name"]
        else:
            available_metrics[key] = None  # mark unavailable

    # Build selectbox options (only available metrics are selectable)
    metric_options = [k for k, v in available_metrics.items() if v is not None]
    metric_labels = {k: OPERATOR_METRICS[k]["display_name"] for k in metric_options}

    # Default to diameter; if somehow missing, take first available
    default_idx = metric_options.index(_DEFAULT_METRIC) if _DEFAULT_METRIC in metric_options else 0

    # Guard: if the persisted selection is no longer available (e.g. user
    # switched to a run without vol_conc), reset to default before the
    # widget is created so Streamlit doesn't raise a mismatch error.
    persisted = st.session_state.get("op_metric_select")
    if persisted is not None and persisted not in metric_options:
        st.session_state["op_metric_select"] = metric_options[default_idx]

    # Compact row: selector + unavailability note
    sel_col, note_col = st.columns([2, 3])
    with sel_col:
        selected_metric = st.selectbox(
            "Y-axis metric",
            metric_options,
            index=default_idx,
            format_func=lambda k: metric_labels[k],
            key="op_metric_select",
        )

    # Show inline note for unavailable metrics
    unavailable = [OPERATOR_METRICS[k]["display_name"]
                   for k, v in available_metrics.items() if v is None]
    if unavailable:
        with note_col:
            st.caption(
                f"_{', '.join(unavailable)}_ not available in this export"
            )

    fig = plot_metric_simple(
        df, phases, meta.short_label,
        go, phase_colors, phase_labels, chart_layout, phase_by_name,
        metric_key=selected_metric,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── E) Metadata (collapsible) ──
    with st.expander("Run Details", expanded=False):
        md_rows = [
            ("Protocol", meta.protocol_title or "\u2014"),
            ("Chemistry", meta.run_chemistry or "\u2014"),
            ("Dosage", meta.run_dosage or "\u2014"),
            ("Comments", meta.comments or "\u2014"),
            ("File", meta.filename),
            ("Generated", meta.generated_timestamp or "\u2014"),
        ]
        md_table = "| Field | Value |\n|:--|:--|\n"
        md_table += "\n".join(f"| {k} | {v} |" for k, v in md_rows)
        st.markdown(md_table)
