"""
ui_operator.py – Operator Mode dashboard for FlocBot Run Insights.

Provides a simplified, traffic-light-based view of run quality
with recommended actions. Reuses existing KPI pipeline outputs.

Evaluation priority:
  1. Multi-run comparison (relative to other loaded runs)
  2. Plant-specific baseline (saved representative performance)
  3. Fallback heuristic thresholds (generic, used provisionally)
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import re as _re


# ═══════════════════════════════════════════════════════════════════════════
# Metric selector configuration – single source of truth
# ═══════════════════════════════════════════════════════════════════════════

OPERATOR_METRICS = {
    "diameter": {
        "display_name": "Mean Floc Diameter",
        "column": "diameter_um",
        "y_axis_label": "Mean Diameter (\u00b5m)",
        "chart_title": "Floc Growth Over Time",
        "tooltip_template": "Time: %{x:.1f} min<br>Diameter: %{y:.0f} \u00b5m<extra></extra>",
        "legend_label_suffix": "",
    },
    "vol_conc": {
        "display_name": "Volume Concentration",
        "column": "vol_conc_mm3_L",
        "y_axis_label": "Vol. Concentration (mm\u00b3/L)",
        "chart_title": "Floc Concentration Over Time",
        "tooltip_template": "Time: %{x:.1f} min<br>Vol. Conc: %{y:.1f} mm\u00b3/L<extra></extra>",
        "legend_label_suffix": "",
    },
}

_DEFAULT_METRIC = "diameter"

_PHASE_SUBTITLES = {
    "rapid_mix": "disperse coagulant",
    "flocculation": "grow flocs",
    "settling": "clarify water",
}


# ═══════════════════════════════════════════════════════════════════════════
# Plant baseline storage (Supabase DB)
# ═══════════════════════════════════════════════════════════════════════════

import supabase_client as _db
import streamlit as _st


@dataclass
class PlantBaseline:
    """Representative performance values for a plant/protocol."""
    protocol: str
    growth_rate_um_per_min: Optional[float] = None
    time_to_300_min: Optional[float] = None
    pre_settle_diameter_um: Optional[float] = None
    t50_min: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> PlantBaseline:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _get_auth():
    """Get access_token and user_id from Streamlit session_state."""
    user = _st.session_state.get("user", {})
    return user.get("access_token", ""), user.get("id", "")


def save_baseline(protocol: str, baseline: PlantBaseline) -> None:
    token, user_id = _get_auth()
    if not token:
        return
    _db.save_baseline(token, user_id, protocol, baseline.to_dict())


def load_baseline(protocol: str) -> Optional[PlantBaseline]:
    token, _ = _get_auth()
    if not token:
        return None
    try:
        data = _db.get_baseline(token, protocol)
        if data is None:
            return None
        return PlantBaseline.from_dict(data)
    except Exception:
        return None


def save_baseline_from_kpi(protocol: str, kpi) -> PlantBaseline:
    """Create and save a baseline from a run's KPI values."""
    bl = PlantBaseline(
        protocol=protocol,
        growth_rate_um_per_min=kpi.growth_rate_um_per_min,
        time_to_300_min=kpi.time_to_thresholds_min.get(300),
        pre_settle_diameter_um=kpi.pre_settle_diameter_um,
        t50_min=kpi.t50_min,
    )
    save_baseline(protocol, bl)
    return bl


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation context – determines how runs are classified
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalContext:
    """Contextual information for evaluating a run."""
    mode: str = "fallback"    # "comparison", "baseline", "fallback"
    mode_label: str = "Preliminary assessment based on generic thresholds"
    # Comparison-set stats (populated when mode == "comparison")
    best_score: Optional[float] = None
    worst_score: Optional[float] = None
    mean_score: Optional[float] = None
    is_best_run: bool = False
    run_rank: int = 0          # 0-based, 0 = best
    run_count: int = 1
    # Plant baseline (populated when mode == "baseline")
    baseline: Optional[PlantBaseline] = None


def build_eval_context(runs: list[dict], current_kpi, current_meta) -> EvalContext:
    """Build the appropriate evaluation context for a run."""
    protocol = getattr(current_meta, "protocol_title", "") or ""
    current_score = current_kpi.score

    # Priority 1: multi-run comparison
    if len(runs) >= 2:
        scores = [r["kpi"].score for r in runs if r["kpi"].score is not None]
        if len(scores) >= 2:
            scores_sorted = sorted(scores, reverse=True)
            rank = scores_sorted.index(current_score) if current_score in scores_sorted else len(scores_sorted)
            return EvalContext(
                mode="comparison",
                mode_label=f"Evaluated relative to {len(scores)} loaded runs",
                best_score=max(scores),
                worst_score=min(scores),
                mean_score=sum(scores) / len(scores),
                is_best_run=(current_score == max(scores)),
                run_rank=rank,
                run_count=len(scores),
            )

    # Priority 2: plant baseline
    if protocol:
        bl = load_baseline(protocol)
        if bl is not None:
            return EvalContext(
                mode="baseline",
                mode_label=f"Evaluated against {protocol} baseline",
                baseline=bl,
            )

    # Priority 3: fallback
    return EvalContext(
        mode="fallback",
        mode_label="Preliminary assessment based on generic thresholds",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fallback thresholds (used only when no comparison or baseline exists)
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
    status: str          # "good", "marginal", "poor"
    icon: str            # traffic-light emoji
    value_text: str      # formatted key number
    message: str         # short explanation


def _status_icon(status: str) -> str:
    return {"good": "\U0001f7e2", "marginal": "\U0001f7e1", "poor": "\U0001f534"}[status]


def _relative_status(value: Optional[float], baseline_value: Optional[float],
                     higher_is_better: bool = True,
                     marginal_pct: float = 0.20, poor_pct: float = 0.40) -> str:
    """Classify a value relative to a baseline.
    marginal_pct = how far below baseline before 'marginal' (20% default).
    poor_pct = how far below baseline before 'poor' (40% default)."""
    if value is None or baseline_value is None or baseline_value == 0:
        return "marginal"
    if higher_is_better:
        ratio = value / baseline_value
        if ratio >= (1.0 - marginal_pct):
            return "good"
        elif ratio >= (1.0 - poor_pct):
            return "marginal"
        else:
            return "poor"
    else:
        # Lower is better (e.g. time-to-threshold, settling t50)
        ratio = value / baseline_value
        if ratio <= (1.0 + marginal_pct):
            return "good"
        elif ratio <= (1.0 + poor_pct):
            return "marginal"
        else:
            return "poor"


def evaluate_formation(kpi, ctx: Optional[EvalContext] = None) -> StageResult:
    gr = kpi.growth_rate_um_per_min
    t300 = kpi.time_to_thresholds_min.get(300)

    if gr is None:
        return StageResult("poor", _status_icon("poor"), "N/A",
                           "Growth rate could not be measured")

    val = f"{gr:.0f} \u00b5m/min"

    # --- Comparison mode: best run gets "good", others relative ---
    if ctx and ctx.mode == "comparison":
        if ctx.is_best_run:
            parts = [f"Best formation in set ({gr:.0f} \u00b5m/min)"]
            if t300 is not None:
                parts.append(f"300 \u00b5m at {t300:.1f} min")
            return StageResult("good", _status_icon("good"), val, ", ".join(parts))
        else:
            parts = [f"Growth {gr:.0f} \u00b5m/min"]
            if t300 is not None:
                parts.append(f"300 \u00b5m at {t300:.1f} min")
            # Use score rank for classification
            if ctx.run_count > 0 and ctx.run_rank <= ctx.run_count * 0.5:
                status = "good"
            else:
                status = "marginal"
            return StageResult(status, _status_icon(status), val, ", ".join(parts))

    # --- Baseline mode ---
    if ctx and ctx.mode == "baseline" and ctx.baseline:
        bl = ctx.baseline
        status = _relative_status(gr, bl.growth_rate_um_per_min, higher_is_better=True)
        if bl.growth_rate_um_per_min:
            msg = f"Growth {gr:.0f} \u00b5m/min (baseline: {bl.growth_rate_um_per_min:.0f})"
        else:
            msg = f"Growth {gr:.0f} \u00b5m/min"
        if t300 is not None:
            t_status = _relative_status(t300, bl.time_to_300_min, higher_is_better=False)
            if t_status == "poor" and status != "poor":
                status = "marginal"
            msg += f", 300 \u00b5m at {t300:.1f} min"
        return StageResult(status, _status_icon(status), val, msg)

    # --- Fallback thresholds ---
    gr_good = gr >= _FORMATION_THRESHOLDS["growth_good"]
    gr_poor = gr < _FORMATION_THRESHOLDS["growth_poor"]

    if t300 is not None:
        t_good = t300 <= _FORMATION_THRESHOLDS["time_good"]
        t_poor = t300 > _FORMATION_THRESHOLDS["time_poor"]
    else:
        t_good = False
        t_poor = True

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
        status = "marginal"
        parts = [f"Growth {gr:.0f} \u00b5m/min"]
        if t300 is not None:
            parts.append(f"300 \u00b5m at {t300:.1f} min")
        msg = ", ".join(parts)

    return StageResult(status, _status_icon(status), val, msg)


def evaluate_floc_size(kpi, ctx: Optional[EvalContext] = None) -> StageResult:
    d = kpi.pre_settle_diameter_um
    if d is None:
        return StageResult("poor", _status_icon("poor"), "N/A",
                           "Pre-settle diameter not available")

    val = f"{d:.0f} \u00b5m"

    # --- Comparison mode ---
    if ctx and ctx.mode == "comparison":
        if ctx.is_best_run:
            return StageResult("good", _status_icon("good"), val,
                               f"Best floc size in set ({d:.0f} \u00b5m)")
        if ctx.run_count > 0 and ctx.run_rank <= ctx.run_count * 0.5:
            status = "good"
        else:
            status = "marginal"
        return StageResult(status, _status_icon(status), val,
                           f"Pre-settle floc size {d:.0f} \u00b5m")

    # --- Baseline mode ---
    if ctx and ctx.mode == "baseline" and ctx.baseline:
        bl = ctx.baseline
        status = _relative_status(d, bl.pre_settle_diameter_um, higher_is_better=True)
        if bl.pre_settle_diameter_um:
            msg = f"Pre-settle {d:.0f} \u00b5m (baseline: {bl.pre_settle_diameter_um:.0f})"
        else:
            msg = f"Pre-settle floc size {d:.0f} \u00b5m"
        return StageResult(status, _status_icon(status), val, msg)

    # --- Fallback ---
    if d >= _SIZE_THRESHOLDS["good"]:
        status = "good"
        msg = f"Large flocs ({d:.0f} \u00b5m) before settling"
    elif d < _SIZE_THRESHOLDS["poor"]:
        status = "poor"
        msg = f"Small flocs ({d:.0f} \u00b5m) \u2013 may settle slowly"
    else:
        status = "marginal"
        msg = f"Moderate floc size ({d:.0f} \u00b5m)"

    return StageResult(status, _status_icon(status), val, msg)


def evaluate_settling(kpi, ctx: Optional[EvalContext] = None) -> StageResult:
    """Evaluate settling quality. Informational only — not used in scoring."""
    _INFO = " (informational \u2014 not used in scoring)"
    t50 = kpi.t50_min
    if t50 is None:
        return StageResult("marginal", _status_icon("marginal"), "N/A",
                           "Settling data inconclusive" + _INFO)

    val = f"{t50:.1f} min"

    # --- Comparison mode ---
    if ctx and ctx.mode == "comparison":
        if ctx.is_best_run:
            msg = f"Settling t50 = {t50:.1f} min (best run)"
            return StageResult("good", _status_icon("good"), val, msg + _INFO)
        msg = f"Settling t50 = {t50:.1f} min"
        if ctx.run_count > 0 and ctx.run_rank <= ctx.run_count * 0.5:
            status = "good"
        else:
            status = "marginal"
        return StageResult(status, _status_icon(status), val, msg + _INFO)

    # --- Baseline mode ---
    if ctx and ctx.mode == "baseline" and ctx.baseline:
        bl = ctx.baseline
        status = _relative_status(t50, bl.t50_min, higher_is_better=False)
        if bl.t50_min:
            msg = f"Settling t50 = {t50:.1f} min (baseline: {bl.t50_min:.1f})"
        else:
            msg = f"Settling t50 = {t50:.1f} min"
        return StageResult(status, _status_icon(status), val, msg + _INFO)

    # --- Fallback ---
    if t50 <= _SETTLING_THRESHOLDS["good"]:
        status = "good"
        msg = f"Fast settling (t50 = {t50:.1f} min)"
    elif t50 > _SETTLING_THRESHOLDS["poor"]:
        status = "poor"
        msg = f"Slow settling (t50 = {t50:.1f} min)"
    else:
        status = "marginal"
        msg = f"Moderate settling (t50 = {t50:.1f} min)"

    return StageResult(status, _status_icon(status), val, msg + _INFO)


def evaluate_overall(formation: StageResult, size: StageResult,
                     settling: StageResult, kpi,
                     ctx: Optional[EvalContext] = None) -> StageResult:
    """Overall run result. Based on formation and size only."""

    # --- Comparison mode: best run is always "good" ---
    if ctx and ctx.mode == "comparison" and ctx.is_best_run:
        return StageResult("good", _status_icon("good"), "",
                           "Best performing run in this set")

    statuses = [formation.status, size.status]
    n_good = statuses.count("good")
    n_poor = statuses.count("poor")

    if n_poor == 2:
        status = "poor"
        msg = "Formation and size both underperforming"
    elif n_poor == 1:
        status = "marginal"
        which = "Formation" if formation.status == "poor" else "Floc size"
        msg = f"{which} needs attention"
    elif n_good == 2:
        status = "good"
        msg = "Strong formation and floc size"
    elif n_good == 1:
        status = "good"
        msg = "Good overall \u2013 one stage marginal"
    else:
        # both marginal
        status = "marginal"
        msg = "Room for improvement \u2013 performance is within a workable range"

    # Add context label for baseline/fallback
    if ctx and ctx.mode == "baseline":
        msg += " (relative to plant baseline)"
    elif ctx and ctx.mode == "fallback":
        msg += " (based on generic thresholds)"

    return StageResult(status, _status_icon(status), "", msg)


# ═══════════════════════════════════════════════════════════════════════════
# Single-run recommendation (no comparative data)
# ═══════════════════════════════════════════════════════════════════════════

def recommend_action_single(formation: StageResult, size: StageResult,
                            settling: StageResult,
                            overall: StageResult) -> tuple[str, str]:
    """Single-run recommendation (no comparative data available)."""
    if overall.status == "good":
        return "Keep current dose", "Formation and floc size look good."

    if formation.status == "poor":
        return ("No clear dose recommendation",
                "Formation is poor \u2013 verify mixing conditions and coagulant feed before adjusting dose.")

    if size.status == "poor" and formation.status != "poor":
        return ("Consider increasing coagulant dose",
                "Flocs are forming but staying small \u2013 a higher dose may improve size.")

    if overall.status == "marginal":
        return ("No clear dose recommendation",
                "Results are marginal \u2013 upload additional runs at different doses to compare.")

    return ("No clear dose recommendation",
            "Insufficient data to make a dosing recommendation \u2013 upload additional runs to compare.")


# ═══════════════════════════════════════════════════════════════════════════
# Dose parsing and comparative recommendation
# ═══════════════════════════════════════════════════════════════════════════

def _parse_doses(meta) -> tuple[dict[str, Optional[float]], set[str]]:
    """Parse chemistry/dosage fields into ({chemical_name: numeric_dose}, unnamed_set).

    Expects slash-delimited chemistry names and dose values, e.g.:
        chemistry = "Alum/Poly/Reclaim %"
        dosage    = "20/2.75/4/81"
    Returns e.g. ({"Alum": 20.0, "Poly": 2.75, ...}, {"Variable 4"})
    """
    dosage_str = getattr(meta, "run_dosage", "") or ""
    chemistry_str = getattr(meta, "run_chemistry", "") or ""

    if not dosage_str.strip():
        return {}, set()

    dosage_clean = _re.sub(r"\s+[a-zA-Z/]+\s*$", "", dosage_str.strip())

    dose_parts = [p.strip() for p in dosage_clean.split("/")]
    chem_parts = [p.strip() for p in chemistry_str.split("/")] if chemistry_str.strip() else []

    result: dict[str, Optional[float]] = {}
    unnamed: set[str] = set()
    for i, raw_dose in enumerate(dose_parts):
        if i < len(chem_parts) and chem_parts[i]:
            name = chem_parts[i]
        else:
            name = f"Variable {i + 1}"
            unnamed.add(name)
        try:
            result[name] = float(raw_dose)
        except (ValueError, TypeError):
            result[name] = None

    return result, unnamed


@dataclass
class ChemicalRec:
    """Dose recommendation for a single chemical."""
    chemical: str
    direction: str       # "increase", "decrease", "keep", "unclear"
    explanation: str
    is_named: bool = True


@dataclass
class DoseRecommendation:
    """Full dose recommendation from multi-run comparison."""
    chemicals: list       # list[ChemicalRec]
    summary: str
    explanation: str


def _compare_doses(runs: list[dict]) -> Optional[DoseRecommendation]:
    """Compare runs by dose and performance, returning per-chemical guidance."""
    entries = []
    all_unnamed: set[str] = set()
    for r in runs:
        score = r["kpi"].score
        if score is None:
            continue
        doses, unnamed = _parse_doses(r["meta"])
        if not doses:
            continue
        all_unnamed |= unnamed
        entries.append((doses, score))

    if len(entries) < 2:
        return None

    all_chems: list[str] = []
    seen: set[str] = set()
    for doses, _ in entries:
        for c in doses:
            if c not in seen:
                all_chems.append(c)
                seen.add(c)

    recs: list[ChemicalRec] = []

    for chem in all_chems:
        named = chem not in all_unnamed
        dose_suffix = " dose" if named else ""
        values_label = f"{chem}{dose_suffix}"

        pairs = []
        for doses, score in entries:
            val = doses.get(chem)
            if val is not None:
                pairs.append((val, score))

        if len(pairs) < 2:
            continue

        unique_doses = set(d for d, _ in pairs)
        if len(unique_doses) <= 1:
            continue

        other_chems_changed = []
        for other in all_chems:
            if other == chem:
                continue
            other_vals = set()
            for doses, _ in entries:
                v = doses.get(other)
                if v is not None:
                    other_vals.add(v)
            if len(other_vals) > 1:
                other_chems_changed.append(other)

        pairs.sort(key=lambda x: x[0])

        n = len(pairs)
        higher_better = 0
        lower_better = 0
        for i in range(n):
            for j in range(i + 1, n):
                d_i, s_i = pairs[i]
                d_j, s_j = pairs[j]
                if d_j > d_i:
                    if s_j > s_i:
                        higher_better += 1
                    elif s_j < s_i:
                        lower_better += 1

        total_comparisons = higher_better + lower_better
        if total_comparisons == 0:
            recs.append(ChemicalRec(
                chem, "keep",
                f"Performance was similar across {values_label} values tested.",
                is_named=named,
            ))
            continue

        confounded = len(other_chems_changed) > 0

        if confounded:
            changed_names = ", ".join(other_chems_changed)
            if higher_better > lower_better:
                recs.append(ChemicalRec(
                    chem, "unclear",
                    f"Higher {values_label} trended better, but {changed_names} also "
                    f"changed between runs \u2013 effect cannot be isolated.",
                    is_named=named,
                ))
            elif lower_better > higher_better:
                recs.append(ChemicalRec(
                    chem, "unclear",
                    f"Lower {values_label} trended better, but {changed_names} also "
                    f"changed between runs \u2013 effect cannot be isolated.",
                    is_named=named,
                ))
            else:
                recs.append(ChemicalRec(
                    chem, "unclear",
                    f"No clear trend for {chem} \u2013 {changed_names} also "
                    f"changed between runs.",
                    is_named=named,
                ))
            continue

        ratio = max(higher_better, lower_better) / total_comparisons

        if higher_better > lower_better:
            consistency = "consistently" if ratio >= 0.75 else "generally"
            extra = "" if ratio >= 0.75 else f" ({higher_better}/{total_comparisons} comparisons)"
            recs.append(ChemicalRec(
                chem, "increase",
                f"Higher {values_label} {consistency} performed better.{extra}",
                is_named=named,
            ))
        elif lower_better > higher_better:
            consistency = "consistently" if ratio >= 0.75 else "generally"
            extra = "" if ratio >= 0.75 else f" ({lower_better}/{total_comparisons} comparisons)"
            recs.append(ChemicalRec(
                chem, "decrease",
                f"Lower {values_label} {consistency} performed better.{extra}",
                is_named=named,
            ))
        else:
            recs.append(ChemicalRec(
                chem, "keep",
                f"Performance was similar across {values_label} values tested.",
                is_named=named,
            ))

    if not recs:
        return None

    summary, explanation = _build_dose_summary(recs)
    return DoseRecommendation(chemicals=recs, summary=summary, explanation=explanation)


def _build_dose_summary(recs: list[ChemicalRec]) -> tuple[str, str]:
    """Build a one-line summary and explanation from per-chemical recs."""
    direction_labels = {
        "increase": "Consider increasing",
        "decrease": "Consider decreasing",
        "keep": "Keep current",
        "unclear": "No clear recommendation for",
    }

    actionable = [r for r in recs if r.direction in ("increase", "decrease")]
    keeps = [r for r in recs if r.direction == "keep"]
    unclear = [r for r in recs if r.direction == "unclear"]

    if not actionable and not keeps and unclear:
        return ("No clear dose recommendation",
                unclear[0].explanation)

    if not actionable and keeps:
        return ("Keep current dose",
                keeps[0].explanation)

    if len(actionable) == 1 and not unclear:
        r = actionable[0]
        dose_suffix = " dose" if r.is_named else ""
        return (f"{direction_labels[r.direction]} {r.chemical}{dose_suffix}",
                r.explanation)

    # Multiple actionable or mixed with unclear
    parts = []
    for r in actionable:
        dose_suffix = " dose" if r.is_named else ""
        parts.append(f"{direction_labels[r.direction]} {r.chemical}{dose_suffix}")
    for r in unclear:
        parts.append(f"{direction_labels[r.direction]} {r.chemical}")

    summary = parts[0] if len(parts) >= 1 else "No clear dose recommendation"
    explanation = " ".join(r.explanation for r in recs)
    return summary, explanation


def recommend_action(formation: StageResult, size: StageResult,
                     settling: StageResult,
                     overall: StageResult,
                     runs: Optional[list] = None) -> tuple[str, str]:
    """Returns (action_text, explanation).

    When multiple runs are available, uses comparative dose analysis.
    Falls back to single-run heuristics otherwise."""
    if runs and len(runs) >= 2:
        dose_rec = _compare_doses(runs)
        if dose_rec is not None:
            return dose_rec.summary, dose_rec.explanation

    return recommend_action_single(formation, size, settling, overall)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-run comparison (simplified)
# ═══════════════════════════════════════════════════════════════════════════

def compare_runs(runs: list[dict]) -> dict:
    """Pick the best run and explain why in 2-3 bullets."""
    scored = [(i, r) for i, r in enumerate(runs) if r["kpi"].score is not None]
    if not scored:
        return {"best_idx": 0, "best_label": runs[0]["meta"].label,
                "status": "marginal", "icon": _status_icon("marginal"),
                "bullets": ["No scores available for comparison"]}

    scored.sort(key=lambda x: x[1]["kpi"].score, reverse=True)
    best_i, best_run = scored[0]
    best_kpi = best_run["kpi"]

    ctx = build_eval_context(runs, best_kpi, best_run["meta"])
    formation = evaluate_formation(best_kpi, ctx)
    size = evaluate_floc_size(best_kpi, ctx)
    settling = evaluate_settling(best_kpi, ctx)
    overall = evaluate_overall(formation, size, settling, best_kpi, ctx)

    bullets = []
    if best_kpi.score is not None:
        bullets.append(f"Score: {best_kpi.score}/100")
    if best_kpi.pre_settle_diameter_um is not None:
        bullets.append(f"Pre-settle size: {best_kpi.pre_settle_diameter_um:.0f} \u00b5m")
    if best_kpi.t50_min is not None:
        bullets.append(f"Settling (t50): {best_kpi.t50_min:.1f} min")
    elif best_kpi.growth_rate_um_per_min is not None:
        bullets.append(f"Growth: {best_kpi.growth_rate_um_per_min:.0f} \u00b5m/min")

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

    for p in phases:
        fig.add_vrect(
            x0=p.start_min, x1=p.end_min,
            fillcolor=phase_colors.get(p.name, "rgba(200,200,200,0.1)"),
            layer="below", line_width=0,
        )
        mid_x = (p.start_min + p.end_min) / 2
        fig.add_annotation(
            x=mid_x, y=1.02, yref="paper",
            xanchor="center", yanchor="bottom",
            text=phase_labels.get(p.name, p.name),
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

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
    .op-result-card.marginal { border-top: 4px solid #D97706; }
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
        color: #0F1D2E;
        margin-bottom: 4px;
    }
    .op-result-msg {
        font-size: 0.88rem;
        color: #526580;
        margin: 0;
    }
    .op-eval-mode {
        font-size: 0.72rem;
        color: #8C9BB0;
        font-style: italic;
        margin-top: 8px;
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
    .op-action-chem {
        font-size: 0.82rem;
        color: #334155;
        margin: 6px 0 0 0;
        padding-left: 4px;
    }
    .op-action-chem-detail {
        color: #526580;
        font-weight: 400;
    }

    .op-stage-card {
        background: #FFFFFF;
        border: 1px solid rgba(2,132,199,0.10);
        border-radius: 14px;
        padding: 18px 16px;
        box-shadow: 0 1px 3px rgba(27,42,61,0.06);
        text-align: center;
        min-height: 200px;
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
        color: #0F1D2E;
        margin-bottom: 4px;
    }
    .op-stage-msg {
        font-size: 0.78rem;
        color: #526580;
        margin: 0;
    }
    .op-stage-hint {
        font-size: 0.68rem;
        color: #8C9BB0;
        font-style: italic;
        margin: 6px 0 0 0;
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
        color: #0F1D2E;
        margin-bottom: 8px;
    }

    /* Secondary button style for "View advanced comparison" */
    .op-compare-panel + div button {
        background: transparent !important;
        border: 1px solid #CBD5E1 !important;
        color: #526580 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        padding: 6px 16px !important;
        box-shadow: none !important;
    }
    .op-compare-panel + div button:hover {
        background: #F1F5F9 !important;
        border-color: #94A3B8 !important;
        color: #334155 !important;
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
            f'<div class="op-compare-panel">'
            f'<div class="op-compare-card">'
            f'<p class="op-compare-title">Recommended Run</p>'
            f'<p class="op-compare-best">{cmp["icon"]} {cmp["best_label"]}</p>'
            + "".join(f'<p class="op-result-msg">\u2022 {b}</p>' for b in cmp["bullets"])
            + '</div></div>',
            unsafe_allow_html=True,
        )
        def _go_advanced():
            st.session_state["_switch_to_advanced"] = True
        st.button("View advanced comparison \u2192",
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

    # ── Build evaluation context ──
    ctx = build_eval_context(runs, kpi, meta)

    # ── Evaluate stages ──
    formation = evaluate_formation(kpi, ctx)
    size = evaluate_floc_size(kpi, ctx)
    settling = evaluate_settling(kpi, ctx)
    overall = evaluate_overall(formation, size, settling, kpi, ctx)
    action, action_reason = recommend_action(
        formation, size, settling, overall, runs=runs if len(runs) >= 2 else None,
    )

    # Get per-chemical detail for multi-run
    dose_rec = _compare_doses(runs) if len(runs) >= 2 else None

    status_word = {"good": "Good", "marginal": "Marginal", "poor": "Poor"}[overall.status]

    # ── A) RUN RESULT card ──
    st.markdown(
        f'<div class="op-result-card {overall.status}">'
        f'<p class="op-result-label">Run Result</p>'
        f'<p class="op-result-icon">{overall.icon}</p>'
        f'<p class="op-result-status">{status_word}</p>'
        f'<p class="op-result-msg">{overall.message}</p>'
        f'<p class="op-eval-mode">{ctx.mode_label}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── B) RECOMMENDED ACTION card ──
    _dir_icons = {"increase": "\u2191", "decrease": "\u2193",
                  "keep": "\u2713", "unclear": "?"}
    chem_detail_html = ""
    if dose_rec and dose_rec.chemicals:
        lines = []
        for cr in dose_rec.chemicals:
            icon = _dir_icons.get(cr.direction, "")
            label = cr.direction.capitalize()
            lines.append(
                f'<p class="op-action-chem">'
                f'{icon} <strong>{cr.chemical}:</strong> {label} '
                f'<span class="op-action-chem-detail">\u2013 {cr.explanation}</span></p>'
            )
        chem_detail_html = "".join(lines)

    st.markdown(
        f'<div class="op-action-card">'
        f'<p class="op-action-title">Recommended Action</p>'
        f'<p class="op-action-text">{action}</p>'
        f'<p class="op-action-reason">{action_reason}</p>'
        f'{chem_detail_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── C) Three stage cards ──
    cols = st.columns(3)
    stages = [
        ("Floc Formation", formation, "Higher growth = stronger coagulation"),
        ("Floc Size", size, "Larger flocs usually settle faster"),
        ("Settling", settling, "Faster settling = better clarifier performance"),
    ]
    for col, (name, stage, hint) in zip(cols, stages):
        with col:
            st.markdown(
                f'<div class="op-stage-card">'
                f'<p class="op-stage-icon">{stage.icon}</p>'
                f'<p class="op-stage-name">{name}</p>'
                f'<p class="op-stage-value">{stage.value_text}</p>'
                f'<p class="op-stage-msg">{stage.message}</p>'
                f'<p class="op-stage-hint">{hint}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── D) Plot with metric selector ──
    st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)

    available_metrics = {}
    for key, cfg in OPERATOR_METRICS.items():
        if cfg["column"] in df.columns:
            available_metrics[key] = cfg["display_name"]
        else:
            available_metrics[key] = None

    metric_options = [k for k, v in available_metrics.items() if v is not None]
    metric_labels = {k: OPERATOR_METRICS[k]["display_name"] for k in metric_options}

    default_idx = metric_options.index(_DEFAULT_METRIC) if _DEFAULT_METRIC in metric_options else 0

    persisted = st.session_state.get("op_metric_select")
    if persisted is not None and persisted not in metric_options:
        st.session_state["op_metric_select"] = metric_options[default_idx]

    sel_col, note_col = st.columns([2, 3])
    with sel_col:
        selected_metric = st.selectbox(
            "Chart metric",
            metric_options,
            index=default_idx,
            format_func=lambda k: metric_labels[k],
            key="op_metric_select",
        )

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

    # Plant baseline is now managed from the sidebar, not here.
