"""
flocbot_metrics.py – Phase detection, KPI computation, scoring, and data-quality flags.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy.interpolate import interp1d
from scipy.stats import linregress


# ═══════════════════════════════════════════════════════════════════════════
# Phase detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Phase:
    name: str
    start_min: float
    end_min: float
    rpm: Optional[float] = None

    @property
    def duration_min(self) -> float:
        return self.end_min - self.start_min


def detect_phases(df: pd.DataFrame) -> list[Phase]:
    """
    Detect rapid_mix / flocculation / settling from RPM column.
    Returns list of Phase objects sorted by start time.
    """
    if "rpm" not in df.columns or df["rpm"].isna().all():
        return []

    rpm = df["rpm"].fillna(0).values
    time_min = df["time_min"].values

    # Build contiguous segments of constant RPM
    segments: list[tuple[float, float, float]] = []  # (start, end, rpm)
    seg_start = 0
    for i in range(1, len(rpm)):
        if rpm[i] != rpm[seg_start]:
            segments.append((time_min[seg_start], time_min[i - 1], rpm[seg_start]))
            seg_start = i
    segments.append((time_min[seg_start], time_min[-1], rpm[seg_start]))

    if not segments:
        return []

    # Classify
    phases: list[Phase] = []
    max_rpm = max(s[2] for s in segments)

    for start, end, seg_rpm in segments:
        if seg_rpm == max_rpm and max_rpm > 0:
            phases.append(Phase("rapid_mix", start, end, seg_rpm))
        elif seg_rpm > 0:
            phases.append(Phase("flocculation", start, end, seg_rpm))
        elif seg_rpm == 0:
            phases.append(Phase("settling", start, end, 0))

    # Merge consecutive phases of the same name
    merged: list[Phase] = []
    for p in phases:
        if merged and merged[-1].name == p.name:
            merged[-1] = Phase(p.name, merged[-1].start_min, p.end_min, p.rpm)
        else:
            merged.append(p)

    return merged


def phase_by_name(phases: list[Phase], name: str) -> Optional[Phase]:
    for p in phases:
        if p.name == name:
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Interpolation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _interp_time_to_value(time_arr, value_arr, target, direction="rising"):
    """Interpolate the time at which *value_arr* first crosses *target*.
    direction: 'rising' (crosses upward) or 'falling' (crosses downward)."""
    for i in range(1, len(value_arr)):
        if np.isnan(value_arr[i]) or np.isnan(value_arr[i - 1]):
            continue
        if direction == "rising" and value_arr[i - 1] < target <= value_arr[i]:
            frac = (target - value_arr[i - 1]) / (value_arr[i] - value_arr[i - 1])
            return time_arr[i - 1] + frac * (time_arr[i] - time_arr[i - 1])
        if direction == "falling" and value_arr[i - 1] > target >= value_arr[i]:
            frac = (value_arr[i - 1] - target) / (value_arr[i - 1] - value_arr[i])
            return time_arr[i - 1] + frac * (time_arr[i] - time_arr[i - 1])
    return None


# ═══════════════════════════════════════════════════════════════════════════
# KPIs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RunKPIs:
    rapid_mix_duration_min: Optional[float] = None
    flocculation_duration_min: Optional[float] = None
    settling_duration_min: Optional[float] = None

    growth_rate_um_per_min: Optional[float] = None
    growth_rate_window: Optional[str] = None  # human-readable window used
    growth_rate_r2: Optional[float] = None

    time_to_thresholds_min: dict = field(default_factory=dict)  # {threshold: time or None}

    pre_settle_diameter_um: Optional[float] = None
    plateau_mean_um: Optional[float] = None
    plateau_cv: Optional[float] = None

    settle_baseline_vol_conc: Optional[float] = None
    t50_min: Optional[float] = None
    t10_min: Optional[float] = None

    score: Optional[float] = None
    score_components: dict = field(default_factory=dict)
    score_reason: str = ""

    quality_flags: list = field(default_factory=list)


def compute_kpis(
    df: pd.DataFrame,
    phases: list[Phase],
    thresholds: list[float] | None = None,
) -> RunKPIs:
    if thresholds is None:
        thresholds = [250, 300, 350, 400, 450]

    kpi = RunKPIs()
    rm = phase_by_name(phases, "rapid_mix")
    fl = phase_by_name(phases, "flocculation")
    se = phase_by_name(phases, "settling")

    # Phase durations
    kpi.rapid_mix_duration_min = rm.duration_min if rm else None
    kpi.flocculation_duration_min = fl.duration_min if fl else None
    kpi.settling_duration_min = se.duration_min if se else None

    has_diameter = "diameter_um" in df.columns and df["diameter_um"].notna().any()
    has_vol = "vol_conc_mm3_L" in df.columns and df["vol_conc_mm3_L"].notna().any()

    # --- Growth rate (linear fit over flocculation window) ---
    if has_diameter and fl:
        win_start = max(fl.start_min + 1, 2.0)
        win_end = min(fl.start_min + 10, fl.end_min - 1)
        if win_end <= win_start:
            win_start = fl.start_min
            win_end = fl.end_min
            kpi.quality_flags.append("growth_rate: run too short, used full flocculation window")
        mask = (df["time_min"] >= win_start) & (df["time_min"] <= win_end)
        seg = df.loc[mask].dropna(subset=["diameter_um"])
        if len(seg) >= 2:
            slope, intercept, r, p, se_val = linregress(seg["time_min"], seg["diameter_um"])
            kpi.growth_rate_um_per_min = round(slope, 2)
            kpi.growth_rate_r2 = round(r ** 2, 3)
            kpi.growth_rate_window = f"{win_start:.1f}–{win_end:.1f} min"

    # --- Time-to-thresholds ---
    if has_diameter:
        t = df["time_min"].values
        d = df["diameter_um"].values
        for thr in thresholds:
            tt = _interp_time_to_value(t, d, thr, "rising")
            kpi.time_to_thresholds_min[thr] = round(tt, 2) if tt is not None else None

    # --- Pre-settle diameter (last 60 s before settling) ---
    if has_diameter and se:
        pre_start = se.start_min - 1.0
        pre_end = se.start_min
        mask = (df["time_min"] >= pre_start) & (df["time_min"] <= pre_end)
        seg = df.loc[mask, "diameter_um"].dropna()
        if len(seg) > 0:
            kpi.pre_settle_diameter_um = round(seg.mean(), 1)

    # --- Plateau stats (last 3 min of flocculation) ---
    if has_diameter and fl:
        plat_start = max(fl.start_min, fl.end_min - 3.0)
        mask = (df["time_min"] >= plat_start) & (df["time_min"] <= fl.end_min)
        seg = df.loc[mask, "diameter_um"].dropna()
        if len(seg) > 0:
            kpi.plateau_mean_um = round(seg.mean(), 1)
            kpi.plateau_cv = round(seg.std() / seg.mean() * 100, 1) if seg.mean() > 0 else None

    # --- Settling metrics (vol conc) ---
    if has_vol and se:
        # baseline = mean vol_conc in last 60 s BEFORE settling
        bl_start = se.start_min - 1.0
        bl_end = se.start_min
        bl_mask = (df["time_min"] >= bl_start) & (df["time_min"] <= bl_end)
        bl_seg = df.loc[bl_mask, "vol_conc_mm3_L"].dropna()
        if len(bl_seg) > 0:
            baseline = bl_seg.mean()
            kpi.settle_baseline_vol_conc = round(baseline, 2)

            # subset to settling phase
            settle_mask = df["time_min"] >= se.start_min
            st = df.loc[settle_mask].copy()
            st_time = (st["time_min"] - se.start_min).values
            st_vol = st["vol_conc_mm3_L"].values

            t50_val = _interp_time_to_value(st_time, st_vol, baseline * 0.5, "falling")
            t10_val = _interp_time_to_value(st_time, st_vol, baseline * 0.1, "falling")
            kpi.t50_min = round(t50_val, 2) if t50_val is not None else None
            kpi.t10_min = round(t10_val, 2) if t10_val is not None else None

    # --- Data quality flags ---
    _check_quality(df, phases, kpi)

    return kpi


def _check_quality(df: pd.DataFrame, phases: list[Phase], kpi: RunKPIs):
    # Missing/duplicate timestamps
    if df["time_s"].duplicated().any():
        kpi.quality_flags.append("Duplicate timestamps detected")
    if df["time_s"].isna().any():
        kpi.quality_flags.append("Missing timestamps detected")

    # Irregular sampling interval
    diffs = df["time_s"].diff().dropna()
    if len(diffs) > 2:
        median_step = diffs.median()
        if median_step > 0:
            n_irregular = (diffs > 2 * median_step).sum()
            if n_irregular > len(diffs) * 0.1:
                kpi.quality_flags.append(
                    f"Irregular sampling: {n_irregular} gaps > 2x median step ({median_step:.0f}s)"
                )

    # Low sample count
    if "n_samples" in df.columns:
        med_n = df["n_samples"].median()
        if pd.notna(med_n) and med_n < 50:
            kpi.quality_flags.append(f"Low median sample count ({med_n:.0f})")

    # Diameter reliability during settling
    se = phase_by_name(phases, "settling")
    if se and "floc_count_ml" in df.columns and "diameter_um" in df.columns:
        settle_mask = df["time_min"] >= se.start_min
        seg = df.loc[settle_mask]
        if seg["floc_count_ml"].notna().any():
            low = (seg["floc_count_ml"] < 10).sum()
            if low > len(seg) * 0.5:
                kpi.quality_flags.append(
                    "Diameter may be unreliable during settling (very low floc count)"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "time_to_300": 0.30,
    "pre_settle_diameter": 0.30,
    "plateau_cv": 0.20,
    "settling_t50": 0.20,
}

# Normalization reference ranges (used to scale each component to 0–100)
_NORM = {
    "time_to_300": (2.0, 20.0),      # min – lower is better
    "pre_settle_diameter": (100, 600),  # μm – higher is better
    "plateau_cv": (1.0, 30.0),        # % – lower is better
    "settling_t50": (1.0, 15.0),      # min – lower is better
}


def compute_score(
    kpi: RunKPIs,
    weights: dict | None = None,
    threshold_for_time: float = 300,
) -> None:
    """Mutates *kpi* in-place, setting score and score_components."""
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    components: dict[str, Optional[float]] = {}
    reasons: list[str] = []

    # 1) time_to_threshold (lower is better)
    t_val = kpi.time_to_thresholds_min.get(threshold_for_time)
    if t_val is not None:
        lo, hi = _NORM["time_to_300"]
        components["time_to_300"] = _scale_lower_better(t_val, lo, hi)
    else:
        components["time_to_300"] = None
        reasons.append(f"{threshold_for_time} μm threshold not reached")

    # 2) pre_settle_diameter (higher is better)
    if kpi.pre_settle_diameter_um is not None:
        lo, hi = _NORM["pre_settle_diameter"]
        components["pre_settle_diameter"] = _scale_higher_better(kpi.pre_settle_diameter_um, lo, hi)
    else:
        components["pre_settle_diameter"] = None
        reasons.append("No pre-settle diameter")

    # 3) plateau_cv (lower is better)
    if kpi.plateau_cv is not None:
        lo, hi = _NORM["plateau_cv"]
        components["plateau_cv"] = _scale_lower_better(kpi.plateau_cv, lo, hi)
    else:
        components["plateau_cv"] = None
        reasons.append("No plateau CV")

    # 4) settling t50 (lower is better)
    if kpi.t50_min is not None:
        lo, hi = _NORM["settling_t50"]
        components["settling_t50"] = _scale_lower_better(kpi.t50_min, lo, hi)
    else:
        components["settling_t50"] = None
        reasons.append("No settling t50")

    kpi.score_components = components

    # Weighted average over available components
    total_w = 0.0
    total_score = 0.0
    for key, val in components.items():
        if val is not None:
            w = weights.get(key, 0)
            total_w += w
            total_score += w * val

    if total_w > 0:
        kpi.score = round(total_score / total_w, 1)
        if reasons:
            kpi.score_reason = "Partial score – missing: " + "; ".join(reasons)
    else:
        kpi.score = None
        kpi.score_reason = "Insufficient data: " + "; ".join(reasons)


def _scale_lower_better(val, lo, hi):
    """0-100 where lo→100 and hi→0."""
    clamped = max(lo, min(hi, val))
    return round((1.0 - (clamped - lo) / (hi - lo)) * 100, 1)


def _scale_higher_better(val, lo, hi):
    """0-100 where lo→0 and hi→100."""
    clamped = max(lo, min(hi, val))
    return round(((clamped - lo) / (hi - lo)) * 100, 1)
