"""
metric_help.py – Centralized metric definitions for tooltips and help text.

Each entry maps a metric key to a dict with:
  - label:  display name
  - short:  one-line tooltip (shown on hover / st.metric help param)
  - detail: longer explanation for "How to read this page" or info popovers
  - good:   what a good value looks like (plain English)
"""

METRIC_HELP = {
    "score": {
        "label": "Score",
        "short": "Weighted composite score (0–100) combining growth speed, floc size, stability, and settling.",
        "detail": (
            "The overall score combines four sub-metrics into a single 0–100 number: "
            "time to reach 300 μm (30%), pre-settle diameter (30%), signal noise (20%), "
            "and settling t50 (20%). Higher is better. If any sub-metric is unavailable, "
            "the remaining weights are re-normalized."
        ),
        "good": "70+ is strong; 40–70 is acceptable; below 40 needs investigation.",
    },
    "growth_rate": {
        "label": "Growth Rate",
        "short": "How fast floc diameter increases during early flocculation (μm/min).",
        "detail": (
            "Linear slope of mean diameter vs. time over the first ~10 minutes of "
            "flocculation (after rapid mix ends). Measured by least-squares fit. "
            "R² indicates how well the data fits a straight line."
        ),
        "good": "Higher is better — indicates faster floc formation.",
    },
    "pre_settle_diameter": {
        "label": "Pre-settle Ø",
        "short": "Average floc diameter in the last 60 s before settling begins (μm).",
        "detail": (
            "Mean diameter over the final minute of flocculation, just before the "
            "impeller stops. Represents the largest floc size achieved before gravity "
            "settling takes over."
        ),
        "good": "Larger is better — bigger flocs settle faster.",
    },
    "settling_t50": {
        "label": "Settling t50",
        "short": "Time for vol. concentration to drop to 50% of baseline after settling starts (min).",
        "detail": (
            "Interpolated time at which the volume concentration falls to half its "
            "pre-settle baseline value. A proxy for how quickly solids leave the "
            "measurement zone."
        ),
        "good": "Lower is better — faster settling means better performance.",
    },
    "settling_t90": {
        "label": "Settling t90",
        "short": "Time for vol. concentration to drop to 10% of baseline (min).",
        "detail": (
            "Interpolated time at which vol. concentration reaches 10% of the "
            "pre-settle baseline — i.e., 90% of solids have settled out."
        ),
        "good": "Lower is better — indicates nearly complete clarification.",
    },
    "rapid_mix_duration": {
        "label": "Rapid Mix",
        "short": "Duration of the high-RPM rapid mix phase (min).",
        "detail": (
            "Time spent at maximum impeller speed to disperse the coagulant/flocculant. "
            "Detected automatically from the RPM data."
        ),
        "good": "Typically 1–3 min. Longer isn't always better.",
    },
    "flocculation_duration": {
        "label": "Flocculation",
        "short": "Duration of the lower-RPM flocculation phase (min).",
        "detail": (
            "Time at reduced impeller speed where flocs grow. "
            "Detected automatically as the period between rapid mix and settling."
        ),
        "good": "Depends on protocol — usually 10–30 min.",
    },
    "settling_duration": {
        "label": "Settling",
        "short": "Duration of the zero-RPM settling phase (min).",
        "detail": (
            "Time after impeller stops. Flocs settle under gravity. "
            "Volume concentration should decrease during this phase."
        ),
        "good": "Usually 5–15 min. Longer allows more complete settling.",
    },
    "floc_noise_mad": {
        "label": "Signal Noise",
        "short": "Typical reading-to-reading jump in floc size during flocculation (μm). Lower = steadier signal.",
        "detail": (
            "Measures how much the particle size reading jumps around between consecutive "
            "measurements during flocculation. Calculated as the median absolute difference "
            "between successive diameter readings (MAD of first differences). "
            "A low number means the sensor is seeing a smooth, consistent signal. "
            "A high number means the readings are erratic — which can make it harder to "
            "trust the growth rate or spot real changes in floc behaviour."
        ),
        "good": "Lower is better — under 10 μm is a clean signal; above 25 μm is very noisy.",
    },
    "time_to_threshold": {
        "label": "Time to threshold",
        "short": "Time for mean diameter to first reach a given size threshold (min).",
        "detail": (
            "Interpolated time at which the mean floc diameter first crosses the "
            "specified threshold (e.g., 250, 300, 350 μm). 'Not reached' means "
            "the floc never grew that large."
        ),
        "good": "Lower is better — faster to reach target size.",
    },
}


# ── Diagnostics severity classification ──────────────────────────────────

DIAGNOSTIC_RULES = [
    {
        "pattern": "Duplicate timestamps",
        "severity": "info",
        "icon": "ℹ️",
        "summary": "Some rows share the same timestamp.",
        "action": "Usually harmless — caused by sensor polling faster than data changes. No action needed unless values differ between duplicates.",
    },
    {
        "pattern": "Missing timestamps",
        "severity": "caution",
        "icon": "⚠️",
        "summary": "Some rows have blank or unparseable timestamps.",
        "action": "Check the source Excel file for empty rows. These rows are excluded from analysis and may cause small gaps in charts.",
    },
    {
        "pattern": "Irregular sampling",
        "severity": "caution",
        "icon": "⚠️",
        "summary": "Large gaps detected between consecutive measurements.",
        "action": "May indicate sensor dropouts or paused data collection. Check charts for visible gaps. Growth rate and threshold times may be less precise in gap regions.",
    },
    {
        "pattern": "Low median sample count",
        "severity": "warning",
        "icon": "🔴",
        "summary": "The instrument averaged very few particles per measurement.",
        "action": "Low sample counts (below ~50) make diameter readings noisy and less reliable. Consider increasing sample volume or concentration if possible.",
    },
    {
        "pattern": "Diameter may be unreliable during settling",
        "severity": "warning",
        "icon": "🔴",
        "summary": "Very few flocs remain in view during settling, making size measurements unreliable.",
        "action": "Settling metrics (t50, t90) based on volume concentration are still valid. Ignore diameter values after settle start — they likely reflect measurement noise, not real floc size.",
    },
    {
        "pattern": "growth_rate: run too short",
        "severity": "info",
        "icon": "ℹ️",
        "summary": "Flocculation phase was too short for the standard growth rate window.",
        "action": "Growth rate was computed over the full flocculation phase instead of the first 10 min. The value is still meaningful but covers a different time window.",
    },
]


def classify_flag(flag_text: str) -> dict:
    """Match a quality flag string to its diagnostic rule.

    Returns a dict with severity, icon, summary, action, and the original flag.
    Falls back to a generic 'caution' entry if no rule matches.
    """
    flag_lower = flag_text.lower()
    for rule in DIAGNOSTIC_RULES:
        if rule["pattern"].lower() in flag_lower:
            return {
                "severity": rule["severity"],
                "icon": rule["icon"],
                "summary": rule["summary"],
                "action": rule["action"],
                "raw": flag_text,
            }
    # Fallback
    return {
        "severity": "caution",
        "icon": "⚠️",
        "summary": flag_text,
        "action": "Review the raw data for anomalies.",
        "raw": flag_text,
    }
