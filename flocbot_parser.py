"""
flocbot_parser.py – Parse RoboJar/FlocBot Excel exports into standardized DataFrames.
"""

import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RunMetadata:
    filename: str = ""
    generated_timestamp: str = ""
    protocol_title: str = ""
    run_chemistry: str = ""
    run_dosage: str = ""
    comments: str = ""
    warnings: list = field(default_factory=list)

    @property
    def label(self) -> str:
        parts = []
        if self.run_dosage:
            parts.append(f"Dose {self.run_dosage}")
        if self.run_chemistry:
            parts.append(self.run_chemistry)
        if self.protocol_title:
            parts.append(self.protocol_title)
        return " | ".join(parts) if parts else self.filename


# ---------------------------------------------------------------------------
# Column name mapping – normalizes whatever the export provides
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "elapsed time (s)": "time_s",
    "elapsed time(s)": "time_s",
    "elapsed_time_s": "time_s",
    "floc count(per ml)": "floc_count_ml",
    "floc count (per ml)": "floc_count_ml",
    "floc_count_per_ml": "floc_count_ml",
    "mean diameter(μm)": "diameter_um",
    "mean diameter (μm)": "diameter_um",
    "mean diameter(um)": "diameter_um",
    "mean diameter (um)": "diameter_um",
    "mean_diameter_um": "diameter_um",
    "mean volume(mm3)": "mean_vol_mm3",
    "mean volume (mm3)": "mean_vol_mm3",
    "vol. concentration(mm3/l)": "vol_conc_mm3_L",
    "vol. concentration (mm3/l)": "vol_conc_mm3_L",
    "vol concentration(mm3/l)": "vol_conc_mm3_L",
    "vol_concentration_mm3_l": "vol_conc_mm3_L",
    "rpm": "rpm",
    "g value (sec-1)": "g_s_1",
    "g value(sec-1)": "g_s_1",
    "g_value_sec_1": "g_s_1",
    "number of samples": "n_samples",
    "number_of_samples": "n_samples",
}


def _normalize_col(name: str) -> str:
    """Return standardized column name or the original lowered string."""
    key = name.strip().lower()
    return COLUMN_MAP.get(key, key)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
_META_PATTERNS = {
    "protocol_title": re.compile(r"Protocol\s+Title\s*:\s*(.+?)(?:\||$)", re.I),
    "run_chemistry": re.compile(r"Run\s+Chemistry\s*:\s*(.+?)(?:\||$)", re.I),
    "run_dosage": re.compile(r"Run\s+Dosage\s*:\s*(.+?)(?:\||$)", re.I),
    "comments": re.compile(r"Comments\s*:\s*(.+?)(?:\||$)", re.I),
}

_GENERATED_RE = re.compile(r"Generated\s*:\s*(.+)", re.I)


def _extract_metadata(raw_rows: list[list], filename: str) -> RunMetadata:
    meta = RunMetadata(filename=filename)
    # Combine the first few rows into strings for searching
    for row in raw_rows[:5]:
        line = " ".join(str(c) for c in row if pd.notna(c))
        # Generated timestamp
        m = _GENERATED_RE.search(line)
        if m:
            meta.generated_timestamp = m.group(1).strip()
        # Other fields
        for attr, pat in _META_PATTERNS.items():
            m = pat.search(line)
            if m:
                setattr(meta, attr, m.group(1).strip())
    return meta


# ---------------------------------------------------------------------------
# Header row detection
# ---------------------------------------------------------------------------
def _find_header_row(df_raw: pd.DataFrame) -> Optional[int]:
    """Find the row index where columns start with Date, Time, ..."""
    for idx in range(min(20, len(df_raw))):
        row_vals = [str(v).strip().lower() for v in df_raw.iloc[idx] if pd.notna(v)]
        if "date" in row_vals and "time" in row_vals:
            return idx
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_file(uploaded_file) -> tuple[pd.DataFrame, RunMetadata]:
    """
    Parse a single RoboJar Excel export.

    Returns (dataframe, metadata).
    Raises ValueError with a descriptive message on failure.
    """
    fname = getattr(uploaded_file, "name", str(uploaded_file))
    is_xls = fname.lower().endswith(".xls")

    # Read the raw sheet – pick the first (and usually only) sheet
    try:
        engine = "xlrd" if is_xls else "openpyxl"
        raw = pd.read_excel(
            uploaded_file,
            sheet_name=0,
            header=None,
            engine=engine,
        )
    except Exception as e:
        raise ValueError(f"Cannot read '{fname}': {e}")

    # --- metadata from the first rows ---
    raw_rows = raw.head(10).values.tolist()
    meta = _extract_metadata(raw_rows, fname)

    # --- locate header row ---
    header_idx = _find_header_row(raw)
    if header_idx is None:
        raise ValueError(
            f"'{fname}': Could not find header row containing 'Date' and 'Time'."
        )

    # Re-read with correct header
    try:
        uploaded_file.seek(0)  # rewind
    except Exception:
        pass
    df = pd.read_excel(
        uploaded_file,
        sheet_name=0,
        header=header_idx,
        engine=engine,
    )

    # Drop fully-empty rows
    df.dropna(how="all", inplace=True)

    # --- normalize column names ---
    df.columns = [_normalize_col(c) for c in df.columns]

    # --- build time_s / time_min ---
    if "time_s" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    else:
        # Try to derive from Date + Time columns
        meta.warnings.append("'Elapsed Time (s)' column not found; attempting derivation from Date/Time.")
        if "date" in df.columns and "time" in df.columns:
            try:
                dt_strings = df["date"].astype(str) + " " + df["time"].astype(str)
                # Strip timezone abbreviations like MST, MDT, EST ...
                dt_strings = dt_strings.str.replace(
                    r"\s+[A-Z]{2,4}\s*$", "", regex=True
                )
                dts = pd.to_datetime(dt_strings, errors="coerce")
                df["time_s"] = (dts - dts.iloc[0]).dt.total_seconds()
            except Exception:
                raise ValueError(
                    f"'{fname}': Cannot determine elapsed time – "
                    "no 'Elapsed Time (s)' column and Date/Time parsing failed."
                )
        else:
            raise ValueError(f"'{fname}': No time columns found.")

    df["time_min"] = df["time_s"] / 60.0

    # Ensure numeric columns
    for col in [
        "diameter_um", "floc_count_ml", "vol_conc_mm3_L",
        "rpm", "g_s_1", "n_samples", "mean_vol_mm3",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where time is NaN (junk rows)
    df = df.dropna(subset=["time_s"]).reset_index(drop=True)

    return df, meta
