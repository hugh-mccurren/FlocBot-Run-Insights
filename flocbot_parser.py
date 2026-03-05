"""
flocbot_parser.py – Parse RoboJar/FlocBot Excel exports into standardized DataFrames.
"""
# v3 – multi-sheet auto-detection, .xls corruption fallback, import_debug

import io
import re
import shutil
import subprocess
import tempfile
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
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
    import_debug: Optional[dict] = None   # sheet-selection info (None = not yet set)

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

    @property
    def short_label(self) -> str:
        """Concise label for chart legends — dose only, with fallbacks."""
        if self.run_dosage:
            return f"Dose {self.run_dosage}"
        if self.protocol_title:
            return self.protocol_title
        return self.filename


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


# Columns the pipeline requires — used for sheet scoring
_REQUIRED_COLS = {"time_s", "diameter_um"}
_DESIRED_COLS = {"time_s", "diameter_um", "rpm", "vol_conc_mm3_L",
                 "floc_count_ml", "g_s_1", "mean_vol_mm3", "n_samples"}


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
# Sheet scoring — decides which sheet has RoboJar time-series data
# ---------------------------------------------------------------------------
@dataclass
class _SheetScore:
    """Scoring result for one sheet."""
    sheet_name: str
    header_row: Optional[int]
    matched_required: set        # subset of _REQUIRED_COLS found
    matched_desired: set         # subset of _DESIRED_COLS found
    numeric_rows: int            # rows with at least one numeric value below header
    missing_required: set        # required cols NOT found
    missing_desired: set         # desired cols NOT found

    @property
    def is_valid(self) -> bool:
        """Must have header row + Date/Time + at least one required col."""
        return self.header_row is not None and len(self.matched_required) > 0

    @property
    def sort_key(self) -> tuple:
        """Higher is better: (required_count, desired_count, numeric_rows)."""
        return (len(self.matched_required), len(self.matched_desired), self.numeric_rows)


def _score_sheet(sheet_df: pd.DataFrame, sheet_name: str) -> _SheetScore:
    """Score a raw (header=None) DataFrame for RoboJar column presence."""
    header_idx = _find_header_row(sheet_df)
    if header_idx is None:
        return _SheetScore(
            sheet_name=sheet_name,
            header_row=None,
            matched_required=set(),
            matched_desired=set(),
            numeric_rows=0,
            missing_required=_REQUIRED_COLS.copy(),
            missing_desired=_DESIRED_COLS.copy(),
        )

    # Normalize the header-row cell values to our standard names
    header_cells = sheet_df.iloc[header_idx]
    normalized = {_normalize_col(str(c)) for c in header_cells if pd.notna(c)}

    matched_req = _REQUIRED_COLS & normalized
    matched_des = _DESIRED_COLS & normalized

    # Also accept Date+Time as a proxy for time_s (we derive it later)
    raw_lower = {str(c).strip().lower() for c in header_cells if pd.notna(c)}
    if "time_s" not in matched_req and "date" in raw_lower and "time" in raw_lower:
        matched_req.add("time_s")
        matched_des.add("time_s")

    # Count numeric data rows below the header
    data_rows = sheet_df.iloc[header_idx + 1: header_idx + 51]  # sample up to 50
    numeric_rows = 0
    for _, row in data_rows.iterrows():
        vals = [v for v in row if pd.notna(v)]
        if any(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            numeric_rows += 1

    return _SheetScore(
        sheet_name=sheet_name,
        header_row=header_idx,
        matched_required=matched_req,
        matched_desired=matched_des,
        numeric_rows=numeric_rows,
        missing_required=_REQUIRED_COLS - matched_req,
        missing_desired=_DESIRED_COLS - matched_des,
    )


# ---------------------------------------------------------------------------
# Workbook loader — centralized engine selection + .xls fallback
# ---------------------------------------------------------------------------
def load_robojar_workbook(uploaded_file) -> tuple[pd.ExcelFile, str, list[str]]:
    """
    Open the uploaded Excel file and return (ExcelFile, engine_used, warnings).

    Handles:
    - .xlsx → openpyxl
    - .xls  → xlrd, with fallback to openpyxl if xlrd chokes (corruption)
    - If xlrd fails, tries server-side LibreOffice conversion to .xlsx

    Raises ValueError with a user-friendly message on total failure.
    """
    fname = getattr(uploaded_file, "name", str(uploaded_file))
    is_xls = fname.lower().endswith(".xls")
    warnings: list[str] = []

    def _rewind():
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    # --- Attempt 1: primary engine ---
    engine = "xlrd" if is_xls else "openpyxl"
    try:
        _rewind()
        xf = pd.ExcelFile(uploaded_file, engine=engine)
        return xf, engine, warnings
    except Exception as primary_err:
        if not is_xls:
            # .xlsx failed with openpyxl — nothing else to try
            raise ValueError(
                f"Unable to read '{fname}'. The file may be damaged or not a "
                f"valid Excel file. Please re-export from RoboJar and try again."
            ) from primary_err
        # .xls failed — try fallbacks
        warnings.append(f"Primary reader (xlrd) could not open file: {_sanitize_error(primary_err)}")

    # --- Attempt 2 (.xls only): xlrd with ignore_workbook_corruption ---
    # Many RoboJar .xls exports have minor container-level corruption
    # (e.g., non-standard sector sizes) that xlrd rejects by default
    # but the actual data is perfectly fine.
    try:
        import xlrd
        _rewind()
        file_bytes = uploaded_file.read()
        wb = xlrd.open_workbook(file_contents=file_bytes, ignore_workbook_corruption=True)
        xf = pd.ExcelFile(wb, engine="xlrd")
        warnings.append("File opened with xlrd (minor format issues ignored).")
        return xf, "xlrd", warnings
    except Exception:
        pass

    # --- Attempt 4 (.xls only): try openpyxl directly ---
    # Some .xls files are actually .xlsx with wrong extension
    try:
        _rewind()
        xf = pd.ExcelFile(uploaded_file, engine="openpyxl")
        warnings.append("File opened with openpyxl (may be .xlsx saved with .xls extension).")
        return xf, "openpyxl", warnings
    except Exception:
        pass

    # --- Attempt 5 (.xls only): LibreOffice conversion ---
    if shutil.which("libreoffice") or shutil.which("soffice"):
        try:
            converted = _convert_xls_to_xlsx(uploaded_file, fname)
            if converted is not None:
                xf = pd.ExcelFile(converted, engine="openpyxl")
                warnings.append("File was converted from .xls to .xlsx (LibreOffice) for compatibility.")
                return xf, "openpyxl", warnings
        except Exception:
            pass

    # --- All attempts failed ---
    raise ValueError(
        f"Unable to read '{fname}'. This legacy .xls file could not be opened. "
        f"Please open the file in Excel, choose File → Save As → "
        f"Excel Workbook (.xlsx), and re-upload the .xlsx version."
    )


def _sanitize_error(err: Exception) -> str:
    """Strip scary internal wording from error messages shown to users."""
    msg = str(err)
    # Remove xlrd's alarming "workbook corruption" phrasing
    msg = re.sub(r"(?i)workbook\s+corrupt(ion|ed)?", "file format issue", msg)
    msg = re.sub(r"(?i)corrupt(ion|ed)?", "format issue", msg)
    # Truncate overly long messages
    if len(msg) > 200:
        msg = msg[:200] + "…"
    return msg


def _convert_xls_to_xlsx(uploaded_file, fname: str) -> Optional[io.BytesIO]:
    """Try to convert .xls to .xlsx via LibreOffice headless. Returns BytesIO or None."""
    lo = shutil.which("libreoffice") or shutil.which("soffice")
    if not lo:
        return None

    try:
        uploaded_file.seek(0)
    except Exception:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / fname
        src.write_bytes(uploaded_file.read())

        try:
            subprocess.run(
                [lo, "--headless", "--convert-to", "xlsx", "--outdir", tmpdir, str(src)],
                capture_output=True, timeout=30, check=True,
            )
        except (subprocess.SubprocessError, OSError):
            return None

        xlsx_path = src.with_suffix(".xlsx")
        if xlsx_path.exists():
            return io.BytesIO(xlsx_path.read_bytes())
    return None


# ---------------------------------------------------------------------------
# Sheet selection — scan all sheets, pick the best RoboJar data sheet
# ---------------------------------------------------------------------------
def _select_data_sheet(
    xf: pd.ExcelFile,
) -> tuple[str, _SheetScore, list[_SheetScore]]:
    """
    Scan every sheet in the workbook and return (chosen_sheet_name, best_score, all_scores).
    Raises ValueError if no sheet contains valid RoboJar data.
    """
    sheet_names = xf.sheet_names
    all_scores: list[_SheetScore] = []

    for sn in sheet_names:
        try:
            raw = pd.read_excel(xf, sheet_name=sn, header=None, nrows=70)
        except Exception:
            # Sheet is unreadable — skip
            all_scores.append(_SheetScore(
                sheet_name=sn, header_row=None,
                matched_required=set(), matched_desired=set(),
                numeric_rows=0, missing_required=_REQUIRED_COLS.copy(),
                missing_desired=_DESIRED_COLS.copy(),
            ))
            continue
        all_scores.append(_score_sheet(raw, sn))

    valid = [s for s in all_scores if s.is_valid]
    if not valid:
        sheet_list = ", ".join(f"'{s.sheet_name}'" for s in all_scores)
        raise ValueError(
            f"No sheet contains the expected RoboJar time-series columns. "
            f"Sheets found: {sheet_list}. "
            f"Each sheet needs at least a header row with Date, Time, and "
            f"measurement columns (e.g. Mean Diameter, RPM)."
        )

    # Pick best: most required cols → most desired cols → most numeric rows
    best = max(valid, key=lambda s: s.sort_key)
    return best.sheet_name, best, all_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_file(uploaded_file) -> tuple[pd.DataFrame, RunMetadata]:
    """
    Parse a single RoboJar Excel export.

    Handles multi-sheet workbooks (auto-detects the data sheet) and
    legacy .xls files that fail with xlrd.

    Returns (dataframe, metadata).
    Raises ValueError with a descriptive, user-friendly message on failure.
    """
    fname = getattr(uploaded_file, "name", str(uploaded_file))

    # ── Step 1: Open the workbook ──
    xf, engine_used, load_warnings = load_robojar_workbook(uploaded_file)

    # ── Step 2: Find the correct data sheet ──
    chosen_sheet, best_score, all_scores = _select_data_sheet(xf)

    # Build import debug info
    debug = {
        "engine": engine_used,
        "sheets_found": [s.sheet_name for s in all_scores],
        "sheet_scores": {
            s.sheet_name: {
                "valid": s.is_valid,
                "required_matched": sorted(s.matched_required),
                "desired_matched": sorted(s.matched_desired),
                "missing_required": sorted(s.missing_required),
                "missing_desired": sorted(s.missing_desired),
                "numeric_rows": s.numeric_rows,
                "header_row": s.header_row,
            }
            for s in all_scores
        },
        "chosen_sheet": chosen_sheet,
    }

    # ── Step 3: Read the chosen sheet (raw, for metadata) ──
    raw = pd.read_excel(xf, sheet_name=chosen_sheet, header=None)

    # --- metadata from the first rows ---
    raw_rows = raw.head(10).values.tolist()
    meta = _extract_metadata(raw_rows, fname)
    meta.import_debug = debug
    meta.warnings.extend(load_warnings)

    if len(all_scores) > 1:
        meta.warnings.append(
            f"Multi-sheet workbook: selected sheet '{chosen_sheet}' "
            f"({len(best_score.matched_desired)} data columns matched)."
        )

    # --- locate header row ---
    header_idx = best_score.header_row
    if header_idx is None:
        raise ValueError(
            f"'{fname}': Could not find header row containing 'Date' and 'Time'."
        )

    # ── Step 4: Re-read with correct header ──
    df = pd.read_excel(xf, sheet_name=chosen_sheet, header=header_idx)

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
