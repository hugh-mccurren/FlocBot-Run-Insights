"""
test_parser.py – Verification script for flocbot_parser multi-sheet + .xls robustness.

Creates synthetic Excel workbooks in memory and verifies that parse_file:
  1. Single-sheet .xlsx works as before
  2. Multi-sheet workbook: picks the correct data sheet (ignores Summary/Settings)
  3. Data sheet is NOT the first sheet
  4. Multiple candidate sheets: picks the one with the most columns
  5. No valid sheet raises ValueError with friendly message
  6. import_debug is populated correctly
  7. .xls extension triggers xlrd engine (or fallback)
"""

import io
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is importable regardless of where the test is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flocbot_parser import parse_file, RunMetadata


# ---------------------------------------------------------------------------
# Helpers to create synthetic RoboJar workbooks
# ---------------------------------------------------------------------------
def _make_robojar_sheet_data(n_rows=20):
    """Return a dict of lists mimicking a RoboJar data sheet."""
    # Metadata rows (will be rows 0-2 in the sheet)
    # The actual column header row (Date, Time, etc.) is row 3
    t = np.arange(n_rows) * 30.0  # 30-second intervals
    return {
        "Date": ["2024-01-15"] * n_rows,
        "Time": [f"10:{int(s//60):02d}:{int(s%60):02d}" for s in t],
        "Elapsed Time (s)": t,
        "Mean Diameter(μm)": 100 + np.random.rand(n_rows) * 300,
        "RPM": [200] * 5 + [50] * 10 + [0] * 5,
        "Vol. Concentration(mm3/l)": np.random.rand(n_rows) * 10,
        "Floc Count(per ml)": np.random.randint(100, 5000, n_rows),
        "G Value (sec-1)": np.random.rand(n_rows) * 100,
        "Number of Samples": [10] * n_rows,
    }


def _make_metadata_rows():
    """Return a DataFrame of metadata rows to prepend above the header."""
    return pd.DataFrame([
        ["Generated: 2024-01-15 10:30:00", None, None, None, None, None, None, None, None],
        ["Protocol Title: Test Protocol | Run Chemistry: Alum | Run Dosage: 25 mg/L", None, None, None, None, None, None, None, None],
        ["Comments: Test run", None, None, None, None, None, None, None, None],
    ])


def _write_workbook(sheets: dict[str, pd.DataFrame], fname="test.xlsx") -> io.BytesIO:
    """Write multiple sheets to an in-memory .xlsx file."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False, header=True)
    buf.seek(0)
    buf.name = fname
    return buf


def _write_workbook_with_meta(sheets: dict[str, pd.DataFrame], fname="test.xlsx") -> io.BytesIO:
    """Write workbook with metadata rows prepended to data sheets."""
    buf = io.BytesIO()
    meta_rows = _make_metadata_rows()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            # For data sheets (ones with Date column), prepend metadata rows
            if "Date" in df.columns:
                # Write metadata rows + empty row + header + data
                # We need to write raw to get the meta rows above the header
                combined = pd.concat([meta_rows, df], ignore_index=True)
                combined.to_excel(writer, sheet_name=name, index=False, header=False)
            else:
                df.to_excel(writer, sheet_name=name, index=False, header=True)
    buf.seek(0)
    buf.name = fname
    return buf


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_single_sheet():
    """Standard single-sheet .xlsx import."""
    data = pd.DataFrame(_make_robojar_sheet_data())
    buf = _write_workbook({"Data": data})

    df, meta = parse_file(buf)
    assert "time_s" in df.columns, "time_s column missing"
    assert "diameter_um" in df.columns, "diameter_um column missing"
    assert len(df) > 0, "DataFrame is empty"
    assert meta.import_debug.get("chosen_sheet") == "Data"
    assert meta.import_debug.get("engine") == "openpyxl"
    print(f"  PASS: Single-sheet: {len(df)} rows, sheet='{meta.import_debug['chosen_sheet']}'")


def test_multi_sheet_data_first():
    """Multi-sheet workbook where data sheet is first."""
    data = pd.DataFrame(_make_robojar_sheet_data())
    summary = pd.DataFrame({"Metric": ["Score", "Grade"], "Value": [85, "A"]})
    settings = pd.DataFrame({"Setting": ["Mode"], "Config": ["Auto"]})

    buf = _write_workbook({"Run Data": data, "Summary": summary, "Settings": settings})

    df, meta = parse_file(buf)
    assert meta.import_debug["chosen_sheet"] == "Run Data"
    assert len(meta.import_debug["sheets_found"]) == 3
    assert len(df) > 0
    print(f"  PASS: Multi-sheet (data first): chose '{meta.import_debug['chosen_sheet']}' from {meta.import_debug['sheets_found']}")


def test_multi_sheet_data_not_first():
    """Multi-sheet workbook where data sheet is NOT the first sheet (the Northglenn bug)."""
    summary = pd.DataFrame({"Metric": ["Score", "Grade"], "Value": [85, "A"]})
    data = pd.DataFrame(_make_robojar_sheet_data())
    settings = pd.DataFrame({"Setting": ["Mode"], "Config": ["Auto"]})

    buf = _write_workbook({"Summary": summary, "Run Data": data, "Settings": settings})

    df, meta = parse_file(buf)
    assert meta.import_debug["chosen_sheet"] == "Run Data", \
        f"Expected 'Run Data', got '{meta.import_debug['chosen_sheet']}'"
    assert len(df) > 0
    print(f"  PASS: Multi-sheet (data NOT first): chose '{meta.import_debug['chosen_sheet']}' from {meta.import_debug['sheets_found']}")


def test_multi_sheet_best_match():
    """Two sheets with headers: pick the one with more matching columns."""
    # Sheet with only Date + Time (minimal)
    minimal = pd.DataFrame({
        "Date": ["2024-01-15"] * 5,
        "Time": ["10:00:00"] * 5,
        "Elapsed Time (s)": [0, 30, 60, 90, 120],
        "Some Other Col": [1, 2, 3, 4, 5],
    })
    # Sheet with full RoboJar columns
    full = pd.DataFrame(_make_robojar_sheet_data())

    buf = _write_workbook({"Partial": minimal, "Full Data": full})

    df, meta = parse_file(buf)
    assert meta.import_debug["chosen_sheet"] == "Full Data", \
        f"Expected 'Full Data', got '{meta.import_debug['chosen_sheet']}'"
    print(f"  PASS: Best-match selection: chose '{meta.import_debug['chosen_sheet']}'")


def test_no_valid_sheet():
    """No sheet has RoboJar data -> friendly error."""
    summary = pd.DataFrame({"Metric": ["Score"], "Value": [85]})
    settings = pd.DataFrame({"Setting": ["Mode"], "Config": ["Auto"]})

    buf = _write_workbook({"Summary": summary, "Settings": settings})

    try:
        parse_file(buf)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        msg = str(e)
        assert "No sheet contains" in msg, f"Unexpected error message: {msg}"
        assert "corruption" not in msg.lower(), "Error message contains scary wording"
        print(f"  PASS: No valid sheet: raised ValueError with friendly message")


def test_debug_info_populated():
    """import_debug has all expected fields."""
    data = pd.DataFrame(_make_robojar_sheet_data())
    buf = _write_workbook({"Data": data})

    df, meta = parse_file(buf)
    dbg = meta.import_debug
    assert "engine" in dbg
    assert "sheets_found" in dbg
    assert "chosen_sheet" in dbg
    assert "sheet_scores" in dbg
    scores = dbg["sheet_scores"]
    assert "Data" in scores
    assert scores["Data"]["valid"] is True
    assert "time_s" in scores["Data"]["required_matched"] or \
           "diameter_um" in scores["Data"]["required_matched"]
    print(f"  PASS: Debug info populated: {list(dbg.keys())}")


def test_xls_extension_handling():
    """Verify .xls extension triggers xlrd engine selection (or graceful fallback)."""
    data = pd.DataFrame(_make_robojar_sheet_data())
    # Write as .xlsx but name it .xls -- tests engine selection logic
    buf = _write_workbook({"Data": data}, fname="legacy_file.xls")

    try:
        df, meta = parse_file(buf)
        # If it succeeds, it used the openpyxl fallback (since the file is really .xlsx)
        assert meta.import_debug.get("engine") == "openpyxl"
        assert any("openpyxl" in w for w in meta.warnings)
        print(f"  PASS: .xls extension: fell back to openpyxl (file is actually .xlsx)")
    except ValueError as e:
        msg = str(e)
        assert "corruption" not in msg.lower(), "Error contains scary wording"
        assert "Save As" in msg or "re-upload" in msg.lower()
        print(f"  PASS: .xls extension: raised friendly error for conversion")


def test_metadata_extraction():
    """Metadata is correctly extracted from the data sheet (raw write with meta rows)."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    # Write metadata rows first
    ws.append(["Generated: 2024-01-15 10:30:00"])
    ws.append(["Protocol Title: Test Protocol | Run Chemistry: Alum | Run Dosage: 25 mg/L"])
    ws.append(["Comments: Test run"])
    # Write header row
    headers = ["Date", "Time", "Elapsed Time (s)", "Mean Diameter(um)",
               "RPM", "Vol. Concentration(mm3/l)", "Floc Count(per ml)",
               "G Value (sec-1)", "Number of Samples"]
    ws.append(headers)
    # Write data rows
    for i in range(20):
        t = i * 30
        ws.append([
            "2024-01-15", f"10:{t//60:02d}:{t%60:02d}", t,
            100 + i * 15, 200 if i < 5 else (50 if i < 15 else 0),
            round(1.0 + i * 0.3, 2), 1000 + i * 100, round(50.0 + i * 2, 1), 10,
        ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    buf.name = "meta_test.xlsx"

    df, meta = parse_file(buf)
    assert len(df) > 0, "DataFrame is empty"
    assert meta.protocol_title == "Test Protocol", f"Protocol: '{meta.protocol_title}'"
    assert meta.run_chemistry == "Alum", f"Chemistry: '{meta.run_chemistry}'"
    assert meta.run_dosage == "25 mg/L", f"Dosage: '{meta.run_dosage}'"
    print(f"  PASS: Metadata extraction: protocol='{meta.protocol_title}', "
          f"chemistry='{meta.run_chemistry}', dosage='{meta.run_dosage}'")


# ---------------------------------------------------------------------------
# Real-file tests (skip gracefully if files not found)
# ---------------------------------------------------------------------------
def _load_real_file(path, name):
    """Load a real file as a BytesIO with .name attribute."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    buf = io.BytesIO(p.read_bytes())
    buf.name = name
    return buf


def test_real_northglenn():
    """Operator Northglenn Data.xls -- multi-sheet + xlrd corruption."""
    buf = _load_real_file(
        r"C:\Users\mccur\Downloads\Operator Northglenn Data.xls",
        "Operator Northglenn Data.xls",
    )
    if buf is None:
        print("  SKIP: Northglenn file not found at expected path")
        return

    df, meta = parse_file(buf)
    assert len(df) > 200, f"Expected 200+ rows, got {len(df)}"
    assert "diameter_um" in df.columns, "diameter_um missing"
    assert "rpm" in df.columns, "rpm missing"
    dbg = meta.import_debug
    assert len(dbg["sheets_found"]) == 2, f"Expected 2 sheets, got {dbg['sheets_found']}"
    assert meta.protocol_title, "Protocol title should be extracted"
    assert meta.run_dosage, "Dosage should be extracted"
    print(f"  PASS: Northglenn: {len(df)} rows, "
          f"sheet='{dbg['chosen_sheet']}', "
          f"protocol='{meta.protocol_title}', "
          f"dosage='{meta.run_dosage}'")


def test_real_demo_run_a():
    """Demo Run A.xls -- single-sheet, standard .xls."""
    buf = _load_real_file(
        r"C:\Users\mccur\OneDrive\Business\Demo Run A.xls",
        "Demo Run A.xls",
    )
    if buf is None:
        print("  SKIP: Demo Run A file not found at expected path")
        return

    df, meta = parse_file(buf)
    assert len(df) > 100, f"Expected 100+ rows, got {len(df)}"
    assert "diameter_um" in df.columns
    dbg = meta.import_debug
    assert len(dbg["sheets_found"]) == 1, f"Expected 1 sheet, got {dbg['sheets_found']}"
    assert dbg["engine"] == "xlrd", f"Expected xlrd engine, got {dbg['engine']}"
    assert meta.protocol_title == "Standard Protocol"
    print(f"  PASS: Demo Run A: {len(df)} rows, "
          f"sheet='{dbg['chosen_sheet']}', "
          f"protocol='{meta.protocol_title}', "
          f"dosage='{meta.run_dosage}'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("flocbot_parser verification tests")
    print("=" * 60)

    tests = [
        ("Single-sheet .xlsx", test_single_sheet),
        ("Multi-sheet (data first)", test_multi_sheet_data_first),
        ("Multi-sheet (data NOT first -- Northglenn bug)", test_multi_sheet_data_not_first),
        ("Multi-sheet best-match selection", test_multi_sheet_best_match),
        ("No valid sheet -> friendly error", test_no_valid_sheet),
        ("import_debug populated", test_debug_info_populated),
        (".xls extension handling", test_xls_extension_handling),
        ("Metadata extraction", test_metadata_extraction),
        ("REAL: Operator Northglenn Data.xls", test_real_northglenn),
        ("REAL: Demo Run A.xls", test_real_demo_run_a),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n> {name}")
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"  FAIL: FAILED")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
