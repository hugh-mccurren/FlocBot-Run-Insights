# FlocBot Run Insights

A Streamlit web app for analyzing **RoboJar / FlocBot** jar-test Excel exports.
Upload one or more `.xls` / `.xlsx` runs and get phase-detected KPIs, scoring,
plots, multi-run comparison, and a saved Run Library per user.

Live deployment runs on Render. Authentication and persistent storage are
handled by Supabase.

---

## Quick start (local)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # then fill in your Supabase keys
streamlit run run_insights.py
```

The app opens at `http://localhost:8501`.

### Environment variables

The app needs a Supabase project for login and per-user data. Copy `.env.example`
to `.env` and set:

| Variable | Where to find it |
|---|---|
| `SUPABASE_URL` | Supabase dashboard → Settings → API → Project URL |
| `SUPABASE_ANON_KEY` | Supabase dashboard → Settings → API → anon/public key |

---

## What's in this repo

### Entry point
| File | Purpose |
|---|---|
| **`run_insights.py`** | The Streamlit app. This is the file you run. Wires together the sidebar, upload flow, summary tables, per-run tabs, comparison view, exports (CSV / JSON / PDF), and the auth gate. |

### Data pipeline
| File | Purpose |
|---|---|
| **`flocbot_parser.py`** | Reads RoboJar Excel exports (`.xls` / `.xlsx`), auto-detects the data sheet in multi-sheet workbooks, falls back through `xlrd` / `openpyxl` engines for corrupt files, and returns a normalized DataFrame plus `RunMetadata`. |
| **`flocbot_metrics.py`** | Detects rapid-mix / flocculation / settling phases from RPM, then computes KPIs (growth rate, pre-settle diameter, signal noise, settling t50, etc.), the 0–100 composite score, and data-quality flags. |
| **`metric_help.py`** | Central dictionary of metric definitions used for tooltips, "How to read this" help text, and flag classification. Keeps all user-facing metric copy in one place. |

### UI / features
| File | Purpose |
|---|---|
| **`ui_operator.py`** | "Operator Mode" dashboard — a simplified traffic-light view of run quality with recommended dose adjustments. Also handles plant-baseline save/load used for baseline-aware scoring. |

### Backend / infra
| File | Purpose |
|---|---|
| **`supabase_client.py`** | Lightweight HTTP wrapper around Supabase Auth (sign-up / sign-in / sign-out) and PostgREST (saved runs, baselines, per-user preferences). Uses `httpx` directly — no heavy SDK. |
| **`keep_alive.py`** | Tiny background thread that pings the Streamlit health endpoint every 5 minutes so the process stays warm on Render's free tier. Idempotent — safe to start multiple times. |
| **`render.yaml`** | Render deployment config (Python 3.11, start command, port binding). |
| **`requirements.txt`** | Python dependencies, all version-pinned. |
| **`.streamlit/config.toml`** | Streamlit theme / server settings. |
| **`.env.example`** | Template for the Supabase environment variables. |

### Tests
| File | Purpose |
|---|---|
| **`tests/test_parser.py`** | Verification suite for the parser — synthetic Excel workbooks covering single-sheet, multi-sheet auto-detection, sheet-priority fallback, and `.xls` engine handling. Run with `python tests/test_parser.py` or `pytest tests/`. |

---

## Using the app

1. **Sign in** (or sign up) on the auth page.
2. **Upload** one or more `.xls` / `.xlsx` exports via the sidebar.
3. View the **Run Summary Table** — all KPIs and scores at a glance.
4. Open per-run tabs for detailed plots and metrics.
5. Use **Run Comparison** to overlay two runs side-by-side.
6. Save runs to your **Run Library** for later, or set a **Plant Baseline** so future runs are scored relative to that benchmark.
7. **Export** results as CSV, JSON, or PDF.

---

## Scoring (0–100)

The composite Operator Score combines four normalized components:

| Component | Weight | Direction |
|---|---|---|
| Time to threshold diameter (default 300 μm) | 30% | lower is better |
| Pre-settle mean diameter | 30% | higher is better |
| Signal noise (MAD-based stability) | 20% | lower is better |
| Settling t50 | 20% | lower is better |

Weights and the threshold diameter are adjustable in the sidebar and saved per
user. If a sub-metric is unavailable for a run, the remaining weights are
re-normalized and the score is flagged as partial.

---

## Assumptions about input data

- Each Excel file contains at least one sheet with run data; multi-sheet
  workbooks are auto-resolved (the parser picks the sheet with the most
  candidate columns).
- Row 1 is a title / generated line; row 2 contains pipe-separated metadata.
- The data table starts where column A = `Date` and column B = `Time`.
- Phase detection: highest RPM = rapid mix, lower non-zero RPM = flocculation,
  RPM = 0 = settling.
- Settling metrics use `Vol. Concentration` against a baseline taken from the
  last 60 seconds before settling begins.
