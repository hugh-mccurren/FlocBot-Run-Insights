# FlocBot Run Insights

Operator-friendly run summaries and comparisons for RoboJar/FlocBot Excel exports.

## Quick Start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

1. **Upload** one or more `.xls` / `.xlsx` RoboJar export files using the sidebar.
2. View the **Run Summary Table** with all KPIs and scores at a glance.
3. Click on individual run tabs for detailed plots and metrics.
4. Use the **Run Comparison** section to overlay two runs side-by-side.
5. **Export** results as CSV or JSON using the download buttons.

## Assumptions

- Each Excel file contains one sheet with the run data.
- Row 1 is a title/generated line; Row 2 contains pipe-separated metadata.
- The actual data table starts at the row where column A = "Date" and column B = "Time".
- Phase detection relies on RPM changes: highest RPM = rapid mix, lower RPM > 0 = flocculation, RPM = 0 = settling.
- Growth rate is computed via linear fit over a default window of flocculation time (start+1 min to 10 min or end-1 min).
- Settling metrics use Vol. Concentration, comparing against a baseline from the last 60 seconds before settling.

## Adjustable Parameters (in sidebar)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Diameter thresholds | 250, 300, 350, 400, 450 μm | Target sizes to measure time-to-reach |
| Time-to-threshold weight | 30% | Score weight for speed of floc formation |
| Pre-settle diameter weight | 30% | Score weight for max achieved floc size |
| Plateau CV weight | 20% | Score weight for floc size stability |
| Settling t50 weight | 20% | Score weight for settling speed |
| Scoring threshold | 300 μm | Which diameter threshold to use in the score |

## File Structure

```
app.py                 – Streamlit UI
flocbot_parser.py      – Excel parsing and metadata extraction
flocbot_metrics.py     – Phase detection, KPIs, and scoring
requirements.txt       – Python dependencies
README.md              – This file
```

## Scoring

The Operator Score (0–100) combines four normalized components:

- **Time to threshold**: How quickly flocs reach a target diameter (lower is better)
- **Pre-settle diameter**: Average floc size just before settling (higher is better)
- **Plateau CV**: Coefficient of variation in floc size at end of flocculation (lower = more stable)
- **Settling t50**: Time for vol. concentration to drop to 50% of baseline (lower is better)

Each component is scaled to 0–100 and combined using adjustable weights. If some KPIs are unavailable, the score is computed from the remaining components and flagged as partial.
