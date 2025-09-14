# Job Bias Advert Checker

A notebooks-first project to **inspect job adverts for potential age-related bias** (e.g., “digital native”, “energetic”, “recent graduate”) and produce simple reports.

## What’s in this repo
- `notebooks/` — Jupyter notebooks for exploration and reporting
- `data/` — small, shareable inputs (e.g., `lexicon.csv` with flagged terms)
- `src/age_inclusive_ai/` — helper Python scripts:
  - `app.py` — main helpers / functions
  - `batch_runner.py` — process a folder or CSV of adverts in one go
  - `html_report.py` — build a simple HTML summary
- `scripts/windows/` (optional) — `run.bat`, `run_batch_csv.bat`
- `requirements.txt` — Python packages used by the notebooks

## Quick start (simplest path)
1. Open your Python/Jupyter environment (Anaconda, JupyterLab, or VS Code).
2. Install packages (if you use a terminal):
   ```bash
   pip install -r requirements.txt

If you don’t use a terminal, just add the packages listed in requirements.txt via your environment UI.
3. Open a notebook in notebooks/ and run the cells.

Using the helper scripts (optional)

Run analysis from a notebook: import functions from src/age_inclusive_ai/*.

Run in one go on Windows: double-click scripts/windows/run.bat
(or run_batch_csv.bat to process a CSV or folder).

Reports: html_report.py can generate a simple HTML file summarising flagged terms.

Data notes

data/lexicon.csv contains example terms/phrases often associated with age bias.
You can edit or extend this list in place.

Keep private or large datasets off GitHub; only put small, shareable samples in data/.

Roadmap (next steps)

Add a notebook that loads adverts from CSV and produces a bias summary

Tune the lexicon (weights, context rules) and add false-positive handling

Optional: export a clean HTML/PDF report for each advert
