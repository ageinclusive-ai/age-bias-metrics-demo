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
