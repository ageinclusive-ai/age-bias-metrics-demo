# Job Bias Advert Checker

A notebooks-first project to **inspect job adverts for potential age-related bias** (e.g., “digital native”, “energetic”, “recent graduate”) and produce simple summaries/reports.

---

## What’s in this repo
- `notebooks/` — Jupyter notebooks (start here)
- `data/` — small, shareable inputs (e.g., `lexicon.csv` with flagged terms)
- `src/age_inclusive_ai/` — helper Python scripts  
  - `app.py` — main helpers / functions  
  - `batch_runner.py` — run checks over many adverts (CSV or folder)  
  - `html_report.py` — build a simple HTML summary
- `scripts/windows/` (optional) — `run.bat`, `run_batch_csv.bat`
- `requirements.txt` — Python packages used by the notebooks

---

## Quick start (no terminal needed)
1. Open your Python/Jupyter environment (Anaconda Navigator, JupyterLab, or VS Code).
2. Install the packages listed in **`requirements.txt`** using your environment’s package manager  
   *(in Anaconda: Environments → your env → search and install; or open a terminal and run `pip install -r requirements.txt`)*.
3. Open a notebook in **`notebooks/`** and run the cells.

> Tip: keep big/private data **off GitHub**. Only small samples go in `data/`.

---

## Using the helper scripts (optional)
- **From a notebook:** import functions from `src/age_inclusive_ai/*` and call them.
- **One-click on Windows:** double-click `scripts/windows/run.bat`  
  (or `run_batch_csv.bat` to process a CSV or a folder of adverts).
- **HTML report:** `html_report.py` can generate a simple summary page of flagged terms.

---

## Data
- **`data/lexicon.csv`** — example terms/phrases commonly linked to age bias.  
  Edit or extend this list as you learn (add/remove rows).
- If you have your own adverts, place a small sample CSV in `data/`  
  (e.g., columns: `advert_id`, `text`).

---

## Roadmap
- Add a notebook that loads adverts from CSV and produces a bias summary table
- Tune the lexicon (weights, context rules) and reduce false positives
- Optional: export a clean HTML/PDF report per advert

---

## License
MIT (see `LICENSE`).
