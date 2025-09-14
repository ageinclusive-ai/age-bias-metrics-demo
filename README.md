# Age‑Inclusive AI — Project Template

A reusable, ethics‑first template for data projects on age inclusion and AI fairness.

## Why this template
- **Ethics‑first**: prompts you to document consent, bias risks, and mitigations
- **Reproducible**: clear structure for data, code, and notebooks
- **Portable**: CI checks, pre‑commit hooks, and standard Python tooling

## Getting started
```bash
# Clone this template into a new project (replace NEW_PROJECT)
git clone git@github.com:YOURORG/age-inclusive-ai-template.git NEW_PROJECT
cd NEW_PROJECT
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
pytest -q
```

## Project layout
- `data/` holds **no raw personal data**. Only pointers and derived, privacy‑safe artifacts.
- `notebooks/` for exploration and reporting (keep small; push heavy outputs to `reports/` if added).
- `src/` for reusable Python modules.
- `tests/` for sanity checks.

## Ethics & data use
- Document data sources in `data/README.md`.
- Note inclusion/exclusion criteria and potential **age bias**.
- If any sensitive attributes are used for fairness evaluation, store **only aggregates** in the repo.

## License
- Code: MIT (see `LICENSE`)
- Docs/Notebooks: CC BY 4.0 (see `LICENSE-docs`)

## Citation
If you use this template, please cite:
```yaml
# See CITATION.cff
```
