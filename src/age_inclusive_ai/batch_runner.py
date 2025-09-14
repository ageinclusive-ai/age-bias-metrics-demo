
"""
batch_runner.py
----------------
Batch-test your Bias Checker prompt against multiple adverts and generate per-ad reports.

Usage examples:
  # From a folder of .txt files (one advert per file):
  python batch_runner.py --ads-folder ./ads_folder --system-prompt system_prompt.txt --lexicon lexicon.csv --endpoint http://localhost:1234/v1 --model meta-llama-3.1-8b-instruct

  # From a CSV with a column 'ad_text' (and optional 'ad_id'):
  python batch_runner.py --ads-csv adverts.csv --ad-column ad_text --id-column ad_id --system-prompt system_prompt.txt --lexicon lexicon.csv --endpoint http://localhost:1234/v1 --model meta-llama-3.1-8b-instruct

Output:
  - results/results.csv           → aggregated results table
  - results/<ad_id>_raw.txt       → raw model output
  - results/<ad_id>_parsed.json   → parsed JSON block (flags/score/summary/etc.)
  - results/<ad_id>_rewrite.txt   → rewritten advert
  - results/<ad_id>.pdf           → nicely formatted PDF report (if reportlab is installed)
"""

import argparse
import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

# HTML helper (no dependencies)
from html_report import save_html_report as save_html_report_html

# Optional PDF helper
try:
    from pdf_report import save_pdf_report
except Exception:
    save_pdf_report = None

# ----------------------------
# Utilities
# ----------------------------

def read_lexicon_csv(lexicon_path: Path) -> str:
    """
    Read a lexicon CSV and return a properly quoted CSV string (header included)
    to inject into the system prompt placeholder <<<LEXICON_CSV>>>.
    """
    df = pd.read_csv(lexicon_path)
    required = {"phrase", "why", "alternative", "severity"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Lexicon is missing required columns: {missing}")
    # Reorder columns to a stable order (if present)
    cols = [c for c in ["phrase", "why", "alternative", "severity"] if c in df.columns]
    df = df[cols]
    # Quote via csv module
    buf = StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(cols)
    for _, row in df.iterrows():
        writer.writerow([row[c] for c in cols])
    return buf.getvalue().strip()

def load_system_prompt(path: Path, lexicon_csv_text: str) -> str:
    text = path.read_text(encoding="utf-8")
    return text.replace("<<<LEXICON_CSV>>>", lexicon_csv_text)

def balanced_json_extract(s: str):
    """
    Return the first balanced JSON object found in string s and its (start, end) indices.
    Uses brace counting instead of regex to avoid over/under-matching.
    """
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(s[start:], start=start):
            if ch == '"' and not esc:
                in_str = not in_str
            if ch == "\\" and not esc:
                esc = True
                continue
            else:
                esc = False
            if not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(s[start:i+1])
                            return obj, start, i+1
                        except Exception:
                            break  # not valid JSON; try next '{'
        # try next '{'
        start = s.find("{", start + 1)
    return None, -1, -1

def parse_model_output(text: str):
    """
    Expect format:
      <JSON>
      ---
      <rewrite>
    Returns dict with keys: parsed_json, rewrite, raw_text
    """
    parsed, s, e = balanced_json_extract(text)
    rewrite = None
    if e != -1:
        tail = text[e:].lstrip()
        # Look for delimiter line
        m = re.search(r'^\s*---\s*[\r\n]+', tail)
        if m:
            rewrite = tail[m.end():].strip()
    return {
        "parsed_json": parsed,
        "rewrite": rewrite,
        "raw_text": text,
    }

def call_chat_completion(endpoint: str, model: str, system_prompt: str, ad_text: str, temperature: float = 0.1, seed: int = 42, timeout: int = 300):
    """
    Calls an OpenAI-compatible /chat/completions endpoint (LM Studio).
    """
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ad_text},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
        "seed": seed,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def load_ads_from_folder(folder: Path):
    ads = []
    for p in sorted(folder.glob("*.txt")):
        ad_id = p.stem
        ad_text = p.read_text(encoding="utf-8")
        ads.append({"ad_id": ad_id, "ad_text": ad_text})
    if not ads:
        raise ValueError(f"No .txt files found in {folder}")
    return pd.DataFrame(ads)

def load_ads_from_csv(csv_path: Path, ad_column: str, id_column: str | None):
    df = pd.read_csv(csv_path)
    if ad_column not in df.columns:
        raise ValueError(f"Column '{ad_column}' not in CSV")
    if id_column and id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not in CSV")
    if not id_column:
        df["ad_id"] = [f"ad_{i+1:04d}" for i in range(len(df))]
    else:
        df["ad_id"] = df[id_column].astype(str)
    return df[["ad_id", ad_column]].rename(columns={ad_column: "ad_text"})

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ads-folder", type=str, help="Folder of .txt adverts")
    src.add_argument("--ads-csv", type=str, help="CSV file with advert texts")
    ap.add_argument("--ad-column", type=str, default="ad_text", help="Column in CSV containing advert text")
    ap.add_argument("--id-column", type=str, default=None, help="Optional ID column in CSV")
    ap.add_argument("--system-prompt", type=str, required=True, help="Path to system_prompt.txt")
    ap.add_argument("--lexicon", type=str, required=True, help="Path to lexicon.csv")
    ap.add_argument("--endpoint", type=str, default="http://localhost:1234/v1", help="OpenAI-compatible endpoint")
    ap.add_argument("--model", type=str, required=True, help="Model ID (as shown in LM Studio)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results", help="Output directory")
    ap.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare prompt with injected lexicon
    lexicon_text = read_lexicon_csv(Path(args.lexicon))
    system_prompt = load_system_prompt(Path(args.system_prompt), lexicon_text)

    # Load adverts
    if args.ads_folder:
        df_ads = load_ads_from_folder(Path(args.ads_folder))
    else:
        df_ads = load_ads_from_csv(Path(args.ads_csv), args.ad_column, args.id_column)

    rows = []
    for i, row in df_ads.iterrows():
        ad_id = row["ad_id"]
        ad_text = str(row["ad_text"])
        print(f"[{i+1}/{len(df_ads)}] Processing {ad_id}...", flush=True)

        ts = datetime.utcnow().isoformat()
        result = {
            "ad_id": ad_id,
            "timestamp_utc": ts,
            "error": None,
            "score": None,
            "flags_count": None,
            "summary": None,
            "rewrite_path": None,
            "raw_path": None,
            "json_path": None,
            "pdf_path": None,
                "html_path": None,
        }
        try:
            content = call_chat_completion(
                endpoint=args.endpoint,
                model=args.model,
                system_prompt=system_prompt,
                ad_text=ad_text,
                temperature=args.temperature,
                seed=args.seed,
                timeout=args.timeout,
            )
            # Save raw
            raw_path = outdir / f"{ad_id}_raw.txt"
            raw_path.write_text(content, encoding="utf-8")
            result["raw_path"] = str(raw_path)

            parsed = parse_model_output(content)
            # Save rewrite
            if parsed["rewrite"]:
                rewrite_path = outdir / f"{ad_id}_rewrite.txt"
                rewrite_path.write_text(parsed["rewrite"], encoding="utf-8")
                result["rewrite_path"] = str(rewrite_path)

            # Save JSON
            if parsed["parsed_json"]:
                json_path = outdir / f"{ad_id}_parsed.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(parsed["parsed_json"], f, ensure_ascii=False, indent=2)
                result["json_path"] = str(json_path)
                # Pull key fields if present
                pj = parsed["parsed_json"]
                result["score"] = pj.get("score")
                # flags might be list under various keys; try common ones
                flags = pj.get("flags") or pj.get("issues") or pj.get("detections")
                if isinstance(flags, list):
                    result["flags_count"] = len(flags)
                result["summary"] = pj.get("summary") or pj.get("explanation")

            # PDF report
            if save_pdf_report:
                try:
                    pdf_path = outdir / f"{ad_id}.pdf"
                    save_pdf_report(
                        pdf_path,
                        ad_id=ad_id,
                        original_ad=ad_text,
                        parsed_json=parsed["parsed_json"],
                        rewrite=parsed["rewrite"],
                        raw_text=content,
                        metadata={
                            "model": args.model,
                            "endpoint": args.endpoint,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "timestamp_utc": ts,
                        },
                    )
                    result["pdf_path"] = str(pdf_path)
                except Exception as e:
                    result["error"] = f"PDF generation failed: {e}"

        except Exception as e:
            result["error"] = str(e)

        rows.append(result)

    out_csv = outdir / "results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nDone. Summary written to: {out_csv}")
    print("Per-ad outputs saved under:", outdir.resolve())

if __name__ == "__main__":
    main()
