
"""
html_report.py
--------------
Zero-dependency HTML report generator for per-ad bias checker outputs.
Produces a self-contained HTML file with light CSS.
"""

import html
from datetime import datetime

def _esc(s):
    return html.escape(s if s is not None else "")

def save_html_report(path, ad_id, original_ad, parsed_json, rewrite, raw_text, metadata=None):
    meta_rows = ""
    if metadata:
        for k, v in metadata.items():
            meta_rows += f"<tr><th>{_esc(str(k))}</th><td>{_esc(str(v))}</td></tr>"

    # Flags table
    flags_html = "<p>No flags reported.</p>"
    flags = []
    if parsed_json:
        flags = parsed_json.get("flags") or parsed_json.get("issues") or parsed_json.get("detections") or []
    if flags:
        rows = "".join(
            f"<tr><td>{_esc(f.get('phrase',''))}</td>"
            f"<td>{_esc(f.get('why', f.get('reason','')))}</td>"
            f"<td>{_esc(f.get('alternative', f.get('suggestion','')))}</td>"
            f"<td>{_esc(f.get('severity', f.get('level','')))}</td></tr>"
            for f in flags
        )
        flags_html = f"""
        <table class="grid">
          <thead><tr><th>Phrase</th><th>Why</th><th>Alternative</th><th>Severity</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    score = "—"
    summary = "—"
    if parsed_json:
        score = parsed_json.get("score", "—")
        summary = parsed_json.get("summary") or parsed_json.get("explanation") or "—"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Bias Checker Report — { _esc(ad_id) }</title>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #111111;
      --muted: #666666;
      --grid: #dddddd;
      --accent: #1f6feb;
    }}
    body {{
      background: var(--bg);
      color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji", "Segoe UI Symbol", sans-serif;
      margin: 2rem;
      line-height: 1.5;
    }}
    h1, h2, h3 {{ margin: 0 0 .5rem 0; }}
    h1 {{ font-size: 1.6rem; }}
    h2 {{ font-size: 1.25rem; margin-top: 1.5rem; }}
    h3 {{ font-size: 1.1rem; margin-top: 1.25rem; }}
    .muted {{ color: var(--muted); }}
    .card {{
      border: 1px solid var(--grid);
      border-radius: 12px;
      padding: 1rem 1.2rem;
      margin-bottom: 1rem;
    }}
    table.grid {{
      width: 100%;
      border-collapse: collapse;
      margin: .5rem 0;
      font-size: .95rem;
    }}
    table.grid th, table.grid td {{
      border: 1px solid var(--grid);
      padding: .5rem .6rem;
      vertical-align: top;
    }}
    table.grid thead th {{
      background: #f3f4f6;
    }}
    pre, .prelike {{
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
      font-size: .9rem;
      background: #fafafa;
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: .75rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Bias Checker Report</h1>
    <div class="muted">Ad ID: <strong>{_esc(ad_id)}</strong></div>
  </header>

  <section class="card">
    <h2>Run Metadata</h2>
    <table class="grid">
      <tbody>
        {meta_rows}
      </tbody>
    </table>
  </section>

  <section class="card">
    <h2>Summary / Score</h2>
    <div><strong>Score:</strong> { _esc(str(score)) }</div>
    <div style="margin-top:.25rem;"><strong>Summary:</strong> <span class="prelike">{ _esc(summary) }</span></div>
  </section>

  <section class="card">
    <h2>Flags</h2>
    {flags_html}
  </section>

  <section class="card">
    <h2>Rewritten Advert</h2>
    <div class="prelike">{ _esc(rewrite or "—") }</div>
  </section>

  <section class="card">
    <h2>Original Advert</h2>
    <div class="prelike">{ _esc(original_ad or "—") }</div>
  </section>

  <section class="card">
    <h2>Raw Model Output</h2>
    <div class="prelike">{ _esc(raw_text or "—") }</div>
  </section>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")
