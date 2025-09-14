# app.py
# Age-Inclusive Job Ad Checker (Local • LM Studio)
# Deterministic analysis, creative rewrites
# Lexicon enrichment, heuristic backstops, and clean downloads

import os
import io
import json
import re
import zipfile

import requests
import pandas as pd
import streamlit as st

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_LM_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL_ID    = "meta-llama-3.1-8b-instruct"
DEFAULT_SEED        = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEXICON_PATH = os.path.join(BASE_DIR, "lexicon.csv")
PROMPT_PATH  = os.path.join(BASE_DIR, "system_prompt.txt")

# ---------------------------
# Helpers
# ---------------------------
def sort_by_len_desc(items):
    return sorted(items, key=lambda x: len(x or ""), reverse=True)

def safe_replace_with_flags(text: str, flags: list) -> str:
    """Deterministically replace flagged phrases with provided alternatives (case-insensitive)."""
    if not text or not isinstance(flags, list):
        return text
    new_text = text
    pairs = []
    for f in flags:
        if not isinstance(f, dict):
            continue
        phrase = (f.get("phrase") or "").strip()
        alt = (f.get("alternative") or "").strip()
        if phrase and alt:
            pairs.append((phrase, alt))
    for phrase, alt in sort_by_len_desc(pairs):
        try:
            pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
            new_text = pattern.sub(alt, new_text)
        except re.error:
            continue
    return new_text

def build_rewrite_prompt(original_text: str, flags: list, mode_label: str) -> str:
    rules = [
        "Preserve all factual content, numbers, job title(s), required skills, benefits, locations, company name, URLs and emails.",
        "Do not introduce new responsibilities, benefits, or claims.",
        "Replace age-coded phrases with inclusive alternatives informed by FLAGS where applicable.",
        "Keep structure and formatting (headings, bullet points) and stay roughly the same length (±10%).",
        "Avoid prohibited or age-coded wording such as 'young', 'energetic', 'digital native', 'recent graduate', 'work hard play hard', 'fast-paced only'.",
        "Use clear, professional British English.",
    ]
    mode_hint = (
        "Polish wording only; do not remove sections."
        if "Polish" in mode_label and "Replace" not in mode_label
        else "Apply replacements first, then smooth for readability."
    )
    prompt = (
        "You are an assistant specialising in age-inclusive editing.\n"
        f"Mode: {mode_label} — {mode_hint}\n\n"
        "Rewrite the JOB AD to be more age-inclusive while following ALL RULES strictly.\n\n"
        "RULES:\n- " + "\n- ".join(rules) + "\n\n"
        "FLAGS (JSON array of {phrase, why, alternative, severity}):\n"
        f"{json.dumps(flags, ensure_ascii=False)}\n\n"
        "JOB AD:\n\"\"\"\n" + original_text + "\n\"\"\"\n\n"
        "Return ONLY the rewritten advertisement text — no self-checks, no notes, no JSON, and no code fences."
    )
    return prompt

PROHIBITED_TERMS = [
    "young","younger","youthful","recent graduate","new grad","digital native",
    "high energy","energetic","work hard play hard","fast-paced","under 30","no more than"
]

def extract_numbers(text: str):
    if not text:
        return {}
    nums = re.findall(r"\b\d+(?:[\.,]\d+)?\b", text)
    from collections import Counter
    return Counter(nums)

def qa_rewrite(original: str, rewritten: str) -> dict:
    issues = []
    if original and rewritten:
        len_o = len(original)
        len_r = len(rewritten)
        if len_o > 0:
            drift = abs(len_r - len_o) / len_o
            if drift > 0.2:
                issues.append(f"Length changed by {int(drift*100)}%, which is more than 20%.")
    low = (rewritten or "").lower()
    for term in PROHIBITED_TERMS:
        if term in low:
            issues.append(f"Contains prohibited phrase: '{term}'.")
    co = extract_numbers(original or "")
    cr = extract_numbers(rewritten or "")
    missing = []
    for k, v in co.items():
        if cr.get(k, 0) < v:
            missing.append(k)
    if missing:
        issues.append(f"Some numbers from the original may be missing: {', '.join(sorted(set(missing))[:10])}")
    return {"ok": len(issues)==0, "issues": issues}

def load_lexicon(path: str = LEXICON_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df[["phrase", "why", "alternative", "severity"]]

def build_system_prompt(lexicon_df: pd.DataFrame, prompt_path: str = PROMPT_PATH) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        sys_tmpl = f.read()
    csv_lines = ["phrase,why,alternative,severity"]
    for _, row in lexicon_df.iterrows():
        csv_lines.append(f"{row['phrase']},{row['why']},{row['alternative']},{row['severity']}")
    return sys_tmpl.replace("<<<LEXICON_CSV>>>", "\n".join(csv_lines))

def call_llm(base_url: str, model_id: str, temperature: float, system_prompt: str, user_prompt: str, seed: int | None = None, timeout: int = 120) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_id,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def extract_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return json.loads(m.group(0)) if m else {}

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def highlight_phrases(text: str, phrases: list[str]) -> str:
    escaped = [re.escape(p) for p in phrases if p]
    if not escaped:
        return text
    pattern = re.compile(r"(" + "|".join(escaped) + r")", flags=re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)

def count_severity_from_df(df: pd.DataFrame) -> dict:
    counts = {1: 0, 2: 0, 3: 0}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "severity" not in df.columns:
        return counts
    for v in df["severity"].tolist():
        s = None
        try:
            s = int(v)
        except Exception:
            sv = str(v).strip().lower()
            if "3" in sv or "high" in sv:
                s = 3
            elif "2" in sv or "med" in sv:
                s = 2
            elif "1" in sv or "low" in sv:
                s = 1
        if s in counts:
            counts[s] += 1
    return counts

def count_severity(flags: list) -> dict:
    counts = {1:0,2:0,3:0}
    for f in flags or []:
        if not isinstance(f, dict):
            continue
        sev_raw = f.get("severity")
        sev = None
        try:
            sev = int(sev_raw)
        except Exception:
            if isinstance(sev_raw, str):
                s = sev_raw.lower()
                if "3" in s or "high" in s: sev = 3
                elif "2" in s or "med" in s: sev = 2
                elif "1" in s or "low" in s: sev = 1
        if sev in counts:
            counts[sev]+=1
    return counts

def severity_label(sev: int) -> str:
    return {3:"High", 2:"Medium", 1:"Low"}.get(sev, str(sev))

def clean_rewrite_output(text: str) -> str:
    """Strip self-checks/notes/JSON/fences and return plain rewrite text."""
    if not isinstance(text, str):
        return ""
    s = text.strip()

    if "\n---\n" in s:
        s = s.split("\n---\n")[-1].strip()

    s = re.sub(r"(?is)self[- ]?check:\s*.*", "", s)
    s = re.sub(r"```json\s*.*?```", "", s, flags=re.DOTALL|re.IGNORECASE)
    s = re.sub(r"^```.*?\n(.*?)\n```$", r"\1", s, flags=re.DOTALL)

    if s.lstrip().startswith("{") and "}" in s:
        last_brace = s.rfind("}")
        if last_brace != -1:
            s = s[last_brace+1:].strip()

    lines = [
        ln for ln in s.splitlines()
        if not ln.strip().lower().startswith(("note:", "explanation:", "self-check:"))
    ]
    s = "\n".join(lines).strip()
    return s

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def build_lexicon_maps(lexicon_df: pd.DataFrame):
    """Return dicts for quick lookup by phrase (include why_map for backfill)."""
    sev_map, alt_map, why_map = {}, {}, {}
    for _, r in lexicon_df.iterrows():
        key = _norm(str(r.get("phrase","")))
        if not key:
            continue
        try:
            sev_map[key] = int(r.get("severity"))
        except Exception:
            pass
        alt_map[key] = str(r.get("alternative",""))
        why_map[key] = str(r.get("why",""))
    return sev_map, alt_map, why_map

def _basic_norm(s: str) -> str:
    return (s or "").lower().strip()\
        .replace("’", "'")\
        .replace("‘", "'")\
        .replace("–", "-")\
        .replace("—", "-")

def heuristic_severity_and_alt(phrase: str):
    """Best-guess fallback if lexicon has no match."""
    p = _basic_norm(phrase)

    if re.search(r"\b(up to|maximum of|no more than)\s*\d+\s*years'?(\s+of)?\s*experience\b", p):
        return 2, "We welcome applicants with varied experience levels; we assess skills and impact over years."

    if re.search(r"\b(recent graduate|new grad)\b", p):
        return 3, "We welcome candidates at all career stages, including those with transferable skills."

    if re.search(r"\bunder\s*\d+\b", p) or re.search(r"\bno more than\s*\d+\s*(years old|yrs old)?\b", p):
        return 3, "We do not impose age limits; suitability is based on skills and competencies."

    if re.search(r"\b(youthful|energetic|high energy)\b", p):
        return 2, "We value commitment and collaboration over age-related descriptors."

    if re.search(r"\b(work hard play hard|fast-paced)\b", p):
        return 1, "We encourage a balanced, supportive work culture."

    return None, None

def detect_heuristic_flags(text: str) -> list:
    """Quick regex checks for age-coded patterns not caught by the LLM."""
    flags = []
    if not text:
        return flags

    pat_exp_cap = re.compile(
        r"(?i)\b(?:with\s+)?(?:up to|maximum of|no more than)\s*\d+\s*years(?:’|')?\s*(?:of\s+)?experience\b"
    )
    for m in pat_exp_cap.finditer(text):
        found = m.group(0)
        flags.append({
            "phrase": found,
            "why": "Caps required experience by years, which can deter later-career applicants.",
            "alternative": "We welcome applicants with varied experience levels; we assess skills and impact over years.",
            "severity": 2,
            "source": "Best guess",
        })

    return flags

def _norm_phrase_for_dedupe(s: str) -> str:
    return (s or "").lower().strip().replace("’", "'").replace("“", '"').replace("”", '"')

def merge_flags(existing: list, extra: list) -> list:
    """Merge model + heuristic flags by phrase key (avoid duplicates)."""
    if not isinstance(existing, list):
        existing = []
    if not isinstance(extra, list):
        extra = []
    seen = {_norm_phrase_for_dedupe(f.get("phrase","")) for f in existing if isinstance(f, dict)}
    merged = list(existing)
    for f in extra:
        if not isinstance(f, dict):
            continue
        key = _norm_phrase_for_dedupe(f.get("phrase",""))
        if key and key not in seen:
            merged.append(f)
            seen.add(key)
    return merged

def _norm_phrase_for_match(s: str) -> str:
    return (s or "").lower().strip().replace("’","'").replace("“",'"').replace("”",'"')

def build_learn_candidates(all_flags_rows: list[dict], min_freq: int = 1, cap: int = 100) -> pd.DataFrame:
    """
    From batch all_flags rows (dicts with ad_id, phrase, reason/why, alternative, severity, source),
    build a deduped candidate table for dry-run learning.
    """
    if not all_flags_rows:
        return pd.DataFrame(columns=["phrase","count","first_seen_ad","reason","alternative","severity"])
    from collections import defaultdict, Counter

    buckets = defaultdict(list)
    order = []  # to preserve first-seen order per phrase
    for r in all_flags_rows:
        if not isinstance(r, dict): 
            continue
        phrase = (r.get("phrase") or "").strip()
        if not phrase:
            continue
        if r.get("source") == "Lexicon":  # skip things the lexicon already knows
            continue
        key = _norm_phrase_for_match(phrase)
        if key not in buckets:
            order.append(key)
        buckets[key].append(r)

    rows = []
    for key in order:
        bunch = buckets[key]
        # frequency & first seen
        count = len(bunch)
        first_seen_ad = next((x.get("ad_id") for x in bunch if x.get("ad_id")), "")
        # pick best fields: prefer non-empty, most frequent
        def pick(field_names):
            vals = []
            for x in bunch:
                for f in field_names:
                    v = (x.get(f) or "").strip() if isinstance(x.get(f), str) else x.get(f)
                    if v:
                        vals.append(v)
                        break
            if not vals:
                return ""
            # most common value
            return Counter(vals).most_common(1)[0][0]

        reason = pick(["reason","why"]) or "May deter candidates at different career stages."
        alternative = pick(["alternative"]) or "We welcome applicants with varied experience; we assess skills and impact."
        severity = pick(["severity"])
        try:
            severity = max(1, min(3, int(severity)))
        except Exception:
            severity = 1

        phrase_original = bunch[0].get("phrase") or ""
        rows.append({
            "phrase": phrase_original,
            "count": count,
            "first_seen_ad": first_seen_ad,
            "reason": reason,
            "alternative": alternative,
            "severity": severity,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["count","phrase"], ascending=[False, True])
    df = df[df["count"] >= int(min_freq)]
    if cap and len(df) > int(cap):
        df = df.head(int(cap))
    return df.reset_index(drop=True)

def learn_from_flags(flags: list[dict], lexicon_df: pd.DataFrame) -> int:
    """Add all flags not already in the lexicon (by normalized phrase). Returns how many added/updated."""
    if not isinstance(flags, list) or not flags:
        return 0
    # Build set of existing keys
    existing = set(lexicon_df["phrase"].astype(str).map(_norm_phrase_for_match)) if isinstance(lexicon_df, pd.DataFrame) and "phrase" in lexicon_df.columns else set()
    changes = 0
    for f in flags:
        if not isinstance(f, dict):
            continue
        phrase = (f.get("phrase") or "").strip()
        if not phrase:
            continue
        key = _norm_phrase_for_match(phrase)
        # Only learn from non-lexicon sources
        if f.get("source") == "Lexicon":
            continue

        # Pull reason/alt; if blank, fallback to heuristic advice
        reason = (f.get("reason") or f.get("why") or "").strip()
        alt = (f.get("alternative") or "").strip()
        sev = f.get("severity") or 1
        try:
            sev = max(1, min(3, int(sev)))
        except Exception:
            sev = 1

        # Lightweight defaulting (keep it generic and safe)
        if not reason:
            reason = "May deter candidates at different career stages."
        if not alt:
            alt = "We welcome applicants with varied experience; we assess skills and impact."

        # Upsert (safe; uses your backup+atomic function you already have)
        ok = upsert_lexicon_row(LEXICON_PATH, phrase=phrase, why=reason, alternative=alt, severity=sev)
        if ok:
            changes += 1
    return changes


def enrich_flags_with_lexicon(flags: list, lexicon_df: pd.DataFrame) -> list:
    """Fill/override severity, alternative, and reason; tag source as AI/Lexicon/Best guess."""
    if not isinstance(flags, list) or not len(flags):
        return flags
    sev_map, alt_map, why_map = build_lexicon_maps(lexicon_df)
    enriched = []
    for f in flags:
        f2 = dict(f) if isinstance(f, dict) else {}
        f2.setdefault("source", "AI")
        phrase_key = _norm(str(f2.get("phrase","")))

        # Lexicon severity
        sev_val = f2.get("severity")
        sev_int = None
        try:
            sev_int = int(sev_val)
        except Exception:
            sev_int = sev_map.get(phrase_key)
        if sev_int is not None:
            f2["severity"] = int(sev_int)
            f2["source"] = "Lexicon"

        # Alternative backfill
        alt_val = f2.get("alternative")
        if not alt_val:
            alt_from_lex = alt_map.get(phrase_key)
            if alt_from_lex:
                f2["alternative"] = alt_from_lex
                f2["source"] = "Lexicon"

        # Reason/Why backfill & normalize
        reason_val = f2.get("reason")
        why_val = f2.get("why")
        if isinstance(reason_val, str) and reason_val.strip() == "":
            reason_val = None
        if isinstance(why_val, str) and why_val.strip() == "":
            why_val = None
        if not (reason_val or why_val):
            why_from_lex = why_map.get(phrase_key)
            if why_from_lex:
                f2["reason"] = why_from_lex
                f2["source"] = "Lexicon"
        else:
            if not reason_val and why_val:
                f2["reason"] = why_val

        # Best guess fallback
        if not f2.get("severity") or f2.get("severity") == 0:
            h_sev, h_alt = heuristic_severity_and_alt(f2.get("phrase", ""))
            if h_sev:
                f2["severity"] = h_sev
                if not f2.get("alternative"):
                    f2["alternative"] = h_alt
                f2["source"] = "Best guess"

        enriched.append(f2)
    return enriched

def upsert_lexicon_row(path: str, phrase: str, why: str, alternative: str, severity: int):
    df = load_lexicon(path)
    key = (phrase or "").strip().lower().replace("’", "'").replace("“","\"").replace("”","\"")
    df["_key"] = df["phrase"].astype(str).str.lower().str.replace("’","'", regex=False)
    if key in df["_key"].values:
        idx = df.index[df["_key"] == key][0]
        df.loc[idx, ["why", "alternative", "severity"]] = [why, alternative, severity]
    else:
        df = pd.concat([df.drop(columns=["_key"], errors="ignore"),
                        pd.DataFrame([{
                            "phrase": phrase,
                            "why": why,
                            "alternative": alternative,
                            "severity": int(severity)
                        }])],
                       ignore_index=True)
    df = df.drop(columns=["_key"], errors="ignore")
    df.to_csv(path, index=False)
    return True

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Age Bias Checker (Local)", page_icon="🛡️", layout="wide")
st.title("🛡️ Age-Inclusive Job Ad Checker (Local)")

with st.sidebar:
    st.subheader("Settings")
    lm_base_url = st.text_input("Endpoint (LM Studio)", DEFAULT_LM_BASE_URL)
    model_id    = st.text_input("Model ID", DEFAULT_MODEL_ID)
    seed        = st.number_input("Seed (optional)", value=DEFAULT_SEED, step=1)
    rewrite_temp = st.slider("Rewrite creativity", 0.0, 1.0, 0.4, 0.05)
    skip_rewrite = st.checkbox("Skip rewrite (faster analysis-only)", value=False)
    rewrite_mode = st.selectbox("Rewrite mode", ["Polish only", "Replace + Polish (use lexicon alternatives)"])
    auto_learn = st.checkbox("Auto-learn new phrases", value=False, help="Add all non-lexicon flags to the lexicon automatically after analysis.")

# Load resources
lexicon_df = load_lexicon(LEXICON_PATH)
system_prompt = build_system_prompt(lexicon_df, PROMPT_PATH)

tab_single, tab_batch = st.tabs(["Single Ad", "Batch (CSV)"])

# ---------------------------
# Single Ad tab
# ---------------------------
with tab_single:
    st.subheader("Paste a job ad")
    job_ad = st.text_area("Job description", height=260)

    col1, col2 = st.columns(2)
    with col1:
        run = st.button("Analyse")
    with col2:
        rerun_rewrite = st.button("Re-run rewrite only")

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "rewritten_text" not in st.session_state:
        st.session_state.rewritten_text = None

    if run:
        if not job_ad.strip():
            st.warning("Please paste a job ad first.")
            st.stop()
        with st.spinner("Analysing job ad for age-coded language…"):
            analysis_prompt = (
                "Analyse the following job ad for age-coded language. "
                "Respond ONLY with a JSON object containing: score, flags[], candidates[], summary.\n\n"
                f"JOB AD:\n\"\"\"\n{job_ad}\n\"\"\""
            )
            try:
                raw_analysis = call_llm(lm_base_url, model_id, 0.0, system_prompt, analysis_prompt, seed=seed)
                parsed = extract_json(raw_analysis)

                # Start with model flags (if any)
                model_flags = parsed.get("flags", []) if isinstance(parsed, dict) else []

                # Add heuristic flags directly from the ad text
                heur_flags = detect_heuristic_flags(job_ad)

                # Merge & enrich via lexicon
                merged_flags = merge_flags(model_flags, heur_flags)
                parsed["flags"] = enrich_flags_with_lexicon(merged_flags, lexicon_df)

                st.session_state.analysis_result = {"job_ad": job_ad, "parsed": parsed}
                st.success("Analysis complete ✅")
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.stop()

        data = st.session_state.analysis_result["parsed"]
        st.metric("Age-Inclusivity Score", f"{data.get('score','—')}/100")
        flags = data.get("flags", [])
        if flags:
            flags_df = pd.DataFrame(flags)

            # Normalize WHY vs REASON for display (prefer single 'reason' column)
            if "why" in flags_df.columns and "reason" in flags_df.columns:
                flags_df["reason"] = flags_df["reason"].astype(str)
                flags_df.loc[flags_df["reason"].str.strip() == "", "reason"] = pd.NA
                flags_df["reason"] = flags_df["reason"].fillna(flags_df["why"])
                flags_df = flags_df.drop(columns=["why"])
            elif "why" in flags_df.columns and "reason" not in flags_df.columns:
                flags_df = flags_df.rename(columns={"why": "reason"})

            # Order: phrase → severity → reason → alternative → source
            cols = list(flags_df.columns)
            desired = ["phrase", "severity", "reason", "alternative", "source"]
            new_order = [c for c in desired if c in cols] + [c for c in cols if c not in desired]
            flags_df = flags_df[new_order]

            st.dataframe(flags_df, use_container_width=True)
            st.download_button(
                "Download Flags (CSV)",
                data=df_to_csv_bytes(flags_df),
                file_name="flags.csv",
                mime="text/csv"
            )
            sev_counts = count_severity_from_df(flags_df)
            st.caption(f"Flags found → High: {sev_counts[3]} | Medium: {sev_counts[2]} | Low: {sev_counts[1]}")
            # --- Simple learning controls
            colL, colR = st.columns([1,1])
            with colL:
                if st.button("Apply learning from this ad"):
                    try:
                        added = learn_from_flags(flags, lexicon_df)
                        if added:
                            st.success(f"Saved {added} phrase(s) to the lexicon.")
                            # reload lexicon so future calls use the new entries immediately
                            lexicon_df = load_lexicon(LEXICON_PATH)
                        else:
                            st.info("No new phrases to add (or they were already in the lexicon).")
                    except Exception as e:
                        st.error(f"Learning failed: {e}")

            # Auto-learn if enabled
            if auto_learn:
                try:
                    added = learn_from_flags(flags, lexicon_df)
                    if added:
                        st.caption(f"Auto-learn added {added} phrase(s) to the lexicon.")
                        lexicon_df = load_lexicon(LEXICON_PATH)
                except Exception as e:
                    st.warning(f"Auto-learn skipped (error: {e})")

        else:
            st.success("No potentially age-coded phrases detected.")
        st.write("Summary:", data.get("summary",""))

        phrases = [f.get("phrase","") for f in (flags or [])]
        highlighted = highlight_phrases(job_ad, phrases)
        st.markdown(
            f"<div style='border:1px solid #ddd;border-radius:8px;padding:.75rem;background:#fff'>{highlighted}</div>",
            unsafe_allow_html=True
        )

        if not skip_rewrite:
            with st.spinner("Creating a more inclusive version of the ad…"):
                rewrite_prompt = (
                    "Rewrite the following job ad to be more age-inclusive, "
                    "using the provided flags and alternatives where possible. "
                    "Preserve meaning and structure.\n\n"
                    f"JOB AD:\n\"\"\"\n{job_ad}\n\"\"\"\n\n"
                    f"FLAGS JSON:\n{json.dumps(flags)}\n\n"
                    "Return ONLY the rewritten advertisement text — no self-checks, no notes, no JSON, and no code fences."
                )
                try:
                    rewritten = call_llm(lm_base_url, model_id, float(rewrite_temp), system_prompt, rewrite_prompt, seed=seed)
                    cleaned = clean_rewrite_output(rewritten)

                    # Ensure flagged phrases are swapped even if the model missed them
                    flags_for_replace = parsed.get("flags", [])
                    final_text = safe_replace_with_flags(cleaned, flags_for_replace)

                    st.session_state.rewritten_text = final_text
                    qa = qa_rewrite(job_ad, final_text)

                    if qa["ok"]:
                        st.success("Rewrite complete ✨")
                    else:
                        st.warning("Rewrite complete with issues:")
                        for issue in qa["issues"]:
                            st.write("- " + issue)
                except Exception as e:
                    st.error(f"Rewrite error: {e}")

    if rerun_rewrite and st.session_state.analysis_result:
        job_ad = st.session_state.analysis_result["job_ad"]
        flags = st.session_state.analysis_result["parsed"].get("flags", [])
        with st.spinner("Refreshing rewrite with your new creativity setting…"):
            rewrite_prompt = (
                "Rewrite the following job ad to be more age-inclusive, "
                "using the provided flags and alternatives where possible.\n\n"
                f"JOB AD:\n\"\"\"\n{job_ad}\n\"\"\"\n\n"
                f"FLAGS JSON:\n{json.dumps(flags)}\n\n"
                "Return ONLY the rewritten advertisement text — no self-checks, no notes, no JSON, and no code fences."
            )
            try:
                rewritten = call_llm(lm_base_url, model_id, float(rewrite_temp), system_prompt, rewrite_prompt, seed=seed)
                cleaned = clean_rewrite_output(rewritten)
                flags_for_replace = flags  # enriched flags from session
                final_text = safe_replace_with_flags(cleaned, flags_for_replace)
                st.session_state.rewritten_text = final_text
                qa = qa_rewrite(job_ad, final_text)

                if qa["ok"]:
                    st.success("Rewrite complete ✨")
                else:
                    st.warning("Rewrite complete with issues:")
                    for issue in qa["issues"]:
                        st.write("- " + issue)
            except Exception as e:
                st.error(f"Rewrite error: {e}")

    # --- SINGLE, CLEAN REWRITTEN-AD BLOCK (kept inside the guard)
    if st.session_state.rewritten_text and not skip_rewrite:
        st.subheader("Rewritten Ad")
        st.text_area("Inclusive version", value=st.session_state.rewritten_text, height=320)

        text = st.session_state.rewritten_text  # reuse below

        # TXT download (no extra packages)
        st.download_button(
            "Download as TXT",
            data=text.encode("utf-8"),
            file_name="rewritten_ad.txt",
            mime="text/plain"
        )

        # RTF download (opens in Word; Word can 'Save as PDF') - no extra packages
        def to_rtf(s: str) -> str:
            esc = s.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
            esc = esc.replace("\r\n", "\n").replace("\r", "\n").replace("\n", r"\par " + "\n")
            return r"{\rtf1\ansi\deff0 " + esc + "}"
        rtf_bytes = to_rtf(text).encode("utf-8")
        st.download_button(
            "Download as RTF (open in Word)",
            data=rtf_bytes,
            file_name="rewritten_ad.rtf",
            mime="application/rtf"
        )

        # HTML download (open in browser; Print -> Save as PDF) - no extra packages
        def html_escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_doc = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Rewritten Ad</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 2rem; }}
  .content {{ white-space: pre-wrap; line-height: 1.5; }}
</style>
<h1>Rewritten Ad</h1>
<div class="content">{html_escape(text)}</div>
</html>"""
        st.download_button(
            "Download as HTML (print to PDF)",
            data=html_doc.encode("utf-8"),
            file_name="rewritten_ad.html",
            mime="text/html"
        )
# ---------------------------
# Batch tab
# ---------------------------
with tab_batch:
    st.subheader("Batch-process a CSV of adverts")
    up = st.file_uploader("Upload CSV", type=["csv"])
    colb1, colb2, colb3 = st.columns(3)
    with colb1:
        ad_col = st.text_input("Advert text column", value="ad_text")
    with colb2:
        id_col = st.text_input("Optional ID column (blank = auto)", value="")
    with colb3:
        limit = st.number_input("Max rows to process (safety cap)", min_value=1, value=10, step=1)
    run_batch = st.button("Run batch")
    # Dry-run learning (Batch)
    dry_run_learning = st.checkbox(
        "Dry run: build lexicon candidates (batch)",
        value=True,
        help="Collect phrases flagged by AI/Best guess and show a pick-list instead of writing immediately."
    )
    min_freq = st.number_input(
        "Min frequency to include as candidate",
        min_value=1, value=2, step=1,
        help="Only include phrases that appear in at least this many ads."
    )
    max_candidates = st.number_input(
        "Candidate cap (safety)",
        min_value=10, value=100, step=10,
        help="Limit how many candidate rows are shown from this batch."
    )
    auto_learn_batch = st.checkbox(
        "Auto-learn (batch) using min frequency filter",
        value=False,
        help="After the run, automatically add non-lexicon phrases that meet the min frequency threshold."
    )
    auto_min_freq = st.number_input(
        "Min frequency for batch auto-learn",
        min_value=1, value=2, step=1,
        help="Only auto-add phrases that appear in at least this many ads in the batch."
    )


    if run_batch:
        if not up:
            st.warning("Please upload a CSV first.")
            st.stop()
        df = pd.read_csv(up)
        if ad_col not in df.columns:
            st.error(f"Column '{ad_col}' not found in your CSV.")
            st.stop()
        if id_col and id_col not in df.columns:
            st.error(f"ID column '{id_col}' not found in your CSV.")
            st.stop()
        work = df.head(int(limit)).copy()
        if id_col:
            work["ad_id"] = work[id_col].astype(str)
        else:
            work["ad_id"] = [f"ad_{i+1:04d}" for i in range(len(work))]

        results = []
        all_flags = []
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            prog = st.progress(0, text="Analysing job ads…")
            for i, row in work.iterrows():
                ad_id = row["ad_id"]
                ad_text = str(row[ad_col])
                try:
                    analysis_prompt = (
                        "Analyse the following job ad for age-coded language. "
                        "Respond ONLY with a JSON object containing: score, flags[], candidates[], summary.\n\n"
                        f"JOB AD:\n\"\"\"\n{ad_text}\n\"\"\""
                    )
                    raw_analysis = call_llm(lm_base_url, model_id, 0.0, system_prompt, analysis_prompt, seed=seed, timeout=300)
                    parsed = extract_json(raw_analysis)
                    model_flags = parsed.get("flags", []) if isinstance(parsed, dict) else []
                    heur_flags = detect_heuristic_flags(ad_text)
                    merged_flags = merge_flags(model_flags, heur_flags)
                    parsed["flags"] = enrich_flags_with_lexicon(merged_flags, lexicon_df)

                    score = parsed.get("score") if isinstance(parsed, dict) else None
                    flags = parsed.get("flags") if isinstance(parsed, dict) else None
                    rewritten = None

                    if not skip_rewrite:
                        rewrite_prompt = (
                            "Rewrite the following job ad to be more age-inclusive, "
                            "using the provided flags and alternatives where possible.\n\n"
                            f"JOB AD:\n\"\"\"\n{ad_text}\n\"\"\"\n\n"
                            f"FLAGS JSON:\n{json.dumps(flags)}\n\n"
                            "Return ONLY the rewritten advertisement text — no self-checks, no notes, no JSON, and no code fences."
                        )
                        rewritten = call_llm(lm_base_url, model_id, float(rewrite_temp), system_prompt, rewrite_prompt, seed=seed, timeout=300)

                    sev_counts = count_severity(flags)
                    results.append({
                        "ad_id": ad_id,
                        "score": score,
                        "flags_count": len(flags) if isinstance(flags, list) else None,
                        "sev3_count": sev_counts[3],
                        "sev2_count": sev_counts[2],
                        "sev1_count": sev_counts[1],
                        "summary": parsed.get("summary") if isinstance(parsed, dict) else None,
                        "error": None,
                    })

                    if isinstance(flags, list):
                        for f in flags:
                            sev_raw = f.get("severity")
                            try:
                                sev_int = int(sev_raw)
                            except Exception:
                                sev_int = 0
                            all_flags.append({"ad_id": ad_id, "phrase": f.get("phrase"), "reason": f.get("reason") or f.get("why"), "alternative": f.get("alternative"), "severity": sev_int, "source": f.get("source"), "severity_label": severity_label(sev_int)})

                    z.writestr(f"{ad_id}_analysis.json", json.dumps(parsed, ensure_ascii=False, indent=2))
                    if rewritten:
                        cleaned = clean_rewrite_output(rewritten)
                        cleaned = safe_replace_with_flags(cleaned, parsed.get("flags", []))
                        z.writestr(f"{ad_id}_rewrite.txt", cleaned)

                except Exception as e:
                    results.append({
                        "ad_id": ad_id,
                        "score": None,
                        "flags_count": None,
                        "sev3_count": 0,
                        "sev2_count": 0,
                        "sev1_count": 0,
                        "summary": None,
                        "error": str(e),
                    })
                prog.progress(min(100, int((len(results)/len(work))*100)))
            prog.empty()
        st.success("Batch analysis complete ✅")

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
        st.download_button("Download results.csv", data=df_to_csv_bytes(res_df), file_name="results.csv", mime="text/csv")
        if all_flags:
            flags_df = pd.DataFrame(all_flags)
            st.download_button("Download all_flags.csv", data=df_to_csv_bytes(flags_df), file_name="all_flags.csv", mime="text/csv")
        st.download_button("Download per-ad ZIP", data=zbuf.getvalue(), file_name="ad_reports.zip", mime="application/zip")
        # --- Auto-learn (batch) using min frequency filter ---
        if "auto_learn_batch" in locals() and auto_learn_batch:
            try:
                if all_flags:
                    # normalise reason
                    for r in all_flags:
                        if not r.get("reason") and r.get("why"):
                            r["reason"] = r["why"]
                    # build candidates with min frequency from sidebar
                    cand_df_auto = build_learn_candidates(all_flags, min_freq=int(auto_min_freq), cap=100000)
                    added_auto = 0
                    for _, row in cand_df_auto.iterrows():
                        ok = upsert_lexicon_row(
                            LEXICON_PATH,
                            phrase=str(row.get("phrase") or ""),
                            why=str(row.get("reason") or ""),
                            alternative=str(row.get("alternative") or ""),
                            severity=int(row.get("severity") or 1),
                        )
                        if ok:
                            added_auto += 1
                    if added_auto:
                        st.success(f"Auto-learned {added_auto} phrase(s) into the lexicon (min frequency ≥ {int(auto_min_freq)}).")
                        try:
                            lexicon_df = load_lexicon(LEXICON_PATH)
                        except Exception:
                            pass
                    else:
                        st.info("Auto-learn found no candidates meeting the frequency threshold.")
            except Exception as e:
                st.warning(f"Auto-learn skipped (error: {e})")

        # Build candidates from all_flags (batch dry-run)
        if dry_run_learning:
            try:
                # Use the same all_flags list you already collect
                if all_flags:
                    # Normalise 'reason' in case some rows only have 'why'
                    for r in all_flags:
                        if not r.get("reason") and r.get("why"):
                            r["reason"] = r["why"]

                    cand_df = build_learn_candidates(all_flags, min_freq=int(min_freq), cap=int(max_candidates))
                    st.subheader("Lexicon candidates (dry run)")
                    if cand_df.empty:
                        st.info("No candidates met the criteria.")
                    else:
                        st.dataframe(cand_df, use_container_width=True)

                        # Download candidates CSV
                        st.download_button(
                            "Download learn_candidates.csv",
                            data=df_to_csv_bytes(cand_df),
                            file_name="learn_candidates.csv",
                            mime="text/csv"
                        )

                        # Pick & add selected
                        phrases = cand_df["phrase"].tolist()
                        selected = st.multiselect("Select phrases to add/update in lexicon", phrases, default=[])
                        if st.button("Add selected to lexicon"):
                            added = 0
                            for ph in selected:
                                row = cand_df[cand_df["phrase"] == ph].iloc[0]
                                ok = upsert_lexicon_row(
                                    LEXICON_PATH,
                                    phrase=ph,
                                    why=str(row.get("reason") or ""),
                                    alternative=str(row.get("alternative") or ""),
                                    severity=int(row.get("severity") or 1),
                                )
                                if ok:
                                    added += 1
                            if added:
                                st.success(f"Added/updated {added} phrase(s) in lexicon.csv")
                                # refresh in-memory lexicon so future runs benefit
                                try:
                                    lexicon_df = load_lexicon(LEXICON_PATH)
                                except Exception:
                                    pass
                            else:
                                st.info("No phrases were added.")
            except Exception as e:
                st.warning(f"Could not build candidates: {e}")
