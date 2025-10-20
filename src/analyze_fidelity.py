#!/usr/bin/env python3
"""
RQ-9 fidelity analysis (expert version): Identifier preservation + expected-script transliteration

Key ideas
---------
1) Identifier source (choose with --id_source):
   - "code":          extract identifiers from CODE only (strongest tie to ground truth, but harsher).
   - "ref":           extract identifiers from REFERENCE SUMMARY only (what humans mention).
   - "code_critical": intersection of code IDs with tokens present in the REFERENCE SUMMARY
                      (recommended default; measures whether model keeps *mention-worthy* code names).

2) Transliteration:
   - For each generated summary, detect tokens whose dominant Unicode script is NOT expected for bt_language.
   - Skip tokens that are known identifiers (from the chosen source) or that look like code (handles backticks).
   - Flag a row only if >=2 suspect tokens OR >=5% suspect tokens.

3) Optional: relative-to-English normalization (--rel_to_english):
   - If English generations exist (bt_language == "english"), compute per (sample_id, model_name) an English baseline
     for identifier preservation, and report ratios target/English to control for model stylistic omission of names.

Usage
-----
  python analyze_fidelity_expert.py --csv path/to/df.csv --outdir results_rq9
  # knobs
  --id_source code_critical --id_preserve_thresh 0.7 --id_min_len 3 --id_max_per_row 0
  --translit_row_min_hits 2 --translit_row_min_ratio 0.05
  --rel_to_english --rel_low_threshold 0.8

Expected columns
----------------
- required: model_name, bt_language, reference_summary, generated_summary
- recommended for code-based identifiers: code
- optional for English baseline: sample_id with English rows in bt_language

Outputs
-------
aggregates/
  by_model_language.csv, by_model.csv, by_language.csv
figures/
  heatmap_identifier_preservation.png
  heatmap_identifier_preservation_rel_en.png   (if --rel_to_english and English exists)
  heatmap_transliteration_rate.png
  stacked_bar_flags_by_model.png
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="RQ-9: expert identifier preservation + transliteration")
    p.add_argument("--csv", required=True, help="Path to CSV with required columns")
    p.add_argument("--outdir", default="results_rq9", help="Output directory")
    p.add_argument("--id_source", choices=["code", "ref", "code_critical"], default="code_critical",
                   help="Source of identifiers for preservation/whitelist")
    p.add_argument("--id_min_len", type=int, default=3, help="Min identifier length (noise filter)")
    p.add_argument("--id_max_per_row", type=int, default=0, help="Cap source identifiers per row (0 = no cap)")
    p.add_argument("--id_preserve_thresh", type=float, default=0.70,
                   help="Below this preservation fraction, a row is flagged (identifier_low_flag)")
    p.add_argument("--translit_row_min_hits", type=int, default=2,
                   help="Min suspect tokens to flag a row as transliteration issue")
    p.add_argument("--translit_row_min_ratio", type=float, default=0.05,
                   help="Min fraction of suspect tokens to flag a row")
    p.add_argument("--rel_to_english", action="store_true",
                   help="Also compute metrics relative to English baseline if English rows exist")
    p.add_argument("--rel_low_threshold", type=float, default=0.8,
                   help="Threshold for relative identifier loss (< this vs English)")
    return p.parse_args()


# ---------------- Identifier extraction ----------------

# Code identifiers (allow PHP-style $var)
CODE_IDENT_RE = re.compile(r"\$?[A-Za-z_][A-Za-z0-9_]*")

# General identifier-looking (used for reference text as well)
TEXT_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Very common throwaway names to ignore (extend as needed)
COMMON_STOPWORDS = {
    "i","j","k","n","m","x","y","z",
    "tmp","temp","val","vals","value","values","data","info","obj","res","ret","out","in","arg","args","kwargs",
    "buf","ptr","ref","num","len","cnt","idx","id","key","keys","item","items","list","arr","vec","map","dict",
    "str","char","byte","bool","int","float","double","var","let","const","function","func","method","class",
    "self","this","super","null","none","true","false"
}

def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for s in seq:
        sl = s.lower()
        if sl in seen:
            continue
        seen.add(sl)
        out.append(s)
    return out

def tokenize_identifiers_from_code(code: str, min_len: int = 3, max_per_row: int = 0):
    """Extract identifier-like tokens from code (ASCII letters/underscore, optional leading $)."""
    if not isinstance(code, str) or not code.strip():
        return []
    toks = CODE_IDENT_RE.findall(code)
    # Normalize and filter
    norm = []
    for t in toks:
        t = t.lstrip("$")  # drop PHP leading $
        if len(t) < min_len:
            continue
        if t.lower() in COMMON_STOPWORDS:
            continue
        # keep only tokens with at least one letter (avoid pure numbers)
        if not re.search(r"[A-Za-z]", t):
            continue
        norm.append(t)
    out = dedupe_preserve_order(norm)
    if max_per_row and len(out) > max_per_row:
        out = out[:max_per_row]
    return out

def tokenize_identifiers_from_ref(text: str, min_len: int = 3, max_per_row: int = 0):
    """Extract identifier-looking tokens from reference summary text."""
    if not isinstance(text, str) or not text.strip():
        return []
    toks = TEXT_IDENT_RE.findall(text)
    norm = []
    for t in toks:
        if len(t) < min_len:
            continue
        if t.lower() in COMMON_STOPWORDS:
            continue
        norm.append(t)
    out = dedupe_preserve_order(norm)
    if max_per_row and len(out) > max_per_row:
        out = out[:max_per_row]
    return out

def build_source_identifiers(row, source: str, id_min_len: int, id_max_per_row: int):
    code_ids = tokenize_identifiers_from_code(row.get("code","") or "", id_min_len, id_max_per_row)
    if source == "code":
        return code_ids
    ref_ids = tokenize_identifiers_from_ref(row.get("reference_summary","") or "", id_min_len, 0)
    if source == "ref":
        return ref_ids
    # code_critical: intersection where ref mentions the identifier (case-insensitive)
    ref_lower = {r.lower() for r in ref_ids}
    critical = [c for c in code_ids if c.lower() in ref_lower]
    return critical

def identifier_preservation_rate_from_ids(source_ids, gen_summary: str) -> float:
    """
    Given a chosen set of source identifiers, measure what fraction appear in the generated summary
    (case-insensitive substring match). Returns NaN if no source ids.
    """
    if not source_ids:
        return float("nan")
    if not isinstance(gen_summary, str) or not gen_summary:
        return 0.0
    gl = gen_summary.lower()
    preserved = sum(1 for ident in source_ids if (ident in gen_summary) or (ident.lower() in gl))
    return preserved / max(1, len(source_ids))


# ---------------- Unicode script detection (expected script) ----------------

def char_script(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "Common"
    if "ARABIC" in name: return "Arabic"
    if "DEVANAGARI" in name: return "Devanagari"
    if ("CJK UNIFIED IDEOGRAPH" in name) or ("CJK" in name) or ("HIRAGANA" in name) or ("KATAKANA" in name) or ("BOPOMOFO" in name):
        return "Han"
    if "LATIN" in name: return "Latin"
    cat = unicodedata.category(ch)
    if cat.startswith(("P","S","Z","C","N")):
        return "Common"
    return "Other"

def token_script(token: str) -> str:
    counts = Counter(char_script(c) for c in token if not c.isspace())
    if "Common" in counts:
        del counts["Common"]
    if not counts:
        return "Common"
    return max(counts.items(), key=lambda kv: kv[1])[0]

EXPECTED_SCRIPTS = {
    "arabic": {"Arabic"},
    "hindi": {"Devanagari"},
    "chinese": {"Han"},
    "spanish": {"Latin"},
    "french": {"Latin"},
    "portuguese": {"Latin"},
    "english": {"Latin"},
    # default → Latin
}

WORD_SPLIT_RE = re.compile(r"[^\s]+")
PUNCT_TO_STRIP = ".,;:!?`'\"()[]{}"

IDENT_LIKE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")

def split_tokens(s: str):
    if not isinstance(s, str): return []
    return WORD_SPLIT_RE.findall(s)

def looks_like_code_token(tok: str, ident_whitelist: set) -> bool:
    s = tok.strip(PUNCT_TO_STRIP)
    sl = s.lower()
    if sl in ident_whitelist:
        return True
    # Heuristic: camelCase/snake_case/digits within → probably code-ish; allow it
    if IDENT_LIKE_RE.fullmatch(s) and ("_" in s or any(c.isupper() for c in s[1:]) or any(c.isdigit() for c in s)):
        return True
    return False

def transliteration_flag_expected(gen_summary: str, bt_language: str,
                                  identifier_whitelist: set,
                                  row_min_hits: int, row_min_ratio: float) -> int:
    """
    Rule: The generated summary should be in the expected script for bt_language.
    Any token whose dominant script is NOT expected (and is not a whitelisted identifier or code-like) is suspect.
    Row flagged if suspects >= row_min_hits OR suspect_ratio >= row_min_ratio.
    """
    if not isinstance(gen_summary, str) or not gen_summary.strip():
        return 0

    expected = EXPECTED_SCRIPTS.get(str(bt_language).lower(), {"Latin"})
    toks = split_tokens(gen_summary)
    if not toks:
        return 0

    suspect = 0
    for t in toks:
        if looks_like_code_token(t, identifier_whitelist):
            continue
        s = token_script(t)
        if s == "Common":
            continue
        if s not in expected and len(t) >= 3:
            suspect += 1

    ratio = suspect / max(1, len(toks))
    return int(suspect >= row_min_hits or ratio >= row_min_ratio)


# ---------------- Plotting helpers (matplotlib only) ----------------

def save_heatmap(matrix_df: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(matrix_df.values, aspect="auto")
    ax.set_xticks(range(matrix_df.shape[1]))
    ax.set_xticklabels(list(matrix_df.columns), rotation=45, ha="right")
    ax.set_yticks(range(matrix_df.shape[0]))
    ax.set_yticklabels(list(matrix_df.index))
    ax.set_title(title)
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            val = matrix_df.values[i, j]
            txt = "NA" if (isinstance(val, float) and np.isnan(val)) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_stacked_bar(flag_df: pd.DataFrame, title: str, out_path: Path):
    models = list(flag_df["model_name"])
    clean = flag_df["clean"].values
    idf = flag_df["id_flag"].values
    trf = flag_df["translit_flag"].values
    both = flag_df["both_flag"].values

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, clean, label="Clean")
    ax.bar(x, idf, bottom=clean, label="Identifier low")
    ax.bar(x, trf, bottom=clean+idf, label="Transliteration")
    ax.bar(x, both, bottom=clean+idf+trf, label="Both")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "aggregates").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = ["model_name","bt_language","reference_summary","generated_summary"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if args.id_source in {"code","code_critical"} and "code" not in df.columns:
        raise ValueError("id_source requires 'code' column, but it is missing.")
    if args.rel_to_english and "sample_id" not in df.columns:
        print("[warn] --rel_to_english requested but 'sample_id' column not found; skipping relative metrics.")
        args.rel_to_english = False

    # Build source identifiers + preservation (row-wise)
    def per_row_source_ids(row):
        ids = build_source_identifiers(row, args.id_source, args.id_min_len, args.id_max_per_row)
        return ids

    source_ids = df.apply(per_row_source_ids, axis=1)
    df["identifier_preservation"] = [
        identifier_preservation_rate_from_ids(ids, gen_summary=row["generated_summary"])
        for ids, row in zip(source_ids, df.to_dict(orient="records"))
    ]

    # Flag rows with identifier loss (skip NaNs)
    df["identifier_low_flag"] = np.where(
        df["identifier_preservation"].notna() & (df["identifier_preservation"] < args.id_preserve_thresh),
        1.0,
        np.where(df["identifier_preservation"].notna(), 0.0, np.nan)
    )

    # Expected-script transliteration flag with whitelist = source identifiers (lowercased)
    whitelists = [{i.lower() for i in ids} for ids in source_ids]
    df["transliteration_flag"] = [
        transliteration_flag_expected(
            gen_summary=row.generated_summary,
            bt_language=row.bt_language,
            identifier_whitelist=wl,
            row_min_hits=args.translit_row_min_hits,
            row_min_ratio=args.translit_row_min_ratio,
        )
        for row, wl in zip(df.itertuples(index=False), whitelists)
    ]

    # Optional: relative to English baseline
    df["identifier_preservation_rel_en"] = np.nan
    df["identifier_low_flag_rel_en"] = np.nan
    if args.rel_to_english:
        is_en = df["bt_language"].str.lower() == "english"
        # Only rows with defined id metric
        en_base = (
            df[is_en & df["identifier_preservation"].notna()]
            .set_index(["sample_id","model_name"])["identifier_preservation"]
        )
        if en_base.empty:
            print("[warn] No English baseline rows found; skipping relative metrics.")
        else:
            def map_en_baseline(row):
                key = (row["sample_id"], row["model_name"])
                return en_base.get(key, np.nan)

            df["id_pres_en_baseline"] = df.apply(map_en_baseline, axis=1)
            eps = 1e-6
            df["identifier_preservation_rel_en"] = np.where(
                df["id_pres_en_baseline"].notna(),
                df["identifier_preservation"] / (df["id_pres_en_baseline"] + eps),
                np.nan
            )
            df["identifier_low_flag_rel_en"] = (df["identifier_preservation_rel_en"] < args.rel_low_threshold).astype("float")

    # -------- Aggregations --------
    agg_fields = {
        "n": ("model_name","size"),
        "id_preserve_mean": ("identifier_preservation","mean"),
        "id_low_rate": ("identifier_low_flag","mean"),
        "translit_rate": ("transliteration_flag","mean"),
    }
    if args.rel_to_english:
        agg_fields.update({
            "id_preserve_rel_en_mean": ("identifier_preservation_rel_en","mean"),
            "id_low_rate_rel_en": ("identifier_low_flag_rel_en","mean"),
        })

    by_model_lang = df.groupby(["model_name","bt_language"]).agg(**agg_fields).reset_index()
    by_model = df.groupby("model_name").agg(**{k:v for k,v in agg_fields.items()}).reset_index()
    by_lang = df.groupby("bt_language").agg(**{k:v for k,v in agg_fields.items()}).reset_index()

    # Save aggregates
    by_model_lang.to_csv(outdir / "aggregates" / "by_model_language.csv", index=False)
    by_model.to_csv(outdir / "aggregates" / "by_model.csv", index=False)
    by_lang.to_csv(outdir / "aggregates" / "by_language.csv", index=False)

    # Heatmaps
    piv_id = by_model_lang.pivot(index="model_name", columns="bt_language", values="id_preserve_mean").astype(float)
    save_heatmap(piv_id, f"Identifier preservation (mean) [{args.id_source}]", outdir / "figures" / "heatmap_identifier_preservation.png")

    # Relative heatmap if available
    if args.rel_to_english and "id_preserve_rel_en_mean" in by_model_lang.columns:
        piv_id_rel = by_model_lang.pivot(index="model_name", columns="bt_language", values="id_preserve_rel_en_mean").astype(float)
        save_heatmap(piv_id_rel, f"Identifier preservation (relative to English) [{args.id_source}]", outdir / "figures" / "heatmap_identifier_preservation_rel_en.png")

    piv_tr = by_model_lang.pivot(index="model_name", columns="bt_language", values="translit_rate").astype(float)
    save_heatmap(piv_tr, "Transliteration rate (expected-script, code/REF whitelist)", outdir / "figures" / "heatmap_transliteration_rate.png")

    # Stacked bar per model (use rows where identifier metric is defined so bars sum to 1)
    def stacked_parts(g: pd.DataFrame) -> pd.Series:
        valid = g["identifier_low_flag"].notna()
        sub = g[valid]
        denom = max(1, len(sub))  # avoid div-by-zero
        clean = ((sub["identifier_low_flag"] == 0.0) & (sub["transliteration_flag"] == 0.0)).sum() / denom
        id_flag = (sub["identifier_low_flag"] == 1.0).sum() / denom
        translit_flag = (sub["transliteration_flag"] == 1.0).sum() / denom
        both_flag = ((sub["identifier_low_flag"] == 1.0) & (sub["transliteration_flag"] == 1.0)).sum() / denom
        return pd.Series({"clean": clean, "id_flag": id_flag, "translit_flag": translit_flag, "both_flag": both_flag})

    flag_comp = df.groupby("model_name").apply(stacked_parts).reset_index()
    save_stacked_bar(flag_comp, f"Fidelity flags by model [{args.id_source}]", outdir / "figures" / "stacked_bar_flags_by_model.png")

    # Console summary
    print("=== RQ-9 Fidelity Analysis (expert) ===")
    print("Identifier source:", args.id_source)
    print("Aggregates →", outdir / "aggregates")
    print("Figures    →", outdir / "figures")
    print("\nLowest identifier preservation (languages):")
    cols = ["bt_language","id_preserve_mean","id_low_rate"]
    print(by_lang.sort_values("id_preserve_mean").head(5)[cols])
    print("\nHighest transliteration rate (models):")
    print(by_model.sort_values("translit_rate", ascending=False).head(5)[["model_name","translit_rate"]])
    if args.rel_to_english and "id_preserve_rel_en_mean" in by_lang.columns:
        print("\nRelative to English — lowest identifier preservation (languages):")
        print(by_lang.sort_values("id_preserve_rel_en_mean").head(5)[["bt_language","id_preserve_rel_en_mean","id_low_rate_rel_en"]])


if __name__ == "__main__":
    main()
