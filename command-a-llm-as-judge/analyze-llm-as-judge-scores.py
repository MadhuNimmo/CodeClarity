#!/usr/bin/env python3
# coding: utf-8
"""
Analyze LLM-as-judge aggregation JSONs across judges and models.

Adds: Spearman correlation heatmaps (per dimension + overall) with annotations.
Outputs go to ./llm-as-judge-figures
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
BASES = {
    "gpt5nano": "/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/gpt-5-nano-llm-as-judge",
    "gemini2flashlite": "/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/gemini2-flash-lite-llm-as-judge",
    "commanda": "/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/command-a-llm-as-judge",
}
JUDGE_DISPLAY = {
    "gpt5nano": "GPT-5-nano",
    "gemini2flashlite": "Gemini-2-Flash-Lite",
    "commanda": "Command-A",
}
MODELS = ["codegemma", "deepseek", "gemma", "qwen"]

OUTDIR = Path("./llm-as-judge-figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def safe_load_json(p: Path):
    if not p.exists():
        print(f"[WARN] Missing file: {p}")
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse {p}: {e}")
        return None

def rows_from_payload(payload, judge_key, model_key):
    if payload is None:
        return []
    mbl = payload.get("means_by_language", {})
    cbl = payload.get("counts_by_language", {})
    n_samples = payload.get("n_samples", None)
    rows = []
    for lang, dim_map in mbl.items():
        for dim, mean_val in dim_map.items():
            count_val = None
            if isinstance(cbl.get(lang, {}), dict):
                count_val = cbl.get(lang, {}).get(dim, None)
            rows.append({
                "judge": judge_key,
                "model": model_key,
                "language": lang,
                "dimension": str(dim).lower(),
                "mean": float(mean_val) if mean_val is not None else np.nan,
                "count": int(count_val) if count_val is not None else np.nan,
                "n_samples": n_samples
            })
    return rows

# ----------------------------
# Ingest
# ----------------------------
all_rows = []
for judge_key, base in BASES.items():
    for model in MODELS:
        fpath = Path(base) / f"{model}_judgments_agg.json"
        payload = safe_load_json(fpath)
        all_rows.extend(rows_from_payload(payload, judge_key, model))

df = pd.DataFrame(all_rows)
if df.empty:
    raise SystemExit("No data loaded. Check paths / filenames.")

# Canonicalize names for presentation
df["judge_name"] = df["judge"].map(JUDGE_DISPLAY).fillna(df["judge"])

# ----------------------------
# Basic sanity and pivoted views
# ----------------------------
DIM_ORDER = ["correctness","completeness","clarity","terminology","brevity","overall"]
df = df[df["dimension"].isin(DIM_ORDER)]

# Long form saved
df.to_csv(OUTDIR / "judges_long.csv", index=False)

# Wide by dimension (one row per judge,model,language)
wide_dim = df.pivot_table(
    index=["judge_name","model","language"],
    columns="dimension",
    values="mean",
    aggfunc="mean"
).reset_index()
wide_dim.to_csv(OUTDIR / "judges_wide_by_dimension.csv", index=False)

# ----------------------------
# Aggregate across judges (per model, language, dimension)
# ----------------------------
agg_judges = df.groupby(["model","language","dimension"], as_index=False)["mean"].agg(
    mean_over_judges="mean",
)
std_judges = df.groupby(["model","language","dimension"], as_index=False)["mean"].agg(
    std_over_judges="std"
)
agg_judges = agg_judges.merge(std_judges, on=["model","language","dimension"], how="left")
agg_judges.to_csv(OUTDIR / "agg_over_judges_per_model_language_dimension.csv", index=False)

# ----------------------------
# Language difficulty index (average across models & judges)
# ----------------------------
lang_difficulty = df.groupby(["language","dimension"], as_index=False)["mean"].mean()
lang_overall = lang_difficulty[lang_difficulty["dimension"]=="overall"].copy()
lang_overall = lang_overall.sort_values("mean", ascending=False)
lang_overall.to_csv(OUTDIR / "language_overall_scores.csv", index=False)

# ----------------------------
# Judge agreement: Spearman correlations (per dimension) over (model,language)
# ----------------------------
def judge_pair_spearman(df_in, dim):
    sub = df_in[df_in["dimension"]==dim].copy()
    pivoted = sub.pivot_table(
        index=["model","language"],
        columns="judge_name",
        values="mean",
        aggfunc="mean"
    )
    corr = pivoted.corr(method="spearman")
    return corr

corr_tables = {}
for dim in DIM_ORDER:
    corr_tables[dim] = judge_pair_spearman(df, dim)
    corr_tables[dim].to_csv(OUTDIR / f"spearman_by_judge_{dim}.csv")

# ----------------------------
# FIGURES
# (Use matplotlib defaults; single plots; no explicit colors.)
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
})

# 1) Language difficulty: overall mean across models & judges
def plot_language_overall_bars(lang_overall_df):
    data = lang_overall_df.copy()
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(data["language"], data["mean"])
    plt.title("Overall score by language (avg across models & judges)")
    plt.xlabel("Language")
    plt.ylabel("Overall score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_language_overall_bars.png")
    plt.close()

plot_language_overall_bars(lang_overall)

# 2) Judge variance by language (overall): std across judges, aggregated over models
judge_var_overall = (
    df[df["dimension"]=="overall"]
    .groupby(["language","judge_name"], as_index=False)["mean"].mean()
    .groupby("language", as_index=False)["mean"].std()
    .rename(columns={"mean":"std_over_judges_overall"})
)

def plot_judge_variance_bars(jv_df):
    d = jv_df.sort_values("std_over_judges_overall", ascending=False)
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(d["language"], d["std_over_judges_overall"])
    plt.title("Judge variance by language (Overall; lower = more agreement)")
    plt.xlabel("Language")
    plt.ylabel("Std across judges")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_judge_variance_overall_by_language.png")
    plt.close()

plot_judge_variance_bars(judge_var_overall)

# 3) Per-model bar charts across languages (overall; averaged across judges)
per_model_lang = agg_judges[agg_judges["dimension"]=="overall"].copy()
for m in MODELS:
    sub = per_model_lang[per_model_lang["model"]==m].sort_values("mean_over_judges", ascending=False)
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(sub["language"], sub["mean_over_judges"])
    plt.title(f"{m}: Overall score by language (avg across judges)")
    plt.xlabel("Language")
    plt.ylabel("Overall score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"fig_{m}_overall_by_language.png")
    plt.close()

# 4) Heatmap: language × model for Overall (avg across judges)
def plot_overall_heatmap(per_model_df):
    table = per_model_df.pivot_table(
        index="language", columns="model", values="mean_over_judges", aggfunc="mean"
    )
    lang_order = lang_overall["language"].tolist()
    table = table.reindex(lang_order)
    plt.figure(figsize=(6.6, 4.8))
    plt.imshow(table.values, aspect="auto")
    plt.title("Overall scores: language × model (avg across judges)")
    plt.xlabel("Model")
    plt.ylabel("Language")
    plt.xticks(ticks=range(len(table.columns)), labels=table.columns, rotation=30, ha="right")
    plt.yticks(ticks=range(len(table.index)), labels=table.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig_overall_heatmap_language_by_model.png")
    plt.close()

plot_overall_heatmap(per_model_lang)

# 5) NEW: Spearman correlation heatmaps (per dimension + overall)
def plot_corr_heatmap(corr_df: pd.DataFrame, title: str, out_name: str):
    # Ensure ordering is consistent and diagonals are present
    judges = list(corr_df.index)
    corr_mat = corr_df.loc[judges, judges].values

    plt.figure(figsize=(4.2, 3.8))
    im = plt.imshow(corr_mat, aspect="equal", vmin=-1.0, vmax=1.0)
    plt.title(title)
    plt.xticks(ticks=range(len(judges)), labels=judges, rotation=30, ha="right")
    plt.yticks(ticks=range(len(judges)), labels=judges)

    # Annotate cells with correlation values
    n = len(judges)
    for i in range(n):
        for j in range(n):
            val = corr_mat[i, j]
            if np.isnan(val):
                text = "—"
            else:
                text = f"{val:.3f}" if i != j else "1.000"
            # Use a modest fontsize so numbers don't crowd
            plt.text(j, i, text, ha="center", va="center")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTDIR / out_name)
    plt.close()

# Per-dimension correlation heatmaps
for dim in DIM_ORDER:
    corr = corr_tables[dim]
    # Use display names for consistency
    corr.index = [c for c in corr.index]
    corr.columns = [c for c in corr.columns]
    plot_corr_heatmap(corr,
                      title=f"Judge–Judge Spearman ({dim.capitalize()})",
                      out_name=f"fig_spearman_heatmap_{dim}.png")

# Additionally export a prominently named "overall" figure
plot_corr_heatmap(
    corr_tables["overall"],
    title="Judge–Judge Spearman (Overall)",
    out_name="fig_spearman_heatmap_overall.png"
)

# ----------------------------
# Quick textual takeaways (prints)
# ----------------------------
print("\n=== Top languages by overall (avg across models & judges) ===")
print(lang_overall.head(6).to_string(index=False))

print("\n=== Bottom languages by overall (avg across models & judges) ===")
print(lang_overall.tail(6).to_string(index=False))

print("\n=== Judge–Judge Spearman (Overall) ===")
print(corr_tables["overall"].round(3))

print("\nDONE. CSVs and PNGs saved under:", OUTDIR.resolve())
