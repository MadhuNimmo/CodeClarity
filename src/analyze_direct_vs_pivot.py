#!/usr/bin/env python3
# coding: utf-8
"""
Direct vs Pivot (paper figures, minimal + publication ready)

Outputs (CSVs + PNGs):
  /Users/madhurimachakraborty/Documents/GitHub/CodeClarity/direct_vs_pivot_results

Figures produced:
  1) fig_dvp_overall_delta_sorted.png
  2) fig_dvp_overall_means_grouped.png
  3) fig_dvp_overall_delta_heatmap_model_x_language.png
  4) fig_dvp_dimension_deltas_hindi.png
  5) fig_dvp_dimension_deltas_arabic.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CONFIG: your 4 model JSONs
# -------------------------
MODEL_FILES = {
    "gemma":     "/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/direct_vs_pivot_results/gemma_pairwise_direct_vs_pivot_agg.json",
    "codegemma": "/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/direct_vs_pivot_results/codegemma_pairwise_direct_vs_pivot_agg.json",
    "deepseek":  "/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/direct_vs_pivot_results/deepseek_pairwise_direct_vs_pivot_agg.json",
    "qwen":      "/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/direct_vs_pivot_results/qwen_pairwise_direct_vs_pivot_agg.json",
}

OUTDIR = Path("/Users/madhurimachakraborty/Documents/GitHub/CodeClarity/direct_vs_pivot_results")
OUTDIR.mkdir(exist_ok=True, parents=True)

DIM_ORDER = ["correctness","completeness","clarity","terminology","brevity","overall"]
LANG_ORDER = ["chinese","french","spanish","portuguese","arabic","hindi"]  # will re-order for sorted plot

# -------------------------
# LOAD
# -------------------------
rows = []
for model, fpath in MODEL_FILES.items():
    p = Path(fpath)
    if not p.exists():
        print(f"[WARN] Missing: {p}")
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    aggs = data.get("aggregates", {})
    dims = data.get("dimensions", DIM_ORDER)

    for lang, bundle in aggs.items():
        for dim in dims:
            wA = bundle["wins_A_direct"].get(dim, np.nan)
            wB = bundle["wins_B_pivot"].get(dim, np.nan)
            ties = bundle["ties"].get(dim, np.nan)
            mA = bundle["mean_scores_A"].get(dim, np.nan)
            mB = bundle["mean_scores_B"].get(dim, np.nan)
            n  = bundle["count_scores"].get(dim, np.nan)
            rows.append({
                "model": model,
                "language": lang,
                "dimension": dim,
                "wins_A": wA,
                "wins_B": wB,
                "ties": ties,
                "mean_A": mA,
                "mean_B": mB,
                "count": n
            })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No data loaded. Check MODEL_FILES paths.")

df["dimension"] = pd.Categorical(df["dimension"], categories=DIM_ORDER, ordered=True)

# Save long form (for reproducibility)
df.to_csv(OUTDIR / "direct_vs_pivot_long.csv", index=False)

# -------------------------
# AGGREGATIONS
# -------------------------
# Win shares (for grouped means we won’t need them, but keep CSV)
win_df = df.copy()
win_df["total_compared"] = win_df["wins_A"] + win_df["wins_B"] + win_df["ties"]
for col in ["wins_A","wins_B","ties"]:
    win_df[col + "_share"] = win_df[col] / win_df["total_compared"]

wins_summary = win_df.groupby(
    ["model","language","dimension"], as_index=False, observed=False
)[["wins_A","wins_B","ties","wins_A_share","wins_B_share","ties_share","total_compared"]].sum()
wins_summary.to_csv(OUTDIR / "direct_vs_pivot_wins_summary.csv", index=False)

# Mean deltas
df["delta_mean"] = df["mean_B"] - df["mean_A"]  # + => Pivot better
delta_by_model = df.groupby(
    ["model","language","dimension"], as_index=False, observed=False
)["delta_mean"].mean()

# Cross-model mean delta + CI
agg = df.groupby(
    ["language","dimension"], as_index=False, observed=False
)["delta_mean"].agg(["mean","std","count"]).reset_index()
agg = agg.rename(columns={"mean":"delta_mean_cross_model","std":"delta_std","count":"n_models"})
agg["delta_ci95"] = 1.96 * agg["delta_std"] / np.sqrt(agg["n_models"].replace(0, np.nan))
agg.to_csv(OUTDIR / "direct_vs_pivot_mean_deltas.csv", index=False)

# Cross-model direct and pivot overall means (for grouped bars)
overall_means = df[df["dimension"]=="overall"].groupby(
    ["language"], as_index=False, observed=False
)[["mean_A","mean_B"]].mean()

# -------------------------
# MATPLOTLIB AESTHETICS
# -------------------------
plt.rcParams.update({
    "figure.dpi": 180,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
})

# -------------------------
# FIG 1: Sorted Δ (Pivot–Direct), Overall
# -------------------------
overall_delta = agg[agg["dimension"]=="overall"].copy()
overall_delta = overall_delta.sort_values("delta_mean_cross_model", ascending=True)  # ascending → Hindi/Arabic at top if large positive
langs_sorted = overall_delta["language"].tolist()

plt.figure(figsize=(7.8, 4.2))
x = np.arange(len(overall_delta))
plt.bar(x, overall_delta["delta_mean_cross_model"])
yerr = overall_delta["delta_ci95"].fillna(0).values
plt.errorbar(x, overall_delta["delta_mean_cross_model"], yerr=yerr, fmt="none", capsize=3)
plt.axhline(0, linewidth=1)
plt.xticks(x, overall_delta["language"], rotation=25, ha="right")
plt.ylabel("Δ mean (Pivot – Direct)")
plt.title("Direct vs Pivot: Overall Δ mean by language (sorted)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig_dvp_overall_delta_sorted.png")
plt.close()

# -------------------------
# FIG 2: Grouped bars — Direct vs Pivot overall means (languages in same sorted order)
# -------------------------
om = overall_means.set_index("language").reindex(langs_sorted).reset_index()
plt.figure(figsize=(8.2, 4.4))
x = np.arange(len(om))
w = 0.38
plt.bar(x - w/2, om["mean_A"], width=w, label="Direct (A)")
plt.bar(x + w/2, om["mean_B"], width=w, label="Pivot (B)")
plt.xticks(x, om["language"], rotation=25, ha="right")
plt.ylabel("Overall mean score")
plt.title("Direct vs Pivot: Overall mean by language")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "fig_dvp_overall_means_grouped.png")
plt.close()

# -------------------------
# FIG 3: Heatmap — Δ mean by Model × Language (Overall)
# -------------------------
hm = delta_by_model[delta_by_model["dimension"]=="overall"].pivot(
    index="language", columns="model", values="delta_mean"
).reindex(index=langs_sorted)  # use same order as fig 1

plt.figure(figsize=(6.6, 4.6))
plt.imshow(hm.values, aspect="auto")
plt.xticks(ticks=np.arange(len(hm.columns)), labels=hm.columns, rotation=25, ha="right")
plt.yticks(ticks=np.arange(len(hm.index)), labels=hm.index)
plt.colorbar()
plt.title("Direct vs Pivot: Overall Δ mean by Model × Language")
plt.xlabel("Model"); plt.ylabel("Language")
plt.tight_layout()
plt.savefig(OUTDIR / "fig_dvp_overall_delta_heatmap_model_x_language.png")
plt.close()

# -------------------------
# FIG 4 & 5: Dimension deltas for Hindi and Arabic only (avg across models)
# -------------------------
def plot_dim_deltas_for_language(lang: str, fname: str):
    sub = agg[(agg["language"]==lang)].copy().sort_values("dimension")
    plt.figure(figsize=(7.8, 4.2))
    x = np.arange(len(sub))
    plt.bar(x, sub["delta_mean_cross_model"])
    yerr = sub["delta_ci95"].fillna(0).values
    plt.errorbar(x, sub["delta_mean_cross_model"], yerr=yerr, fmt="none", capsize=3)
    plt.axhline(0, linewidth=1)
    labels = [d.capitalize() for d in sub["dimension"]]
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Δ mean (Pivot – Direct)")
    plt.title(f"Direct vs Pivot: {lang.capitalize()} by dimension (avg across models)")
    plt.tight_layout()
    plt.savefig(OUTDIR / fname)
    plt.close()

plot_dim_deltas_for_language("hindi", "fig_dvp_dimension_deltas_hindi.png")
plot_dim_deltas_for_language("arabic", "fig_dvp_dimension_deltas_arabic.png")

print("Done. Outputs in:", OUTDIR.resolve())
