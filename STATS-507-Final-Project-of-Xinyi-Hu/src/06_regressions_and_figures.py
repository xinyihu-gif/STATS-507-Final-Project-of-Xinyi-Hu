#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:05:33 2025

@author: huxinyi
"""

"""
06_regressions_and_figures.py

Run the main empirical regressions and generate figures.

- FinBERT model: negative tone vs. 1d/5d/30d returns
- Text-structure model: length/complexity features vs. returns
- Both with controls and fixed effects

Also produce a set of diagnostic and illustrative plots.

Input:
    data/final_dataset_with_controls.pkl
Output:
    results/reg_table_finbert.csv
    results/reg_table_length.csv
    figures/*.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def add_stars(p: float) -> str:
    """Return conventional significance stars based on p-value."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def run_finbert_model(df: pd.DataFrame, dep_var: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS with:
    - Dep var: dep_var (mr_1d_w / mr_5d_w / mr_30d_w)
    - Main var: fb_neg_share_w
    - Controls: log_assets, log_sales, log_liabilities, log_ppe, log_NI, RF1A_total_words_w
    - Fixed effects: year FE + incorporation FE (based on 'incorp')
    """
    df_work = df.copy()

    # Encode year as FE
    year_dummies = pd.get_dummies(df_work["year"].astype(int), prefix="yearFE", dummy_na=False)

    # Encode incorporation state FE (if available)
    if "incorp" in df_work.columns:
        df_work["incorp_code"], _ = pd.factorize(df_work["incorp"])
        incorp_dummies = pd.get_dummies(df_work["incorp_code"], prefix="incorpFE", dummy_na=False)
    else:
        incorp_dummies = pd.DataFrame(index=df_work.index)

    df_work = pd.concat([df_work, year_dummies, incorp_dummies], axis=1)

    main_var = "fb_neg_share_w"
    controls = ["log_assets", "log_sales", "log_liabilities", "log_ppe", "log_NI", "RF1A_total_words_w"]

    fe_cols = [c for c in df_work.columns if c.startswith("yearFE") or c.startswith("incorpFE")]

    # Drop one category per FE block to avoid dummy variable trap
    year_cols = sorted([c for c in fe_cols if c.startswith("yearFE")])
    incorp_cols = sorted([c for c in fe_cols if c.startswith("incorpFE")])
    if year_cols:
        year_cols = year_cols[1:]
    if incorp_cols:
        incorp_cols = incorp_cols[1:]

    reg_cols = [main_var] + controls + year_cols + incorp_cols
    cols_for_reg = [dep_var] + reg_cols

    df_reg = df_work[cols_for_reg].copy()
    df_reg = df_reg.apply(pd.to_numeric, errors="coerce")
    df_reg = df_reg.dropna()

    y = df_reg[dep_var].astype(float)
    X = df_reg[[c for c in df_reg.columns if c != dep_var]].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


def run_length_model(df: pd.DataFrame, dep_var: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS with:
    - Dep var: dep_var
    - Main vars: RF1A_total_words_w, RF1A_avg_sent_len, RF1A_n_sentences
    - Controls: log_assets, log_sales, log_liabilities, log_ppe, log_NI
    - FE: year + incorporation
    """
    df_work = df.copy()

    year_dummies = pd.get_dummies(df_work["year"].astype(int), prefix="yearFE", dummy_na=False)

    if "incorp" in df_work.columns:
        df_work["incorp_code"], _ = pd.factorize(df_work["incorp"])
        incorp_dummies = pd.get_dummies(df_work["incorp_code"], prefix="incorpFE", dummy_na=False)
    else:
        incorp_dummies = pd.DataFrame(index=df_work.index)

    df_work = pd.concat([df_work, year_dummies, incorp_dummies], axis=1)

    main_vars = ["RF1A_total_words_w", "RF1A_avg_sent_len", "RF1A_n_sentences"]
    controls = ["log_assets", "log_sales", "log_liabilities", "log_ppe", "log_NI"]

    fe_cols = [c for c in df_work.columns if c.startswith("yearFE") or c.startswith("incorpFE")]
    year_cols = sorted([c for c in fe_cols if c.startswith("yearFE")])
    incorp_cols = sorted([c for c in fe_cols if c.startswith("incorpFE")])
    if year_cols:
        year_cols = year_cols[1:]
    if incorp_cols:
        incorp_cols = incorp_cols[1:]

    reg_cols = main_vars + controls + year_cols + incorp_cols
    cols_for_reg = [dep_var] + reg_cols

    df_reg = df_work[cols_for_reg].copy()
    df_reg = df_reg.apply(pd.to_numeric, errors="coerce")
    df_reg = df_reg.dropna()

    y = df_reg[dep_var].astype(float)
    X = df_reg[[c for c in df_reg.columns if c != dep_var]].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


def build_regression_table(models: dict, var_list: list) -> pd.DataFrame:
    """
    Build a wide regression table with significance stars.
    models: dict like {"1-day": model1, "5-day": model2, ...}
    var_list: variables to display in rows.
    """
    rows = []
    for var in var_list:
        row = [var]
        for name, model in models.items():
            if var in model.params.index:
                coef = model.params[var]
                pval = model.pvalues[var]
                row.append(f"{coef:.4f}{add_stars(pval)}")
            else:
                row.append("")
        rows.append(row)

    # Add N and Adj R²
    rows.append(["N"] + [str(int(m.nobs)) for m in models.values()])
    rows.append(["Adj R²"] + [f"{m.rsquared_adj:.3f}" for m in models.values()])

    cols = ["Variable"] + list(models.keys())
    table_df = pd.DataFrame(rows, columns=cols)
    return table_df


def make_figures(df: pd.DataFrame, figures_dir: Path):
    """Generate diagnostic and illustrative plots, saving them as PNG files."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Histogram of FinBERT negative tone
    if "fb_neg_share_w" in df.columns:
        plt.figure(figsize=(6, 4))
        series = df["fb_neg_share_w"].dropna()
        plt.hist(series, bins=30, edgecolor="black", alpha=0.7)
        for q in [0.1, 0.5, 0.9]:
            v = series.quantile(q)
            plt.axvline(v, linestyle="--")
            plt.text(v, plt.ylim()[1] * 0.9, f"{int(q * 100)}%", rotation=90,
                     va="top", ha="right")
        plt.xlabel("FinBERT negative tone share (fb_neg_share_w)")
        plt.ylabel("Firm-year count")
        plt.title("Distribution of FinBERT-based negative tone")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig1_hist_fb_neg_share.png", dpi=300)
        plt.close()

    # 2. Quintile bar chart: 5-day return by tone quintile
    if {"fb_neg_share_w", "mr_5d_w"} <= set(df.columns):
        df_plot = df[["fb_neg_share_w", "mr_5d_w"]].dropna().copy()
        df_plot["tone_quintile"] = pd.qcut(df_plot["fb_neg_share_w"], 5, labels=[1, 2, 3, 4, 5])
        group_mean = df_plot.groupby("tone_quintile")["mr_5d_w"].mean().reset_index()

        plt.figure(figsize=(6, 4))
        plt.bar(group_mean["tone_quintile"].astype(str), group_mean["mr_5d_w"])
        plt.xlabel("Quintile of FinBERT negative tone (1 = lowest, 5 = highest)")
        plt.ylabel("Average 5-day return (winsorized)")
        plt.title("Average 5-day return by FinBERT negative tone quintile")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig2_quintile_mr5d_by_tone.png", dpi=300)
        plt.close()

    # 3. Correlation heatmap: tone, text features, and returns
    cols = [
        c for c in [
            "fb_neg_share_w",
            "RF1A_total_words_w",
            "RF1A_avg_sent_len",
            "RF1A_n_sentences",
            "mr_5d_w"
        ] if c in df.columns
    ]
    if len(cols) >= 3:
        corr = df[cols].dropna().corr()
        plt.figure(figsize=(5, 4))
        im = plt.imshow(corr, interpolation="nearest")
        plt.colorbar(im)
        plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
        plt.yticks(range(len(cols)), cols)
        plt.title("Correlation: tone, text features, and 5-day returns")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig3_corr_matrix.png", dpi=300)
        plt.close()

    # 4. LOWESS smooth of tone vs 5-day return
    if {"fb_neg_share_w", "mr_5d_w"} <= set(df.columns):
        df_plot = df[["fb_neg_share_w", "mr_5d_w"]].dropna()
        lowess = sm.nonparametric.lowess
        z = lowess(df_plot["mr_5d_w"], df_plot["fb_neg_share_w"], frac=0.3)

        plt.figure(figsize=(6, 4))
        plt.scatter(df_plot["fb_neg_share_w"], df_plot["mr_5d_w"], alpha=0.2, s=10)
        plt.plot(z[:, 0], z[:, 1], linewidth=2)
        plt.xlabel("FinBERT negative tone share")
        plt.ylabel("5-day return (winsorized)")
        plt.title("LOWESS: negative tone vs. 5-day market reaction")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig4_lowess_tone_vs_mr5d.png", dpi=300)
        plt.close()

    # 5. Hexbin of tone vs 5-day return
    if {"fb_neg_share_w", "mr_5d_w"} <= set(df.columns):
        plt.figure(figsize=(6, 5))
        plt.hexbin(df["fb_neg_share_w"], df["mr_5d_w"], gridsize=30)
        plt.colorbar(label="Count")
        plt.xlabel("FinBERT negative tone share")
        plt.ylabel("5-day return (winsorized)")
        plt.title("Joint distribution: tone and 5-day market reaction")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig5_hexbin_tone_mr5d.png", dpi=300)
        plt.close()

    # 6. Radar plot of average FinBERT probabilities
    for col in ["fb_neg_mean", "fb_neu_mean", "fb_pos_mean"]:
        if col not in df.columns:
            break
    else:
        vals = [
            df["fb_neg_mean"].mean(),
            df["fb_neu_mean"].mean(),
            df["fb_pos_mean"].mean(),
        ]
        labels = ["Negative", "Neutral", "Positive"]
        angles = np.linspace(0, 2 * np.pi, len(vals), endpoint=False)
        vals = np.concatenate((vals, [vals[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.2)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        plt.title("Average FinBERT sentiment composition in Item 1A")
        plt.tight_layout()
        plt.savefig(figures_dir / "fig6_radar_finbert_probs.png", dpi=300)
        plt.close()


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    results_dir = root / "results"
    figures_dir = root / "figures"

    data_path = data_dir / "final_dataset_with_controls.pkl"
    print(f"Loading final dataset from {data_path} ...")
    df = pd.read_pickle(data_path)

    # --- Run FinBERT models ---
    finbert_models = {}
    for dep in ["mr_1d_w", "mr_5d_w", "mr_30d_w"]:
        if dep not in df.columns:
            raise ValueError(f"Dependent variable '{dep}' not found in dataset.")
        print(f"Fitting FinBERT model for {dep} ...")
        finbert_models[dep] = run_finbert_model(df, dep)

    # Build FinBERT regression table
    finbert_var_list = [
        "fb_neg_share_w",
        "log_assets",
        "log_sales",
        "log_liabilities",
        "log_ppe",
        "log_NI",
        "RF1A_total_words_w",
    ]
    finbert_table = build_regression_table(
        models={
            "1-day": finbert_models["mr_1d_w"],
            "5-day": finbert_models["mr_5d_w"],
            "30-day": finbert_models["mr_30d_w"],
        },
        var_list=finbert_var_list,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    finbert_out = results_dir / "reg_table_finbert.csv"
    finbert_table.to_csv(finbert_out, index=False)
    print(f"Saved FinBERT regression table to {finbert_out}")

    # --- Run text-structure (length) models ---
    length_models = {}
    for dep in ["mr_1d_w", "mr_5d_w", "mr_30d_w"]:
        print(f"Fitting length model for {dep} ...")
        length_models[dep] = run_length_model(df, dep)

    length_var_list = [
        "RF1A_total_words_w",
        "RF1A_avg_sent_len",
        "RF1A_n_sentences",
        "log_assets",
        "log_sales",
        "log_liabilities",
        "log_ppe",
        "log_NI",
    ]
    length_table = build_regression_table(
        models={
            "1-day": length_models["mr_1d_w"],
            "5-day": length_models["mr_5d_w"],
            "30-day": length_models["mr_30d_w"],
        },
        var_list=length_var_list,
    )

    length_out = results_dir / "reg_table_length.csv"
    length_table.to_csv(length_out, index=False)
    print(f"Saved length regression table to {length_out}")

    # --- Generate figures ---
    make_figures(df, figures_dir)
    print(f"Figures saved in {figures_dir}")


if __name__ == "__main__":
    main()
