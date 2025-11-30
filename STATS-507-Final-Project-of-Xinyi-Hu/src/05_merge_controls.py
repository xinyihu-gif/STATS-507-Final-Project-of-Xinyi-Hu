#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:05:16 2025

@author: huxinyi
"""

"""
05_merge_controls.py

Merge FinBERT-based firm-year sample with Compustat controls,
winsorize variables, and construct log-transformed controls.

Input:
    data/finbert_doclevel_sample_2010_2020.pkl
    data/compustat_controls.xlsx   (user-provided)
Output:
    data/final_dataset_with_controls.pkl
    data/final_dataset_with_controls.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Winsorize a numeric series at the p and 1-p quantiles."""
    lower = s.quantile(p)
    upper = s.quantile(1 - p)
    return np.clip(s, lower, upper)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    finbert_path = data_dir / "finbert_doclevel_sample_2010_2020.pkl"
    comp_path = data_dir / "compustat_controls.xlsx"

    out_pkl = data_dir / "final_dataset_with_controls.pkl"
    out_csv = data_dir / "final_dataset_with_controls.csv"

    print(f"Loading FinBERT sample from {finbert_path} ...")
    df_finbert = pd.read_pickle(finbert_path)

    print(f"Loading Compustat controls from {comp_path} ...")
    df_controls = pd.read_excel(comp_path)

    # Ensure key columns exist
    for col in ["cik", "year"]:
        if col not in df_controls.columns:
            raise ValueError(f"Column '{col}' must exist in Compustat controls.")

    # Align dtypes for merging
    df_finbert["cik"] = pd.to_numeric(df_finbert["cik"], errors="coerce").astype("Int64")
    df_finbert["year"] = pd.to_numeric(df_finbert["year"], errors="coerce").astype("Int64")

    df_controls["cik"] = pd.to_numeric(df_controls["cik"], errors="coerce").astype("Int64")
    df_controls["year"] = pd.to_numeric(df_controls["year"], errors="coerce").astype("Int64")

    # Keep only the columns needed from Compustat
    expected_fin_cols = ["Assets", "Liabilities", "PPE", "NI", "Sales", "incorp", "state"]
    missing = [c for c in expected_fin_cols if c not in df_controls.columns]
    if missing:
        print(f"Warning: missing columns in Compustat controls: {missing}")
    use_cols = ["cik", "year"] + [c for c in expected_fin_cols if c in df_controls.columns]

    df_all = df_finbert.merge(df_controls[use_cols], on=["cik", "year"], how="left")

    print("Merged dataset head:")
    print(
        df_all[[
            "docID", "cik", "year",
            "mr_1d", "mr_5d", "mr_30d",
            "Assets", "Liabilities", "PPE", "NI", "Sales"
        ]].head()
    )

    # Winsorize returns and key variables
    df_clean = df_all.copy()

    # Returns (binary or numeric; here we treat as numeric for consistency)
    for col in ["mr_1d", "mr_5d", "mr_30d"]:
        if col in df_clean.columns:
            df_clean[f"{col}_w"] = winsorize(df_clean[col])

    # Firm fundamentals
    for col in ["Assets", "Liabilities", "Sales", "PPE", "NI"]:
        if col in df_clean.columns:
            df_clean[f"{col}_w"] = winsorize(df_clean[col])

    # Text intensity
    if "RF1A_total_words" in df_clean.columns:
        df_clean["RF1A_total_words_w"] = winsorize(df_clean["RF1A_total_words"])

    # Winsorize FinBERT tone
    if "fb_neg_share" in df_clean.columns:
        df_clean["fb_neg_share_w"] = winsorize(df_clean["fb_neg_share"])

    # Log controls
    df_clean["log_assets"] = np.log1p(df_clean.get("Assets_w"))
    df_clean["log_liabilities"] = np.log1p(df_clean.get("Liabilities_w"))
    df_clean["log_sales"] = np.log1p(df_clean.get("Sales_w"))
    df_clean["log_ppe"] = np.log1p(df_clean.get("PPE_w"))

    if "NI_w" in df_clean.columns:
        min_NI = df_clean["NI_w"].min()
        df_clean["NI_shifted"] = df_clean["NI_w"] - (min_NI - 1)
        df_clean["log_NI"] = np.log(df_clean["NI_shifted"].replace({0: np.nan}))

    print("Final dataset with controls (head):")
    print(
        df_clean[[
            "docID", "cik", "year",
            "mr_1d_w", "mr_5d_w", "mr_30d_w",
            "log_assets", "log_sales", "log_liabilities", "log_ppe", "log_NI",
            "RF1A_total_words_w", "fb_neg_share_w"
        ]].head()
    )

    print(f"Saving final dataset to {out_pkl} and {out_csv} ...")
    df_clean.to_pickle(out_pkl)
    df_clean.to_csv(out_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
