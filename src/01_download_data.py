#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:02:54 2025

@author: huxinyi
"""

"""
01_download_data.py

Download the HuggingFace dataset "JanosAudran/financial-reports-sec",
convert the train split to a pandas DataFrame, and save it locally.

This script should be run from the project root directory:

    python src/01_download_data.py
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path


def main():
    # Set up paths
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / "sec_sentences.parquet"

    print("Loading HuggingFace dataset: JanosAudran/financial-reports-sec (large_lite)...")
    ds = load_dataset("JanosAudran/financial-reports-sec", "large_lite")
    train = ds["train"]

    print("Converting to pandas DataFrame...")
    df = train.to_pandas()

    # Map section IDs to human-readable names (e.g. "section_1A")
    sec_names = train.features["section"].names
    id2name = {i: n for i, n in enumerate(sec_names)}
    df["section_name"] = df["section"].map(id2name)

    print("Sample of data:")
    print(df.head())

    print(f"Saving full sentence-level data with section_name to {out_path} ...")
    df.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
