#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:04:29 2025

@author: huxinyi
"""

"""
03_compute_text_metrics.py

Compute basic text-structure features for the Item 1A section:
- number of sentences
- total word count
- average sentence length

Also split Item 1A text into a sentence list, which will later be fed
into FinBERT.

Input:
    data/item1a_documents.parquet
Output:
    data/item1a_with_metrics.pkl
    data/item1a_with_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def main():
    # Ensure NLTK tokenizers are available
    nltk.download("punkt", quiet=True)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    in_path = data_dir / "item1a_documents.parquet"
    out_pkl = data_dir / "item1a_with_metrics.pkl"
    out_csv = data_dir / "item1a_with_metrics.csv"

    print(f"Loading Item 1A documents from {in_path} ...")
    df = pd.read_parquet(in_path)

    if "rf_item1a" not in df.columns:
        raise ValueError("Column 'rf_item1a' not found. Please run 02_extract_item1a.py first.")

    # Initialize metrics
    df["RF1A_n_sentences"] = 0
    df["RF1A_total_words"] = 0
    df["RF1A_avg_sent_len"] = 0.0
    df["rf_item1a_sentences"] = None

    print("Computing sentence- and word-level statistics for Item 1A...")
    for idx, text in df["rf_item1a"].items():
        if not isinstance(text, str) or not text.strip():
            df.at[idx, "rf_item1a_sentences"] = []
            continue

        # Sentence tokenization
        sents = sent_tokenize(text)
        df.at[idx, "rf_item1a_sentences"] = sents
        df.at[idx, "RF1A_n_sentences"] = len(sents)

        # Word tokenization (over full text)
        words = word_tokenize(text)
        df.at[idx, "RF1A_total_words"] = len(words)

        if len(sents) > 0:
            df.at[idx, "RF1A_avg_sent_len"] = len(words) / len(sents)
        else:
            df.at[idx, "RF1A_avg_sent_len"] = np.nan

    print("Example of computed metrics:")
    print(
        df[[
            "docID", "cik", "year",
            "RF1A_n_sentences", "RF1A_total_words", "RF1A_avg_sent_len"
        ]].head()
    )

    print(f"Saving metrics + sentences to {out_pkl} and {out_csv} ...")
    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
