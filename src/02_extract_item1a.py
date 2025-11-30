#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:03:53 2025

@author: huxinyi
"""

"""
02_extract_item1a.py

Reconstruct full 10-K text by (cik, filingDate) from sentence-level data,
then extract the Item 1A "Risk Factors" section using regex-based boundaries.

Also collapse the HuggingFace labels into firm-year level market reaction
variables (1d, 5d, 30d).

Input:
    data/sec_sentences.parquet
Output:
    data/item1a_documents.parquet
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def extract_item_1a(full_text: str) -> Optional[str]:
    """
    Extract the Item 1A section from a full 10-K document text.
    Returns the substring corresponding to Item 1A; if not found, returns np.nan.
    """
    if not isinstance(full_text, str):
        return np.nan

    t_upper = full_text.upper()

    # Possible patterns for Item 1A heading
    start_patterns = [
        r"ITEM\s+1A\.\s*RISK FACTORS",
        r"ITEM\s+1A\s*[-:]\s*RISK FACTORS",
        r"ITEM\s+1A\s+RISK\s+FACTORS",
        r"ITEM\s+1A\.\s*RISK\s+FACTOR",
    ]

    # Possible "next section" anchors
    end_patterns = [
        r"ITEM\s+1B\b",
        r"ITEM\s+2\b",
        r"ITEM\s+7\b",
        r"ITEM\s+7A\b",
        r"ITEM\s+8\b",
        r"ITEM\s+9\b",
    ]

    start = None
    for pat in start_patterns:
        m = re.search(pat, t_upper)
        if m:
            start = m.start()
            break

    if start is None:
        return np.nan

    tail = t_upper[start:]
    end = None
    for pat in end_patterns:
        m = re.search(pat, tail)
        if m:
            candidate = m.start()
            if end is None or candidate < end:
                end = candidate

    if end is None:
        item1a_text = full_text[start:]
    else:
        item1a_text = full_text[start:start + end]

    return item1a_text.strip()


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    in_path = data_dir / "sec_sentences.parquet"
    out_path = data_dir / "item1a_documents.parquet"

    print(f"Loading sentence-level data from {in_path} ...")
    df = pd.read_parquet(in_path)

    # Ensure key columns exist
    expected_cols = {"docID", "cik", "filingDate", "sentence", "labels"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Convert CIK to integer (if possible)
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")

    # Reconstruct full 10-K text per (cik, filingDate, docID)
    print("Reconstructing full filing text by (cik, filingDate, docID)...")
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")

    docs = (
        df
        .sort_values(["cik", "filingDate", "docID"])
        .groupby(["docID", "cik", "filingDate"], as_index=False)["sentence"]
        .agg(lambda s: " ".join(s.astype(str)))
    )
    docs = docs.rename(columns={"sentence": "full_text"})
    docs["year"] = docs["filingDate"].dt.year.astype("Int64")

    print("Example reconstructed document:")
    print(docs[["docID", "cik", "year", "full_text"]].head(1))

    # Collapse labels to document level (assuming labels constant within docID)
    print("Collapsing market reaction labels to docID level...")
    lab = (
        df[["docID", "labels"]]
        .drop_duplicates("docID")
        .copy()
    )

    def extract_label(d, key):
        if not isinstance(d, dict):
            return np.nan
        return d.get(key, np.nan)

    lab["mr_1d"] = lab["labels"].apply(lambda d: extract_label(d, "1d"))
    lab["mr_5d"] = lab["labels"].apply(lambda d: extract_label(d, "5d"))
    lab["mr_30d"] = lab["labels"].apply(lambda d: extract_label(d, "30d"))
    lab = lab.drop(columns=["labels"])

    # Merge document text + labels
    doc_df = docs.merge(lab, on="docID", how="left")

    # Extract Item 1A from reconstructed full_text
    print("Extracting Item 1A section via regex...")
    doc_df["rf_item1a"] = doc_df["full_text"].apply(extract_item_1a)

    success_ratio = doc_df["rf_item1a"].notna().mean()
    print(f"Share of documents with a non-missing Item 1A section: {success_ratio:.2%}")

    print(f"Saving Item 1A document-level dataset to {out_path} ...")
    doc_df.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
