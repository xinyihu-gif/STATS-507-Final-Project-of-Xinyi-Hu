#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:04:59 2025

@author: huxinyi
"""

"""
04_run_finbert.py

Run FinBERT ('yiyanghkust/finbert-tone') on sentence-level Item 1A text
and aggregate the results to the document (firm-year) level.

Steps:
- Load item1a_with_metrics.pkl
- Restrict to years 2010–2020
- Randomly sample up to N firm-year documents
- Flatten sentences across documents
- Run FinBERT in batches on CPU
- Aggregate mean neg/neu/pos probabilities and negative/positive shares
- Save the resulting document-level dataset

Input:
    data/item1a_with_metrics.pkl
Output:
    data/finbert_doclevel_sample_2010_2020.pkl
    data/finbert_doclevel_sample_2010_2020.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    in_path = data_dir / "item1a_with_metrics.pkl"
    out_pkl = data_dir / "finbert_doclevel_sample_2010_2020.pkl"
    out_csv = data_dir / "finbert_doclevel_sample_2010_2020.csv"

    print(f"Loading Item 1A metrics from {in_path} ...")
    df = pd.read_pickle(in_path)

    if "rf_item1a_sentences" not in df.columns:
        raise ValueError("Column 'rf_item1a_sentences' not found. Please run 03_compute_text_metrics.py first.")

    # Restrict to 2010–2020
    df_small = df[(df["year"] >= 2010) & (df["year"] <= 2020)].copy()
    print("Filtered to 2010–2020:")
    print(df_small["year"].value_counts().sort_index())

    target_n = 4000
    if len(df_small) <= target_n:
        df_finbert = df_small.copy()
    else:
        df_finbert = df_small.sample(n=target_n, random_state=507).copy()

    print("Sampled firm-year documents:")
    print(df_finbert["year"].value_counts().sort_index())

    # Flatten sentences into a separate DataFrame
    rows = []
    for doc_idx, sent_list in df_finbert["rf_item1a_sentences"].items():
        if not isinstance(sent_list, list) or len(sent_list) == 0:
            continue
        for s in sent_list:
            if not isinstance(s, str):
                continue
            s_strip = s.strip()
            if len(s_strip) < 5:
                continue
            rows.append((doc_idx, s_strip))

    sent_df = pd.DataFrame(rows, columns=["doc_idx", "sentence"])
    print(f"Total sentences to score with FinBERT: {len(sent_df)}")

    # Load FinBERT
    model_name = "yiyanghkust/finbert-tone"
    print(f"Loading FinBERT model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Run FinBERT in batches
    batch_size = 32
    n = len(sent_df)
    all_probs = []

    print("Running FinBERT inference on sentences...")
    for start in tqdm(range(0, n, batch_size)):
        batch_texts = sent_df["sentence"].iloc[start:start + batch_size].tolist()
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    sent_df["neg_prob"] = all_probs[:, 0]
    sent_df["neu_prob"] = all_probs[:, 1]
    sent_df["pos_prob"] = all_probs[:, 2]

    # Determine class winner per sentence
    sent_df["winner"] = sent_df[["neg_prob", "neu_prob", "pos_prob"]].values.argmax(axis=1)

    # Aggregate to document level
    print("Aggregating sentence-level probabilities to document level...")
    group = sent_df.groupby("doc_idx")
    agg = group.agg(
        fb_neg_mean=("neg_prob", "mean"),
        fb_neu_mean=("neu_prob", "mean"),
        fb_pos_mean=("pos_prob", "mean"),
        fb_neg_share=("winner", lambda x: (x == 0).mean()),
        fb_pos_share=("winner", lambda x: (x == 2).mean()),
    )

    print("FinBERT aggregate head:")
    print(agg.head())

    # Attach FinBERT metrics back to df_finbert
    fb_cols = ["fb_neg_mean", "fb_neu_mean", "fb_pos_mean", "fb_neg_share", "fb_pos_share"]
    df_finbert = df_finbert.drop(columns=[c for c in fb_cols if c in df_finbert.columns], errors="ignore")
    df_finbert = df_finbert.join(agg, how="left")

    print("Final FinBERT doc-level sample (head):")
    print(
        df_finbert[[
            "docID", "cik", "year",
            "mr_1d", "mr_5d", "mr_30d",
            "RF1A_n_sentences", "RF1A_total_words", "RF1A_avg_sent_len",
            "fb_neg_mean", "fb_neu_mean", "fb_pos_mean",
            "fb_neg_share", "fb_pos_share"
        ]].head()
    )

    print(f"Saving FinBERT doc-level sample to {out_pkl} and {out_csv} ...")
    df_finbert.to_pickle(out_pkl)
    df_finbert.to_csv(out_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
