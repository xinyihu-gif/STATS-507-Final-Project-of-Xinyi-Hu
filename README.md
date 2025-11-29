# STATS-507-Final-Project-of-Xinyi-Hu

Overview

This repository contains all code, files, and results for my STATS 507 final project, which studies how the tone and linguistic structure of Item 1A “Risk Factors” in 10-K filings relate to short-horizon stock returns.
Using large-scale SEC filings from HuggingFace and financial variables from Compustat, I evaluate whether FinBERT-based sentiment or text-structure characteristics (word count, sentence length, sentence count) better explain market reactions.

Research Motivation

Risk Factor disclosures are one of the few highly standardized and legally mandated sections of 10-K reports. Prior accounting and finance research shows that both how firms describe risks and the tone they use can meaningfully affect how investors update beliefs.
However, most earlier studies rely on dictionary methods that cannot capture contextual meaning. This project adopts FinBERT, a domain-specific transformer model, to examine whether modern NLP tools offer stronger predictive content.

Main Objectives

Build a full pipeline transforming sentence-level HF filings into firm-year Item 1A documents.

Extract linguistic features (disclosure length, sentence complexity).

Apply FinBERT to individual sentences and aggregate sentiment at the document level.

Merge with Compustat variables and market-reaction labels (1d, 5d, 30d).

Estimate OLS models with robust (HC3) standard errors and fixed effects.

Compare FinBERT sentiment vs. text-structure metrics in predictive power.

Visualize tone–return patterns using several advanced statistical graphics.

Data Sources
HuggingFace Dataset

JanosAudran/financial-reports-sec (both small_lite and large_lite)
Contains sentence-level text from 10-K filings with item codes, sentence IDs, and labeled stock-return outcomes.

Compustat

Firm fundamentals used as controls:

Assets

Sales

Liabilities

PPE

Net income

Method Summary
1. Reconstruct & Extract

Combine sentences into full filings by (cik, filingDate).

Extract Item 1A using regex-based boundary detection.

Split Item 1A into sentences using NLTK.

2. Text-Structure Features

Number of sentences

Total word count

Average sentence length

3. FinBERT Sentiment

Apply yiyanghkust/finbert-tone to each sentence.

Aggregate document-level sentiment:

mean neg/neu/pos probability

negative-share (proportion of sentences classified as negative)

4. Empirical Framework

Winsorize all numeric variables (1%).

Construct logs for control variables.

Add year FE + state of incorporation FE.

Estimate three sets of OLS regressions for:

1-day returns

5-day returns

30-day returns

Key Findings

FinBERT negative tone is strongly negatively associated with returns across all horizons.

5-day returns show the strongest association, likely reflecting short-term investor digestion of risk disclosures.

Text-structure variables have minimal explanatory power and low economic significance.

Evidence suggests:

Markets respond more to what firms say in Item 1A than to how long the section is.

How to Run

Install dependencies

pip install -r requirements.txt


Download HuggingFace data (done automatically in code).

Run scripts in the following order:

01_download_data.py

02_extract_item1a.py

03_compute_text_metrics.py

04_run_finbert.py

05_merge_controls.py

06_regressions_and_figures.py
