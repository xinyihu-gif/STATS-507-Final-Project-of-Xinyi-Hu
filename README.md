# STATS 507 Final Project — Xinyi Hu

## Overview

This repository contains all code, data-processing scripts, and empirical results for my STATS 507 final project.  
The project examines whether the tone and linguistic structure of **Item 1A “Risk Factors”** in 10-K filings help explain **short-horizon stock returns**.

Using large-scale SEC filings from **HuggingFace** and firm fundamentals from **Compustat**, the analysis compares the predictive power of:

- **FinBERT-based sentiment**, and  
- **Text-structure characteristics** (word count, sentence count, average sentence length)

---

## Research Motivation

Risk Factor disclosures are standardized and legally mandated, making them a unique window into firms’ perceived uncertainties.  
Prior research suggests that both **what firms say** and **how they say it** affect investor interpretation, but most studies rely on dictionary-based measures that ignore linguistic context.

This project applies **FinBERT**, a finance-domain transformer model, to evaluate whether modern NLP tools provide more meaningful predictive content about market reactions.

---

## Main Objectives

- Build a full pipeline transforming sentence-level HuggingFace filings into firm-year Item 1A documents  
- Extract linguistic features (disclosure length, sentence complexity)  
- Apply FinBERT to individual sentences and aggregate sentiment at the document level  
- Merge FinBERT outputs with Compustat variables and market-reaction labels (1d, 5d, 30d)  
- Estimate OLS models with robust (HC3) standard errors and fixed effects  
- Compare FinBERT sentiment vs. text-structure metrics in predictive power  
- Visualize tone–return patterns using several advanced statistical graphics  

---

## Data Sources

### HuggingFace Dataset

- `JanosAudran/financial-reports-sec` (both `small_lite` and `large_lite`)  
- Contains sentence-level text from 10-K filings with item codes, sentence IDs, and labeled stock-return outcomes  

### Compustat

Firm fundamentals used as controls:

- Assets  
- Sales  
- Liabilities  
- PPE  
- Net income  

---

## Method Summary

### 1. Reconstruct & Extract

- Combine sentences into full filings by `(cik, filingDate)`  
- Extract Item 1A using regex-based boundary detection  
- Split Item 1A into sentences using NLTK  

### 2. Text-Structure Features

- `RF1A_n_sentences`  
- `RF1A_total_words`  
- `RF1A_avg_sent_len`  

### 3. FinBERT Sentiment

- Apply `yiyanghkust/finbert-tone` to each sentence  
- Aggregate document-level sentiment:
  - Mean neg/neu/pos probability  
  - Negative-share = proportion of sentences classified as negative  

### 4. Empirical Framework

- Winsorize all numeric variables (1%)  
- Construct logs for control variables  
- Add year fixed effects and state-of-incorporation fixed effects  
- Estimate three sets of OLS regressions for:
  - 1-day returns  
  - 5-day returns  
  - 30-day returns  

---

## Key Findings

- **FinBERT negative tone is strongly negatively associated with returns** across all horizons  
- **5-day returns show the strongest association**, likely reflecting short-term investor digestion of risk disclosures  
- **Text-structure variables have minimal explanatory power** and low economic significance  

Overall, the evidence suggests that **markets respond more to what firms say in Item 1A than to how long or complex the section is**.

---

## How to Run

### 1. Install dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
