# CS 550 - Movie Recommender System
**Dataset:** MovieLens Small (ml-latest-small)  
**Algorithms:** User-User Collaborative Filtering (Baseline) + SVD Matrix Factorization

---

## Project Structure

---

## Setup

**1. Download the dataset**  
Download `ml-latest-small` from https://grouplens.org/datasets/movielens/latest/  
Place the unzipped folder in the project root directory.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## How to Run (in order)

**Step 1 — Data Preprocessing**
```bash
python data_prep.py
```
Splits ratings 80/20 per user. Saves train/test CSVs to `processed/`.

**Step 2 — Baseline CF Evaluation**
```bash
python baseline_cf.py
```
Runs User-User Cosine Similarity CF. Prints MAE, RMSE, Precision@10, Recall@10, F1, NDCG@10.

**Step 3 — SVD Evaluation**
```bash
python my_svd_approach.py
```
Runs SVD Matrix Factorization (k=50). Prints the same evaluation metrics.

**Step 3b — Metrics Verification (Optional)**
python verify_metrics.py
Generates a per-user breakdown of Top-10 metrics for 3 sample users.
Saved to processed/metrics_verification_report.txt

**Step 4 — Generate Comparison Plots**
```bash
python generate_plots.py
```
Saves two bar chart PNGs to `images/`.

**Step 5 — python generate_report.py**
```bash
python generate_report.py
```

**Step 6 — python verify_metrics.py**
```bash
python verify_metrics.py
```

**Step 7 — Run the Demo App**
```bash
streamlit run app.py
```
Opens interactive recommender in your browser with Controllability and Robustness modules.

---

## Results Summary

| Metric | Baseline CF | SVD |
|---|---|---|
| MAE | 0.7537 | 0.7262 |
| RMSE | 0.9765 | 0.9376 |
| Precision@10 | 0.0010 | 0.1554 |
| Recall@10 | 0.0001 | 0.0789 |
| F1-Measure | 0.0002 | 0.1047 |
| NDCG@10 | 0.0047 | 0.4643 |

---

## Optional Trustworthiness Tasks (in Demo)
- **Option A — Explainability:** Explained why certain movie is being recommended
- **Option C — Controllability:** Exclude genres in real-time via sidebar
- **Option E — Robustness:** Simulate and defend against a shilling/data poisoning attack