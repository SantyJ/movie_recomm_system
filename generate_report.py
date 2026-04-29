# generate_report.py
import os

def generate_comparison_report():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')

    # ── Results (update these if you re-run experiments) ──
    baseline_mae       = 0.7537
    baseline_rmse      = 0.9765
    baseline_precision = 0.0010
    baseline_recall    = 0.0001
    baseline_f1        = 0.0002
    baseline_ndcg      = 0.0047

    svd_mae       = 0.7262
    svd_rmse      = 0.9376
    svd_precision = 0.1554
    svd_recall    = 0.0789
    svd_f1        = 0.1047
    svd_ndcg      = 0.4643

    report = f"""=========================================================
      RECOMMENDER SYSTEM: MODEL COMPARISON REPORT
=========================================================
This file provides a direct 2-way comparative analysis between:
1. Baseline (Classical Memory-Based Collaborative Filtering)
2. SVD (Advanced Mathematical Model-Based Matrix Factorization)

DATASET CONFIGURATION:
- Dataset       : MovieLens Small (ml-latest-small)
- Total Ratings : 100,836
- Total Users   : 610
- Total Movies  : 9,742
- Training Set  : 80% per-user split
- Testing Set   : 20% holdout per-user split
- Candidate Pool (for rankings): ~9,700 items

---------------------------------------------------------
1. OVERALL PREDICTION ERROR (Accuracy on 1-5 Scale)
---------------------------------------------------------
Metric Definitions:
- MAE  : Mean Absolute Error (average stars off target)
- RMSE : Root Mean Square Error (heavily penalizes large misses)

[ Baseline: Cosine Similarity CF ]
- MAE:  {baseline_mae:.4f}
- RMSE: {baseline_rmse:.4f}

[ SciPy SVD (k=50) ]
- MAE:  {svd_mae:.4f}
- RMSE: {svd_rmse:.4f}

* Analysis: SVD outperforms the baseline on both error metrics.
  MAE improvement  : {((baseline_mae  - svd_mae)  / baseline_mae  * 100):.1f}%
  RMSE improvement : {((baseline_rmse - svd_rmse) / baseline_rmse * 100):.1f}%

---------------------------------------------------------
2. TOP-10 RECOMMENDATION METRICS (Ranking Quality)
---------------------------------------------------------
Metric Definitions:
- Precision@10 : Hit-rate inside the top-10 recommendation list.
- Recall@10    : Fraction of user's ground-truth test items found in top-10.
- F1-Measure   : Harmonic mean of Precision and Recall.
- NDCG@10      : Normalized Discounted Cumulative Gain — rewards hits
                 ranked higher (slot #1 vs slot #10).

[ Baseline: Cosine Similarity CF ]
- Precision@10 : {baseline_precision:.4f}  ({baseline_precision*100:.1f}%)
- Recall@10    : {baseline_recall:.4f}
- F1-Measure   : {baseline_f1:.4f}
- NDCG@10      : {baseline_ndcg:.4f}

[ SciPy SVD (k=50) ]
- Precision@10 : {svd_precision:.4f}  ({svd_precision*100:.1f}%)
- Recall@10    : {svd_recall:.4f}
- F1-Measure   : {svd_f1:.4f}
- NDCG@10      : {svd_ndcg:.4f}

* Analysis: SVD dominates ranking quality across all four metrics.
  Precision improvement : {((svd_precision - baseline_precision) / baseline_precision * 100):.0f}x
  NDCG improvement      : {((svd_ndcg      - baseline_ndcg)      / baseline_ndcg      * 100):.0f}x

=========================================================
CONCLUSION / TAKEAWAYS
=========================================================
The traditional Memory-Based Collaborative Filtering algorithm
(Cosine Similarity) produced reasonable explicit rating predictions
(MAE: {baseline_mae:.4f}) but nearly completely failed the ranking
evaluation (NDCG: {baseline_ndcg:.4f}). This is a well-known consequence
of data sparsity: the user-item matrix is ~98% empty, so cosine
similarity scores become unreliable and the model cannot distinguish
strong candidates from weak ones when building a ranked list.

SVD-based Matrix Factorization addresses this directly. By
mean-centering the rating matrix and decomposing it into k=50
latent factors (U, Sigma, Vt), SVD compresses the sparse observed
ratings into a dense low-rank approximation that generalizes to
unseen user-item pairs. This produces a substantially better
ranking signal, achieving Precision@10 of {svd_precision:.4f} and
NDCG@10 of {svd_ndcg:.4f} — a ~{((svd_ndcg - baseline_ndcg) / baseline_ndcg * 100):.0f}x improvement over the baseline.

=========================================================
"""

    output_path = os.path.join(processed_dir, 'model_comparison_results.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"Report saved to: {output_path}")

if __name__ == '__main__':
    generate_comparison_report()