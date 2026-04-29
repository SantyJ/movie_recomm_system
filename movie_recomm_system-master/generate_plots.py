# =============================================================================
# generate_plots.py
# CS 550 - Massive Data Mining | Movie Recommender System
# =============================================================================
# PURPOSE:
#   Generates two side-by-side bar chart comparisons visualising the
#   performance gap between the Baseline CF and SVD algorithms across
#   all evaluation metrics. Charts are saved as high-resolution PNGs
#   for use in the project report and presentation slides.
#
# CHARTS PRODUCED:
#   1. accuracy_comparison.png  — MAE and RMSE (lower is better)
#   2. ranking_comparison.png   — Precision@10, Recall@10, F1, NDCG@10
#                                 (higher is better)
#
# RESULTS USED (from running baseline_cf.py and my_svd_approach.py):
#   Baseline CF : MAE=0.7537, RMSE=0.9765
#                 Precision=0.0010, Recall=0.0001, F1=0.0002, NDCG=0.0047
#   SVD         : MAE=0.7262, RMSE=0.9376
#                 Precision=0.1554, Recall=0.0789, F1=0.1047, NDCG=0.4643
#
# OUTPUT:
#   images/accuracy_comparison.png
#   images/ranking_comparison.png
#
# RUN:
#   python generate_plots.py
#   NOTE: Run baseline_cf.py and my_svd_approach.py first to verify results.
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import os


def generate_comparison_plots():
    """
    Produces and saves both comparison bar charts.
    All metric values are hardcoded from the experimental results obtained
    by running baseline_cf.py and my_svd_approach.py on the processed dataset.
    """

    # Create the images/ output directory if it does not already exist
    os.makedirs('images', exist_ok=True)

    # -------------------------------------------------------------------------
    # CHART 1: Rating Prediction Error (MAE and RMSE)
    # -------------------------------------------------------------------------
    # MAE  (Mean Absolute Error)       — average stars off target, lower = better
    # RMSE (Root Mean Square Error)    — penalises large misses more, lower = better
    #
    # These metrics measure how accurately each model predicts the exact
    # numeric rating a user would give a movie.
    labels          = ['MAE', 'RMSE']
    baseline_scores = [0.7537, 0.9765]   # results from baseline_cf.py
    svd_scores      = [0.7262, 0.9376]   # results from my_svd_approach.py

    # x positions for the two groups of bars (one group per metric)
    x     = np.arange(len(labels))
    width = 0.35   # width of each individual bar

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot baseline bars shifted left of centre, SVD bars shifted right
    rects1 = ax.bar(x - width/2, baseline_scores, width,
                    label='Cosine Baseline', color='#ff9999', edgecolor='black')
    rects2 = ax.bar(x + width/2, svd_scores,      width,
                    label='SVD',             color='#66b3ff', edgecolor='black')

    ax.set_ylabel('Error Score (Lower is Better)')
    ax.set_title('Rating Prediction Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.25)   # fixed y-axis so charts are easy to compare visually

    # -----------------------------------------------------------------
    # Helper: annotate each bar with its exact numeric value
    # Placed 3 points above the bar top for readability
    # -----------------------------------------------------------------
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),                  # 3pt vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('images/accuracy_comparison.png', dpi=300)   # 300 dpi for report quality
    plt.close()
    print("Generated: images/accuracy_comparison.png")

    # -------------------------------------------------------------------------
    # CHART 2: Top-10 Recommendation Ranking Quality
    # -------------------------------------------------------------------------
    # These four metrics evaluate the quality of the Top-10 recommendation
    # lists generated for each user, not individual rating predictions.
    #
    # Precision@10 — fraction of the 10 recommendations that were correct hits
    # Recall@10    — fraction of the user's test movies surfaced in the top 10
    # F1-Measure   — harmonic mean of Precision and Recall
    # NDCG@10      — ranking-aware metric; rewards hits placed at rank 1
    #                more than hits placed at rank 10
    #
    # The dramatic gap here (SVD NDCG 0.4643 vs Baseline 0.0047) is the
    # central finding of the project: sparsity destroys memory-based CF
    # for ranking even when its rating predictions are acceptable.
    labels           = ['Precision@10', 'Recall@10', 'F1-Measure', 'NDCG@10']
    baseline_ranking = [0.0010, 0.0001, 0.0002, 0.0047]   # results from baseline_cf.py
    svd_ranking      = [0.1554, 0.0789, 0.1047, 0.4643]   # results from my_svd_approach.py

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6))

    # Different colours from Chart 1 to visually distinguish the two charts
    rects1 = ax.bar(x - width/2, baseline_ranking, width,
                    label='Cosine Baseline', color='#ffcc99', edgecolor='black')
    rects2 = ax.bar(x + width/2, svd_ranking,      width,
                    label='SVD',             color='#99ff99', edgecolor='black')

    ax.set_ylabel('Score (Higher is Better)')
    ax.set_title('Top-10 Recommendation Ranking Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.55)   # fixed y-axis ceiling slightly above NDCG@10 = 0.4643

    # Reuse the same autolabel helper defined above to annotate bar values
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('images/ranking_comparison.png', dpi=300)   # 300 dpi for report quality
    plt.close()
    print("Generated: images/ranking_comparison.png")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    generate_comparison_plots()