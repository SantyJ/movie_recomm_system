# verify_metrics.py
import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def run_verification():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    ml_dir = os.path.join(project_dir, 'ml-latest-small')

    # Load data
    train_df = pd.read_csv(os.path.join(processed_dir, 'train_ratings.csv'))
    test_df  = pd.read_csv(os.path.join(processed_dir, 'test_ratings.csv'))
    movies_df = pd.read_csv(os.path.join(ml_dir, 'movies.csv'))

    # Build and run SVD (same as my_svd_approach.py)
    R_df = train_df.pivot(index='userId', columns='movieId', values='rating')
    users  = R_df.index.tolist()
    movies = R_df.columns.tolist()

    user_ratings_mean = R_df.mean(axis=1).values
    R_demeaned = R_df.sub(user_ratings_mean, axis=0).fillna(0).values

    U, sigma, Vt = svds(R_demeaned, k=50)
    sigma = np.diag(sigma)

    all_predicted = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_predicted, index=users, columns=movies)
    preds_df = preds_df.clip(lower=0.5, upper=5.0)

    user_train_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    user_test_items  = test_df.groupby('userId')['movieId'].apply(set).to_dict()

    # Pick 3 sample users
    sample_users = [600, 528, 65]

    lines = []
    lines.append("=" * 57)
    lines.append("  RANKING METRICS VERIFICATION REPORT (Precision/Recall)")
    lines.append("=" * 57)
    lines.append("This report details how Precision@10, Recall@10, F-Measure,")
    lines.append("and NDCG@10 are calculated explicitly for 3 sample users.")

    for user_id in sample_users:
        if user_id not in preds_df.index or user_id not in user_test_items:
            continue

        test_items  = user_test_items[user_id]
        train_items = user_train_items.get(user_id, set())

        candidate_items = list(set(movies) - train_items)
        user_preds = preds_df.loc[user_id, candidate_items].sort_values(ascending=False).head(10)
        top_item_ids = user_preds.index.tolist()
        top_scores   = user_preds.values.tolist()

        hits      = [1 if item in test_items else 0 for item in top_item_ids]
        precision = sum(hits) / 10.0
        recall    = sum(hits) / len(test_items)
        f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ndcg      = ndcg_at_k(hits, 10)

        lines.append(f"\n--- USER ID: {user_id} ---")
        lines.append(f"Total Movies in User's Hidden Test Set : {len(test_items)}")
        lines.append(f"Total 'Hits' in Top-10                : {sum(hits)}")
        lines.append(f"\nUser Metrics Calculation:")
        lines.append(f"  Precision@10 = Hits / 10             = {sum(hits)} / 10 = {precision:.2f}")
        lines.append(f"  Recall@10    = Hits / TestSetSize     = {sum(hits)} / {len(test_items)} = {recall:.4f}")
        lines.append(f"  F-Measure    = 2*(P*R)/(P+R)         = {f_measure:.4f}")
        lines.append(f"  NDCG@10                              = {ndcg:.4f}")

        lines.append(f"\nTop 10 Recommendations Generated:")
        lines.append(f"{'| Rank':<7}{'| Movie ID':<12}{'| Score':<9}{'| Hit?':<8}| Movie Title")
        lines.append("|" + "-"*6 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*7 + "|" + "-"*30)

        for rank, (item_id, score, hit) in enumerate(zip(top_item_ids, top_scores, hits), 1):
            title_row = movies_df[movies_df['movieId'] == item_id]
            title = title_row.iloc[0]['title'] if not title_row.empty else "Unknown"
            hit_label = "YES" if hit else "NO"
            lines.append(f"| {rank:<5}| {item_id:<10}| {score:<7.2f}| {hit_label:<6}| {title}")

    output = "\n".join(lines)
    print(output)

    # Save to file
    output_path = os.path.join(processed_dir, 'metrics_verification_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"\nVerification report saved to: {output_path}")

if __name__ == '__main__':
    run_verification()