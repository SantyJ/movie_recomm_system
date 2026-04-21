import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def run_custom_svd():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    ml_dir = os.path.join(project_dir, 'ml-latest-small')

    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(processed_dir, 'train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_ratings.csv'))

    # Build matrix
    print("Building User-Item Pivot Matrix...")
    R_df = train_df.pivot(index='userId', columns='movieId', values='rating')
    users = R_df.index.tolist()
    movies = R_df.columns.tolist()

    # Mean center
    print("Mean Centering...")
    user_ratings_mean = R_df.mean(axis=1).values
    R_demeaned = R_df.sub(user_ratings_mean, axis=0).fillna(0).values

    # Run SVD
    k_components = 50
    print(f"Running SVD with k={k_components}...")
    U, sigma, Vt = svds(R_demeaned, k=k_components)
    sigma = np.diag(sigma)

    print("Reconstructing predicted ratings...")
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, index=users, columns=movies)
    preds_df = preds_df.clip(lower=0.5, upper=5.0)

    # 1. Evaluate MAE & RMSE on test set
    print("\n--- Evaluating Rating Accuracy (MAE/RMSE) ---")
    actuals = []
    predictions = []
    global_mean = train_df['rating'].mean()

    for _, row in test_df.iterrows():
        user = row['userId']
        movie = row['movieId']
        if user in preds_df.index and movie in preds_df.columns:
            pred = preds_df.loc[user, movie]
        else:
            pred = global_mean
        actuals.append(row['rating'])
        predictions.append(pred)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"SVD MAE:  {mae:.4f}")
    print(f"SVD RMSE: {rmse:.4f}")

    # 2. Evaluate Ranking Metrics (Precision, Recall, NDCG)
    print("\n--- Evaluating Top-10 Ranking Metrics ---")
    user_train_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    user_test_items = test_df.groupby('userId')['movieId'].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []
    all_items = set(movies)
    
    for user_id, test_items in tqdm(user_test_items.items()):
        if user_id not in preds_df.index:
            continue
            
        train_items = user_train_items.get(user_id, set())
        candidate_items = list(all_items - train_items)
        
        if not candidate_items:
            continue
            
        # Get SVD predictions for unseen candidate items
        user_preds = preds_df.loc[user_id, candidate_items]
        
        # Sort top 10
        top_candidates = user_preds.sort_values(ascending=False).head(10)
        top_item_ids = top_candidates.index.tolist()
        
        # Hits calculation
        hits = [1 if item in test_items else 0 for item in top_item_ids]
        
        precision = sum(hits) / 10.0
        recall = sum(hits) / len(test_items)
        ndcg = ndcg_at_k(hits, 10)
        
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_ndcg = np.mean(ndcgs)

    print("\n========= SVD TOP-10 RECOMMENDATION METRICS =========")
    print(f"Precision@10 : {avg_precision:.4f}")
    print(f"Recall@10    : {avg_recall:.4f}")
    print(f"F1-Measure   : {avg_f1:.4f}")
    print(f"NDCG@10      : {avg_ndcg:.4f}")
    print("=====================================================")

if __name__ == '__main__':
    run_custom_svd()
