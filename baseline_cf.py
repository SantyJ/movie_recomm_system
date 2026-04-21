import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

def dcg_at_k(r, k):
    r = np.asarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def run_baseline_cf():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    ml_dir = os.path.join(project_dir, 'ml-latest-small')

    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(processed_dir, 'train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_ratings.csv'))

    print("Building User-Item Pivot Matrix...")
    R_df = train_df.pivot(index='userId', columns='movieId', values='rating')
    users = R_df.index.tolist()
    movies = R_df.columns.tolist()

    print("Calculating User-User Cosine Similarity...")
    # Fill NaN with 0 for Cosine Similarity (Memory-based CF standard practice)
    R_filled = R_df.fillna(0)
    user_similarity = cosine_similarity(R_filled)
    user_similarity_df = pd.DataFrame(user_similarity, index=users, columns=users)

    # 1. Evaluate MAE & RMSE on the test set
    print("\n--- Evaluating Rating Accuracy (MAE/RMSE) ---")
    actuals = []
    predictions = []
    global_mean = train_df['rating'].mean()
    
    # Pre-calculate user means to use for adjusting predictions
    user_means = R_df.mean(axis=1)

    print("Predicting test ratings...")
    for _, row in test_df.iterrows():
        u = row['userId']
        i = row['movieId']
        actual = row['rating']
        
        if u in user_similarity_df.index and i in R_df.columns:
            # Get similarities for user u
            sims = user_similarity_df.loc[u].copy()
            # Find users who rated movie i
            rated_by = R_df[i].dropna()
            
            # Keep only similarities for users who rated movie i
            sim_scores = sims.loc[rated_by.index]
            
            # Standard neighborhood-based prediction
            if len(sim_scores) > 0 and sim_scores.sum() > 0:
                weighted_sum = np.dot(sim_scores, rated_by)
                pred = weighted_sum / sim_scores.sum()
            else:
                pred = user_means[u] if pd.notna(user_means[u]) else global_mean
        else:
            pred = global_mean
            
        # Clip
        pred = max(0.5, min(5.0, pred))
        
        actuals.append(actual)
        predictions.append(pred)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"Baseline CF MAE:  {mae:.4f}")
    print(f"Baseline CF RMSE: {rmse:.4f}")

    # 2. Evaluate Ranking Metrics (Precision, Recall, NDCG)
    print("\n--- Evaluating Top-10 Ranking Metrics ---")
    user_train_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    user_test_items = test_df.groupby('userId')['movieId'].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []
    all_items = set(movies)
    
    print("Generating candidate recommendations...")
    for user_id, test_items in tqdm(user_test_items.items()):
        if user_id not in user_similarity_df.index:
            continue
            
        train_items = user_train_items.get(user_id, set())
        candidate_items = list(all_items - train_items)
        
        if not candidate_items:
            continue
            
        # Optimize candidate prediction by taking dot product of user similarities and the entire rating dataframe
        sims = user_similarity_df.loc[user_id].values
        
        # We only want to predict for candidate_items
        # ratings matrix shape: (num_users, num_candidate_items)
        ratings_candidate = R_filled[candidate_items].values
        
        # sum of similarities for users who rated the movie
        # Binary mask indicating if user rated the movie
        rated_mask = (ratings_candidate > 0).astype(float)
        sim_sums = sims.dot(rated_mask)
        
        # Weighted ratings
        weighted_ratings = sims.dot(ratings_candidate)
        
        # Avoid division by zero
        pred_ratings = np.zeros_like(weighted_ratings)
        idx = sim_sums > 0
        pred_ratings[idx] = weighted_ratings[idx] / sim_sums[idx]
        pred_ratings[~idx] = user_means[user_id] if pd.notna(user_means[user_id]) else global_mean
        
        # Sort top 10
        top_indices = np.argsort(pred_ratings)[::-1][:10]
        top_item_ids = np.array(candidate_items)[top_indices]
        
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

    print("\n========= BASELINE CF RECOMMENDATION METRICS =========")
    print(f"Precision@10 : {avg_precision:.4f}")
    print(f"Recall@10    : {avg_recall:.4f}")
    print(f"F1-Measure   : {avg_f1:.4f}")
    print(f"NDCG@10      : {avg_ndcg:.4f}")
    print("======================================================")

if __name__ == '__main__':
    run_baseline_cf()
