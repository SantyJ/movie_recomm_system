import os
import torch
import pandas as pd
import numpy as np
import joblib
import random
from model import MatrixFactorization

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

def run_ranking_verification(processed_dir, ml_dir, output_file, top_n=10):
    train_df = pd.read_csv(os.path.join(processed_dir, 'train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_ratings.csv'))
    movies_df = pd.read_csv(os.path.join(ml_dir, 'movies.csv'))
    
    # Load encoders
    user_encoder = joblib.load(os.path.join(processed_dir, 'user_encoder.pkl'))
    item_encoder = joblib.load(os.path.join(processed_dir, 'item_encoder.pkl'))
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MatrixFactorization(num_users, num_items, embedding_dim=64)
    model.load_state_dict(torch.load(os.path.join(processed_dir, 'mf_model.pth'), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    user_train_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    user_test_items = test_df.groupby('userId')['movieId'].apply(set).to_dict()
    
    all_items = np.array(item_encoder.classes_)
    unique_test_users = list(user_test_items.keys())
    random_users = random.sample([u for u in unique_test_users if str(u) in [str(x) for x in user_encoder.classes_]], 3)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("    RANKING METRICS VERIFICATION REPORT (Precision/Recall)\n")
        f.write("=========================================================\n\n")
        
        f.write("This report details how Precision@10, Recall@10, F-Measure, and NDCG@10\n")
        f.write("are calculated explicitly for 3 sample users.\n\n")
        
        for u in random_users:
            test_items = user_test_items[u]
            train_items = user_train_items.get(u, set())
            candidate_items = np.array(list(set(all_items) - train_items))
            
            user_idx = user_encoder.transform([u])[0]
            item_indices = item_encoder.transform(candidate_items)
            
            u_tensor = torch.tensor([user_idx]*len(item_indices), dtype=torch.long).to(device)
            i_tensor = torch.tensor(item_indices, dtype=torch.long).to(device)
            
            with torch.no_grad():
                preds = model(u_tensor, i_tensor).cpu().numpy()
            
            top_idx = np.argsort(preds)[::-1][:top_n]
            top_item_ids = candidate_items[top_idx]
            top_predictions = preds[top_idx]
            
            hits = [1 if item in test_items else 0 for item in top_item_ids]
            
            precision = sum(hits) / top_n
            recall = sum(hits) / len(test_items)
            f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            ndcg = ndcg_at_k(hits, top_n)
            
            f.write(f"--- USER ID: {u} ---\n")
            f.write(f"Total Movies in User's Hidden Test Set: {len(test_items)}\n")
            f.write(f"Total 'Hits' in Top-10: {sum(hits)}\n\n")
            
            f.write(f"User Metrics Calculation:\n")
            f.write(f"  Precision@10 = Hits / 10 = {sum(hits)} / 10 = {precision:.2f}\n")
            f.write(f"  Recall@10    = Hits / TestSetSize = {sum(hits)} / {len(test_items)} = {recall:.4f}\n")
            f.write(f"  F-Measure    = 2 * (P * R) / (P + R) = {f_measure:.4f}\n")
            f.write(f"  NDCG@10      = {ndcg:.4f}\n\n")
            
            f.write("Top 10 Recommendations Generated:\n")
            f.write(f"| Rank | Movie ID | Score | Hit? | Movie Title \n")
            f.write(f"|------|----------|-------|------|{'-'*30}\n")
            
            for rank, (item_id, score, is_hit) in enumerate(zip(top_item_ids, top_predictions, hits)):
                movie_name_df = movies_df[movies_df['movieId'] == item_id]
                m_name = movie_name_df.iloc[0]['title'] if not movie_name_df.empty else "Unknown"
                hit_str = "YES" if is_hit else "NO"
                f.write(f"| {rank+1:<4} | {item_id:<8} | {score:<5.2f} | {hit_str:<4} | {m_name}\n")
            f.write("\n")
            
    print(f"Ranking verification report generated at: {output_file}")


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    ml_dir = os.path.join(project_dir, 'ml-latest-small')
    output_file = os.path.join(project_dir, 'ranking_verification_report.txt')
    run_ranking_verification(processed_dir, ml_dir, output_file)
