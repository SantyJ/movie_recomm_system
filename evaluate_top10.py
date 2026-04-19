import os
import torch
import pandas as pd
import numpy as np
import joblib
from model import MatrixFactorization
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

def evaluate_top_n(processed_dir, top_n=10):
    train_df = pd.read_csv(os.path.join(processed_dir, 'train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_ratings.csv'))
    
    # Load encoders
    user_encoder = joblib.dump(None, 'temp.user_encoder.pkl') if False else joblib.load(os.path.join(processed_dir, 'user_encoder.pkl'))
    item_encoder = joblib.load(os.path.join(processed_dir, 'item_encoder.pkl'))
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MatrixFactorization(num_users, num_items, embedding_dim=64)
    model.load_state_dict(torch.load(os.path.join(processed_dir, 'mf_model.pth'), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    user_train_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    user_test_items = test_df.groupby('userId')['movieId'].apply(set).to_dict()
    
    all_items = np.array(item_encoder.classes_)
    
    precisions = []
    recalls = []
    ndcgs = []
    
    # Using batches to speed up evaluation
    print(f"Evaluating top-{top_n} recommendation for {len(user_test_items)} users...")
    
    for user_id, test_items in tqdm(user_test_items.items()):
        if str(user_id) not in [str(x) for x in user_encoder.classes_]:
            continue
            
        train_items = user_train_items.get(user_id, set())
        
        # Candidate items (not in train)
        candidate_items = np.array(list(set(all_items) - train_items))
        if len(candidate_items) == 0:
            continue
            
        user_idx = user_encoder.transform([user_id])[0]
        item_indices = item_encoder.transform(candidate_items)
        
        # Prepare tensors
        u_tensor = torch.tensor([user_idx] * len(item_indices), dtype=torch.long).to(device)
        i_tensor = torch.tensor(item_indices, dtype=torch.long).to(device)
        
        with torch.no_grad():
            preds = model(u_tensor, i_tensor).cpu().numpy()
            
        # Top-N recommendations
        top_indices = np.argsort(preds)[::-1][:top_n]
        top_item_ids = candidate_items[top_indices]
        
        # Calculate metrics
        hits = [1 if item in test_items else 0 for item in top_item_ids]
        
        precision = sum(hits) / top_n
        recall = sum(hits) / len(test_items)
        
        precisions.append(precision)
        recalls.append(recall)
        
        # For NDCG, test items relevance is 1, else 0
        ndcg = ndcg_at_k(hits, top_n)
        ndcgs.append(ndcg)
        
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_ndcg = np.mean(ndcgs)
    
    print("\n========= TOP-10 RECOMMENDATION METRICS =========")
    print(f"Precision@10 : {avg_precision:.4f}")
    print(f"Recall@10    : {avg_recall:.4f}")
    print(f"F1-Measure   : {avg_f1:.4f}")
    print(f"NDCG@10      : {avg_ndcg:.4f}")
    print("=================================================")
    
    # Save the global metrics to a text file
    output_path = os.path.join(processed_dir, 'global_evaluation_metrics.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("========= GLOBAL TOP-10 RECOMMENDATION METRICS =========\n")
        f.write("These metrics are averaged across all users in the dataset.\n\n")
        f.write(f"Average Precision@10 : {avg_precision:.4f}\n")
        f.write(f"Average Recall@10    : {avg_recall:.4f}\n")
        f.write(f"Average F1-Measure   : {avg_f1:.4f}\n")
        f.write(f"Average NDCG@10      : {avg_ndcg:.4f}\n")
        f.write("========================================================\n")
    print(f"Global metrics successfully written to: {output_path}")

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    evaluate_top_n(processed_dir, top_n=10)
