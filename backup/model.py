import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, rating_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rating_tensor = rating_tensor
        
    def __len__(self):
        return len(self.user_tensor)
        
    def __getitem__(self, idx):
        return self.user_tensor[idx], self.item_tensor[idx], self.rating_tensor[idx]

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize embeddings for better convergence
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        u_bias = self.user_bias(user_indices).squeeze()
        i_bias = self.item_bias(item_indices).squeeze()
        
        dot_product = (user_emb * item_emb).sum(1)
        prediction = dot_product + u_bias + i_bias + self.global_bias
        return prediction

def train_and_eval(processed_dir):
    train_df = pd.read_csv(os.path.join(processed_dir, 'train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_ratings.csv'))
    
    # Combine to fit LabelEncoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    all_users = pd.concat([train_df['userId'], test_df['userId']]).unique()
    all_items = pd.concat([train_df['movieId'], test_df['movieId']]).unique()
    
    user_encoder.fit(all_users)
    item_encoder.fit(all_items)
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    # Save encoders for later inference
    joblib.dump(user_encoder, os.path.join(processed_dir, 'user_encoder.pkl'))
    joblib.dump(item_encoder, os.path.join(processed_dir, 'item_encoder.pkl'))
    
    train_users = torch.tensor(user_encoder.transform(train_df['userId'].values), dtype=torch.long)
    train_items = torch.tensor(item_encoder.transform(train_df['movieId'].values), dtype=torch.long)
    train_ratings = torch.tensor(train_df['rating'].values, dtype=torch.float32)
    
    test_users = torch.tensor(user_encoder.transform(test_df['userId'].values), dtype=torch.long)
    test_items = torch.tensor(item_encoder.transform(test_df['movieId'].values), dtype=torch.long)
    test_ratings = torch.tensor(test_df['rating'].values, dtype=torch.float32)
    
    train_dataset = RatingDataset(train_users, train_items, train_ratings)
    test_dataset = RatingDataset(test_users, test_items, test_ratings)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MatrixFactorization(num_users, num_items, embedding_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            predictions = model(u, i)
            loss = criterion(predictions, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(u)
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_dataset):.4f}")
        
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for u, i, r in test_loader:
            u, i = u.to(device), i.to(device)
            predictions = model(u, i).cpu().numpy()
            all_preds.extend(predictions)
            all_targets.extend(r.numpy())
            
    # Calculate MAE and RMSE
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    print("\n--- Evaluation on Test Set ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save the model
    model_path = os.path.join(processed_dir, 'mf_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    train_and_eval(processed_dir)
