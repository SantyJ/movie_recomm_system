import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from model import MatrixFactorization

def run_verification(processed_dir, output_file):
    test_df = pd.read_csv(os.path.join(processed_dir, 'test_ratings.csv'))
    
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
    
    # Prepare total test set for global metrics
    test_users = torch.tensor(user_encoder.transform(test_df['userId'].values), dtype=torch.long)
    test_items = torch.tensor(item_encoder.transform(test_df['movieId'].values), dtype=torch.long)
    test_actuals = test_df['rating'].values
    
    with torch.no_grad():
        test_preds = model(test_users.to(device), test_items.to(device)).cpu().numpy()
        
    global_mae = mean_absolute_error(test_actuals, test_preds)
    global_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("      RECOMMENDER SYSTEM METRICS VERIFICATION REPORT     \n")
        f.write("=========================================================\n\n")
        
        f.write("1. GLOBAL ERROR METRICS (Tested on 20% Holdout Data)\n")
        f.write("-" * 55 + "\n")
        f.write(f"Total Test Samples: {len(test_df)} user-item pairs\n")
        f.write(f"Mean Absolute Error (MAE): {global_mae:.4f} stars\n")
        f.write(f"Root Mean Square Error (RMSE): {global_rmse:.4f} stars\n\n")
        
        f.write("2. LINE-BY-LINE SAMPLE VERIFICATION (Actual vs Predicted)\n")
        f.write("-" * 55 + "\n")
        f.write("Below is a random sample of 3 users from the test set.\n")
        f.write("You can verify how close the model's prediction is to ground truth.\n\n")
        
        # Pick 3 random users that exist in our test set
        unique_test_users = test_df['userId'].unique()
        random_users = random.sample(list(unique_test_users), 3)
        
        for u in random_users:
            user_data = test_df[test_df['userId'] == u]
            f.write(f"--- User ID: {u} ---\n")
            f.write(f"| Movie ID | Actual Rating | Predicted Rating | Absolute Error |\n")
            f.write(f"|{'-'*10}|{'-'*15}|{'-'*18}|{'-'*16}|\n")
            
            # Sub-sample up to 5 movies for this user for brevity
            sample_data = user_data.head(5)
            
            u_idx = torch.tensor(user_encoder.transform(sample_data['userId'].values), dtype=torch.long)
            i_idx = torch.tensor(item_encoder.transform(sample_data['movieId'].values), dtype=torch.long)
            actuals = sample_data['rating'].values
            
            with torch.no_grad():
                preds = model(u_idx.to(device), i_idx.to(device)).cpu().numpy()
                
            for i in range(len(preds)):
                error = abs(actuals[i] - preds[i])
                f.write(f"| {sample_data.iloc[i]['movieId']:<8} | {actuals[i]:<13.1f} | {preds[i]:<16.2f} | {error:<14.2f} |\n")
            f.write("\n")
            
        f.write("=========================================================\n")
        f.write("3. MATHEMATICAL EXPLANATION OF CALCULATIONS\n")
        f.write("-" * 55 + "\n")
        f.write("=> MAE calculates:  Sum( |Actual - Predicted| ) / N\n")
        f.write("=> RMSE calculates: Sqrt( Sum( (Actual - Predicted)^2 ) / N )\n")
        f.write("Where N is the total number of items in the test set.\n")
        f.write("An MAE of ~0.8 means our predictions are off by only 0.8 stars on average.\n")

    print(f"Verification report successfully generated at: {output_file}")


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    output_file = os.path.join(project_dir, 'verification_report.txt')
    run_verification(processed_dir, output_file)
