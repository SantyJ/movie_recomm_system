import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_data(data_dir):
    ratings_path = os.path.join(data_dir, 'ml-latest-small', 'ratings.csv')
    
    print(f"Loading ratings from {ratings_path}...")
    df = pd.read_csv(ratings_path)
    
    # We want a strict 80/20 train-test split *per user*
    print("Performing 80/20 per-user train-test split...")
    
    # Group by userId and split each group
    train_list = []
    test_list = []
    
    for user_id, group in df.groupby('userId'):
        # If user has only 1 rating, it will error on split or go entirely to train.
        # usually movie lens users have at least 20 ratings.
        if len(group) < 2:
            train_list.append(group)
            continue
            
        train_group, test_group = train_test_split(group, test_size=0.2, random_state=42)
        train_list.append(train_group)
        test_list.append(test_group)
        
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    print(f"Total ratings: {len(df)}")
    print(f"Training ratings: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Testing ratings: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")
    
    # Create processed dir if it doesn't exist
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, 'train_ratings.csv')
    test_path = os.path.join(processed_dir, 'test_ratings.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved processed datasets to:\n{train_path}\n{test_path}")

if __name__ == "__main__":
    # Ensure working directory is the project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_data(project_dir)
