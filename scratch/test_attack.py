import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import random

def test_attack():
    print("Loading data...")
    train_ratings = pd.read_csv('processed/train_ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    
    target_movieId = 1562
    
    victim_id = 2  # Target user to clone and attack

    def compute_preds(poison):
        df_copy = train_ratings.copy()
        if poison and victim_id is not None:
            # USER-ALIGNED BOT ATTACK (Perfect Cloning)
            victim_rows = df_copy[df_copy['userId'] == victim_id].copy()
            
            bot_list = []
            for bot_id in range(10000, 10500):
                bot_user = victim_rows.copy()
                bot_user['userId'] = bot_id
                bot_list.append(bot_user)
                
                # Add the target movie
                bot_list.append(pd.DataFrame([{'userId': bot_id, 'movieId': target_movieId, 'rating': 5.0}]))
                
            bots_df = pd.concat(bot_list, ignore_index=True)
            df_copy = pd.concat([df_copy, bots_df], ignore_index=True)

        R_df = df_copy.pivot(index='userId', columns='movieId', values='rating')
        users = R_df.index.tolist()
        movies = R_df.columns.tolist()

        user_ratings_mean = R_df.mean(axis=1).values
        R_demeaned = R_df.sub(user_ratings_mean, axis=0).fillna(0).values

        U, sigma, Vt = svds(R_demeaned, k=50)
        sigma = np.diag(sigma)

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, index=users, columns=movies)
        return preds_df
    
    preds_clean = compute_preds(False)
    preds_poison = compute_preds(True)
    
    if victim_id in preds_clean.index:
        u_clean = preds_clean.loc[victim_id].sort_values(ascending=False)
        rank_clean = u_clean.index.get_loc(target_movieId)
        
        u_poison = preds_poison.loc[victim_id].sort_values(ascending=False)
        rank_poison = u_poison.index.get_loc(target_movieId)
        
        print(f"\nUser {victim_id} Results:")
        print(f"Clean Rank : {rank_clean}")
        print(f"Poison Rank: {rank_poison}")
        
        print("\nTOP 5 POISON RESULTS for user:")
        print(u_poison.head(5))


test_attack()
