import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pickle

def train_and_evaluate_svd():
    print("Loading pre-processed data...")
    train_df = pd.read_csv('processed_data/train_ratings.csv')
    test_df = pd.read_csv('processed_data/test_ratings.csv')

    # Calculate the global mean in case we encounter a completely new user/movie in the test set
    global_mean = train_df['rating'].mean()

    print("Building the User-Item Matrix...")
    # 1. Create a pivot table where rows are Users, columns are Movies, and values are Ratings.
    # This matrix will be highly sparse (mostly NaNs) because most users haven't seen most movies.
    R_df = train_df.pivot(index='userId', columns='movieId', values='rating')

    # Save the index and columns so we can map our predictions back to actual User IDs and Movie IDs
    users = R_df.index.tolist()
    movies = R_df.columns.tolist()

    print("Performing Mean-Centering (Linear Algebra prep)...")
    # 2. Mean-Centering: We calculate the average rating for each user.
    # Why? Some users are harsh critics (rate everything a 2), some are generous (rate everything a 5).
    # Subtracting the user's mean normalizes the data.
    user_ratings_mean = R_df.mean(axis=1).values
    
    # Subtract the mean from the ratings, and fill the massive amount of NaNs with 0.
    # A 0 now means "this user would give this movie their average rating".
    R_demeaned = R_df.sub(user_ratings_mean, axis=0).fillna(0).values

    print("Running Singular Value Decomposition (SVD)...")
    # 3. The Math: Factorize the matrix into U, Sigma, and V-transpose.
    # k=50 means we are compressing all movies into 50 "latent features" (e.g., action, romance, 90s, etc.)
    # U represents how much users like these 50 features.
    # Vt represents how much movies belong to these 50 features.
    U, sigma, Vt = svds(R_demeaned, k=50)

    # svds returns sigma as a 1D array. We need to convert it into a diagonal matrix to multiply it.
    sigma = np.diag(sigma)

    print("Reconstructing the predicted matrix...")
    # 4. Multiply U * Sigma * Vt to get our predicted ratings.
    # We also add the user's mean back in to scale it back to the 1-5 star format.
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    # Put the massive numpy array back into a readable Pandas DataFrame
    preds_df = pd.DataFrame(all_user_predicted_ratings, index=users, columns=movies)
    
    # Clip the predictions to be strictly between 0.5 and 5.0 (the bounds of MovieLens)
    preds_df = preds_df.clip(lower=0.5, upper=5.0)

    print("Evaluating predictions against the Test Set...")
    # 5. Evaluate against the 20% unseen test data
    actuals = []
    predictions = []

    for _, row in test_df.iterrows():
        user = row['userId']
        movie = row['movieId']
        actual = row['rating']
        
        # Check if the user and movie existed in our training set
        if user in preds_df.index and movie in preds_df.columns:
            pred = preds_df.loc[user, movie]
        else:
            # "Cold Start" problem: If it's a totally new movie or user, guess the global average
            pred = global_mean 
            
        actuals.append(actual)
        predictions.append(pred)

    # 6. Calculate MAE and RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print("\n==================================")
    print("--- Final Evaluation Metrics ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE  (Mean Absolute Error):     {mae:.4f}")
    print("==================================\n")

    # Save the prediction matrix so we can easily generate Top-10 lists in the next step
    os.makedirs('models', exist_ok=True)
    with open('models/svd_predictions.pkl', 'wb') as f:
        pickle.dump(preds_df, f)
    print("Prediction matrix successfully saved to 'models/svd_predictions.pkl'")

if __name__ == "__main__":
    train_and_evaluate_svd()