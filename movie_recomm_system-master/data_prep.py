# =============================================================================
# data_prep.py
# CS 550 - Massive Data Mining | Movie Recommender System
# =============================================================================
# PURPOSE:
#   This script handles all data preprocessing for the recommender system.
#   It loads the raw MovieLens ratings, performs a per-user 80/20 train/test
#   split, and saves the resulting datasets to the /processed directory for
#   use by all downstream model scripts.
#
# INPUT:
#   ml-latest-small/ratings.csv  (raw MovieLens ratings)
#
# OUTPUT:
#   processed/train_ratings.csv  (80% of each user's ratings)
#   processed/test_ratings.csv   (20% of each user's ratings, held out)
#
# RUN:
#   python data_prep.py
# =============================================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def prepare_data(data_path):
    """
    Loads raw ratings and produces a stratified per-user train/test split.

    A global random split is NOT used here because it can leave some users
    entirely in training (nothing to evaluate) or entirely in testing
    (no history to learn from). The per-user split guarantees every user
    has both a training history and held-out test items.

    Args:
        data_path (str): Absolute path to the ml-latest-small directory.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Load raw ratings
    # -------------------------------------------------------------------------
    # ratings.csv columns: userId, movieId, rating (0.5-5.0), timestamp
    ratings_path = os.path.join(data_path, 'ratings.csv')
    print(f"Loading ratings from {ratings_path}...")
    df = pd.read_csv(ratings_path)

    # -------------------------------------------------------------------------
    # STEP 2: Per-user 80/20 train/test split
    # -------------------------------------------------------------------------
    # For each user independently:
    #   - 80% of their ratings → training set  (model learns from these)
    #   - 20% of their ratings → test set      (model is evaluated on these)
    #
    # random_state=42 ensures the split is deterministic and reproducible —
    # every run of this script produces the exact same train/test partition.
    print("Performing 80/20 per-user train-test split...")

    train_list = []
    test_list  = []

    for user_id, group in df.groupby('userId'):

        # Edge case: users with only 1 rating cannot be split.
        # Their single rating goes to training so the model has some
        # information about them. In the MovieLens small dataset every
        # user has at least 20 ratings, so this guard is rarely triggered.
        if len(group) < 2:
            train_list.append(group)
            continue

        # Split this user's ratings 80% train / 20% test
        train_group, test_group = train_test_split(
            group,
            test_size=0.2,
            random_state=42   # fixed seed for reproducibility
        )
        train_list.append(train_group)
        test_list.append(test_group)

    # Concatenate all per-user splits into two final DataFrames
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df  = pd.concat(test_list).reset_index(drop=True)

    # Confirm the split sizes look correct (~80% / ~20%)
    print(f"Total ratings    : {len(df)}")
    print(f"Training ratings : {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Testing ratings  : {len(test_df)}  ({len(test_df)/len(df)*100:.2f}%)")

    # -------------------------------------------------------------------------
    # STEP 3: Save processed datasets to /processed directory
    # -------------------------------------------------------------------------
    # The processed/ folder sits one level above the ml-latest-small/ folder,
    # i.e. directly in the project root. All model scripts (baseline_cf.py,
    # my_svd_approach.py, app.py) load their data from this directory.
    processed_dir = os.path.join(data_path, '..', 'processed')
    os.makedirs(processed_dir, exist_ok=True)   # create folder if it doesn't exist

    train_path = os.path.join(processed_dir, 'train_ratings.csv')
    test_path  = os.path.join(processed_dir, 'test_ratings.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    # Print absolute paths so the user can verify exactly where files landed
    print(f"Saved processed datasets to:\n{os.path.abspath(train_path)}\n{os.path.abspath(test_path)}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # os.path.abspath(__file__) anchors the path to wherever this script lives
    # on disk, regardless of what directory the user runs it from.
    # This ensures ml-latest-small/ is always found relative to the script,
    # not relative to the current working directory.
    project_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir      = os.path.join(project_dir, 'ml-latest-small')
    prepare_data(ml_dir)