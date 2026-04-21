import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from scipy.sparse.linalg import svds
import random

# Custom CSS for aesthetics
st.set_page_config(page_title="AI Recommender Demo", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #0d1117; color: #c9d1d9;}
    h1, h2, h3 {color: #58a6ff;}
    .stSelectbox label {color: #8b949e;}
    .movie-card {
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 15px; 
        border: 1px solid #30363d;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.02);
        border: 1px solid #58a6ff;
    }
    .movie-title {font-size: 1.2rem; font-weight: bold; color: #58a6ff; margin-bottom: 5px;}
    .movie-genre {font-size: 0.9rem; color: #8b949e; margin-bottom: 10px;}
    .explanation {
        font-size: 0.95rem; 
        color: #e6edf3; 
        padding: 10px; 
        background: rgba(88, 166, 255, 0.1); 
        border-left: 4px solid #58a6ff;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_dir, 'processed')
    ml_small_dir = os.path.join(project_dir, 'ml-latest-small')
    
    user_encoder = joblib.load(os.path.join(processed_dir, 'user_encoder.pkl'))
    
    movies_df = pd.read_csv(os.path.join(ml_small_dir, 'movies.csv'))
    train_ratings = pd.read_csv(os.path.join(processed_dir, 'train_ratings.csv'))
        
    return train_ratings, movies_df, user_encoder

@st.cache_data(show_spinner=False)
def compute_svd(ratings_df):
    df_copy = ratings_df.copy()
    
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


def generate_explanation(user_history, rec_movie_id, movies_df):
    rec_movie_row = movies_df[movies_df['movieId'] == rec_movie_id]
    if rec_movie_row.empty:
        return "Because it is globally popular."
        
    rec_genres = set(rec_movie_row.iloc[0]['genres'].split('|'))
    
    high_rated_history = user_history[user_history['rating'] >= 4.0]
    if high_rated_history.empty:
        return "Based on your general viewing patterns."
        
    hist_movies = pd.merge(high_rated_history, movies_df, on='movieId')
    
    best_match = None
    max_overlap = 0
    
    for _, row in hist_movies.iterrows():
        hist_genres = set(row['genres'].split('|'))
        overlap = len(rec_genres.intersection(hist_genres))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = row['title']
            
    if best_match and max_overlap > 0:
        templates = [
            f"We recommend this because you highly rated structurally similar movies like **{best_match}**.",
            f"Because you enjoyed **{best_match}**, our AI found strong thematic overlaps here.",
            f"This shares similar genre DNA with **{best_match}**, which you previously rated highly.",
            f"Fans of **{best_match}** frequently cluster into this recommendation space."
        ]
        return random.choice(templates)
    else:
        return "Our Model-Based SVD AI found latent collaborative similarities with your profile!"

# Load core data
with st.spinner("Initializing Database..."):
    train_ratings, movies_df, user_encoder = load_data()


st.title("🎬 Movie Recommender AI System")
st.markdown("CS 550 Project Demo - SVD Matrix Factorization & Trustworthiness")

# ---------------------------------------------
# TRUSTWORTHINESS MODULES (SIDEBAR)
# ---------------------------------------------
st.sidebar.title("Trustworthiness Tools")

# MODULE C: CONTROLLABILITY
st.sidebar.subheader("🎛️ Option C: Controllability")
st.sidebar.markdown("Steer your algorithm in real-time by explicitly forbidding genre dimensions.")
all_genres = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
excluded_genres = st.sidebar.multiselect("Exclude Genres:", all_genres)


# ---------------------------------------------
# MAIN UI
# ---------------------------------------------
valid_users = train_ratings['userId'].unique().tolist()
all_users = sorted(valid_users)
selected_user = st.selectbox("👤 Select a User ID to view personalized recommendations:", all_users)

if selected_user:
    st.markdown("---")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("📚 User's Top Rated History")
        user_hist = train_ratings[train_ratings['userId'] == selected_user]
        user_hist_merged = pd.merge(user_hist, movies_df, on='movieId').sort_values('rating', ascending=False).head(10)
        
        for _, row in user_hist_merged.iterrows():
            st.markdown(f"⭐ **{row['rating']}** | {row['title']} *( {row['genres']} )*")
            
    with col2:
        st.subheader("✨ SVD Personalized Rankings (Top 10)")
        
        with st.spinner("Calculating SVD Mathematical Tensor (Live)..."):
            preds_df = compute_svd(train_ratings)
            
        # Isolate User
        if selected_user in preds_df.index:
            user_preds = preds_df.loc[selected_user].copy()
            
            # Filter already watched
            watched_items = set(user_hist['movieId'].unique())
            candidate_items = list(set(preds_df.columns) - watched_items)
            
            # Apply Option C (Controllability)
            if excluded_genres:
                exclude_ids = set()
                for gen in excluded_genres:
                    # Find all movies containing the excluded genre
                    mask = movies_df['genres'].str.contains(gen, na=False)
                    exclude_ids.update(movies_df[mask]['movieId'].tolist())
                candidate_items = [i for i in candidate_items if i not in exclude_ids]
                
            if len(candidate_items) > 0:
                user_preds = user_preds[candidate_items].sort_values(ascending=False).head(10)
                top_items = user_preds.index.tolist()
                
                for item_id in top_items:
                    m_row = movies_df[movies_df['movieId'] == item_id]
                    if not m_row.empty:
                        title = m_row.iloc[0]['title']
                        genres = m_row.iloc[0]['genres'].replace('|', ', ')
                        explain_text = generate_explanation(user_hist, item_id, movies_df)
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">{title}</div>
                            <div class="movie-genre">{genres}</div>
                            <div class="explanation">💡 <b>Why?</b> {explain_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
