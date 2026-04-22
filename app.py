from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from difflib import get_close_matches

app = Flask(__name__)

df_bolly = pd.read_csv("data/IMDB-Movie-Dataset(2023-1951).csv", index_col=0)
df_bolly.rename(columns={"movie_name": "title"}, inplace=True)
df_bolly = df_bolly[df_bolly['overview'].astype(str).str.strip() != ""].reset_index(drop=True)

df_holly = pd.read_csv("data/tmdb_5000_movies.csv")
df_holly = df_holly.dropna(subset=['overview']).reset_index(drop=True)

tf_matrix_bolly = pickle.load(open("models/tfidf_bolly.pkl", "rb"))
knn_bolly_tf = pickle.load(open("models/knn_bolly.pkl", "rb"))
w2v_matrix_bolly = pickle.load(open("models/w2v_matrix_bolly.pkl", "rb"))
knn_bolly_w2v = pickle.load(open("models/knn_bolly_w2v.pkl", "rb"))

tf_matrix_holly = pickle.load(open("models/tfidf_holly.pkl", "rb"))
knn_holly_tf = pickle.load(open("models/knn_holly.pkl", "rb"))
w2v_matrix_holly = pickle.load(open("models/w2v_matrix_holly.pkl", "rb"))
knn_holly_w2v = pickle.load(open("models/knn_holly_w2v.pkl", "rb"))

def find_movie_name(movie_name, df):
    movie_name = movie_name.lower()
    titles = df["title"].str.lower().tolist()
    if movie_name in titles: return movie_name
    partial = df[df["title"].str.lower().str.contains(movie_name)]
    if not partial.empty: return partial["title"].iloc[0].lower()
    match = get_close_matches(movie_name, titles, n=1, cutoff=0.4)
    return match[0] if match else None

def recommend_hybrid(movie_name, df, tf_matrix, tf_knn, w2v_matrix, w2v_knn):
    corrected = find_movie_name(movie_name, df)
    if not corrected: return None, []
    idx = df[df["title"].str.lower() == corrected].index[0]
    
    # Get top 30 neighbors from both to have a good pool for scoring
    # We use n_neighbors=31 because the first match is the movie itself
    _, ind_tf = tf_knn.kneighbors(tf_matrix[idx], n_neighbors=31)
    _, ind_w2v = w2v_knn.kneighbors(w2v_matrix[idx].reshape(1, -1), n_neighbors=31)
    
    tf_list = ind_tf[0][1:].tolist()
    w2v_list = ind_w2v[0][1:].tolist()
    
    scores = {}
    
    # Assign points based on rank. Rank 1 gets 30 points, Rank 30 gets 1 point.
    for rank, movie_idx in enumerate(tf_list):
        scores[movie_idx] = scores.get(movie_idx, 0) + (30 - rank)
        
    for rank, movie_idx in enumerate(w2v_list):
        scores[movie_idx] = scores.get(movie_idx, 0) + (30 - rank)
        
    # Sort by score in descending order
    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 8 unique indices
    final_indices = [movie_idx for movie_idx, score in sorted_movies[:8]]
            
    return corrected.title(), df["title"].iloc[final_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations, error, searched_movie = [], None, None
    user_movie, selected_industry = "", ""
    if request.method == "POST":
        movie = request.form.get("movie")
        industry = request.form.get("industry")
        user_movie, selected_industry = movie, industry
        if not movie or not industry:
            error = "❌ Please enter movie name and select industry."
        else:
            if industry == "bollywood":
                searched_movie, recommendations = recommend_hybrid(movie, df_bolly, tf_matrix_bolly, knn_bolly_tf, w2v_matrix_bolly, knn_bolly_w2v)
            else:
                searched_movie, recommendations = recommend_hybrid(movie, df_holly, tf_matrix_holly, knn_holly_tf, w2v_matrix_holly, knn_holly_w2v)
            if not searched_movie: error = "❌ Movie not found."
    return render_template("index.html", recommendations=recommendations, error=error, searched_movie=searched_movie, user_movie=user_movie, selected_industry=selected_industry)

if __name__ == "__main__":
    app.run(debug=True)