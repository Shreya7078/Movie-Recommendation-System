from flask import Flask, render_template, request
import pandas as pd
import pickle
from difflib import get_close_matches

app = Flask(__name__)



df_bolly = pd.read_csv("data/IMDB-Movie-Dataset(2023-1951).csv")
df_holly = pd.read_csv("data/tmdb_5000_movies.csv")

df_bolly.rename(columns={"movie_name": "title"}, inplace=True)



tfidf_matrix_bolly = pickle.load(open("models/tfidf_bolly.pkl", "rb"))
knn_bolly = pickle.load(open("models/knn_bolly.pkl", "rb"))

tfidf_matrix_holly = pickle.load(open("models/tfidf_holly.pkl", "rb"))
knn_holly = pickle.load(open("models/knn_holly.pkl", "rb"))



def find_movie_name(movie_name, df):
    movie_name = movie_name.lower()
    titles = df["title"].str.lower().tolist()

    if movie_name in titles:
        return movie_name

    partial = df[df["title"].str.lower().str.contains(movie_name)]
    if not partial.empty:
        return partial["title"].iloc[0].lower()

    match = get_close_matches(movie_name, titles, n=1, cutoff=0.4)
    if match:
        return match[0]

    return None


def recommend(movie_name, df, tfidf_matrix, knn, top_n=8):
    corrected = find_movie_name(movie_name, df)

    if corrected is None:
        return None, []

    idx = df[df["title"].str.lower() == corrected].index[0]
    movie_vector = tfidf_matrix[idx]

    distances, indices = knn.kneighbors(
        movie_vector,
        n_neighbors=top_n + 1
    )

    similar_indices = indices[0][1:]
    recommendations = df["title"].iloc[similar_indices].tolist()

    return corrected.title(), recommendations



@app.route("/", methods=["GET", "POST"])
def index():

    recommendations = []
    error = None
    searched_movie = None
    user_movie = ""
    selected_industry = ""

    if request.method == "POST":

        movie = request.form.get("movie")
        industry = request.form.get("industry")

        user_movie = movie
        selected_industry = industry


        if not movie or not industry:
            error = "❌ Please enter movie name and select industry."
            return render_template(
                "index.html",
                recommendations=recommendations,
                error=error,
                searched_movie=searched_movie,
                user_movie=user_movie,
                selected_industry=selected_industry
            )

        if industry == "bollywood":
            searched_movie, recommendations = recommend(
                movie,
                df_bolly,
                tfidf_matrix_bolly,
                knn_bolly,
                top_n=8
            )
        elif industry == "hollywood":
            searched_movie, recommendations = recommend(
                movie,
                df_holly,
                tfidf_matrix_holly,
                knn_holly,
                top_n=8
            )

        if searched_movie is None:
            error = "❌ Movie not found. Please check spelling or try another movie."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error,
        searched_movie=searched_movie,
        user_movie=user_movie,
        selected_industry=selected_industry
    )



if __name__ == "__main__":
    app.run()
