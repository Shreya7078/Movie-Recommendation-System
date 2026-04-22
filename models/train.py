import pandas as pd
import numpy as np
import ast
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec
import pickle
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("Loading datasets...")
# Bollywood Dataset
df_bolly = pd.read_csv('../data/IMDB-Movie-Dataset(2023-1951).csv', index_col=0)
df_bolly.rename(columns={"movie_name": "title"}, inplace=True)
df_bolly = df_bolly[df_bolly['overview'].astype(str).str.strip() != ""].reset_index(drop=True)

# Hollywood Dataset
df_holly = pd.read_csv('../data/tmdb_5000_movies.csv')
df_holly = df_holly.dropna(subset=['overview']).reset_index(drop=True)

print("Feature Engineering...")
# Helper function to clean names (e.g., cast, director)
def clean_names(x):
    if isinstance(x, str):
        # Remove spaces so names are single tokens, replace commas with spaces to separate
        return x.lower().replace(" ", "").replace(",", " ")
    return ""

# Helper function to extract names from JSON strings (TMDB dataset)
def extract_names(obj):
    try:
        # Extract name, remove spaces to make it a single token
        return " ".join([i['name'].lower().replace(" ", "") for i in ast.literal_eval(obj)])
    except:
        return ""

# Bollywood Feature Engineering
df_bolly['overview_clean'] = df_bolly['overview'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
df_bolly['genre_clean'] = df_bolly['genre'].apply(clean_names)
df_bolly['director_clean'] = df_bolly['director'].apply(clean_names)
df_bolly['cast_clean'] = df_bolly['cast'].apply(clean_names)
df_bolly['combined_text'] = df_bolly['overview_clean'] + ' ' + df_bolly['genre_clean'] + ' ' + df_bolly['director_clean'] + ' ' + df_bolly['cast_clean']

# Hollywood Feature Engineering
df_holly['overview_clean'] = df_holly['overview'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
df_holly['genres_clean'] = df_holly['genres'].apply(extract_names)
df_holly['keywords_clean'] = df_holly['keywords'].apply(extract_names)
df_holly['prod_clean'] = df_holly['production_companies'].apply(extract_names)
df_holly['combined_text'] = df_holly['overview_clean'] + " " + df_holly['genres_clean'] + " " + df_holly['keywords_clean'] + " " + df_holly['prod_clean']

print("Training TF-IDF Models...")
tfidf_bolly = TfidfVectorizer(stop_words='english')
tfidf_matrix_bolly = tfidf_bolly.fit_transform(df_bolly['combined_text'])
knn_bolly = NearestNeighbors(metric='cosine', algorithm='brute').fit(tfidf_matrix_bolly)

tfidf_holly = TfidfVectorizer(stop_words='english')
tfidf_matrix_holly = tfidf_holly.fit_transform(df_holly['combined_text'])
knn_holly = NearestNeighbors(metric='cosine', algorithm='brute').fit(tfidf_matrix_holly)

print("Training Word2Vec Models...")
def preprocess_for_w2v(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return tokens

df_bolly['tokens'] = df_bolly['combined_text'].apply(preprocess_for_w2v)
df_holly['tokens'] = df_holly['combined_text'].apply(preprocess_for_w2v)

w2v_bolly = Word2Vec(sentences=df_bolly['tokens'], vector_size=100, window=5, min_count=1, sg=0, workers=4, epochs=100)
w2v_holly = Word2Vec(sentences=df_holly['tokens'], vector_size=100, window=5, min_count=1, sg=0, workers=4, epochs=50)

def get_avg_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

print("Generating Word2Vec Matrices...")
w2v_matrix_bolly = np.array([get_avg_vector(t, w2v_bolly) for t in df_bolly['tokens']])
w2v_matrix_holly = np.array([get_avg_vector(t, w2v_holly) for t in df_holly['tokens']])

knn_bolly_w2v = NearestNeighbors(metric='cosine', algorithm='brute').fit(w2v_matrix_bolly)
knn_holly_w2v = NearestNeighbors(metric='cosine', algorithm='brute').fit(w2v_matrix_holly)

print("Saving Models to Pickles...")
pickle.dump(tfidf_matrix_bolly, open("tfidf_bolly.pkl", "wb"))
pickle.dump(knn_bolly, open("knn_bolly.pkl", "wb"))
pickle.dump(w2v_matrix_bolly, open("w2v_matrix_bolly.pkl", "wb"))
pickle.dump(knn_bolly_w2v, open("knn_bolly_w2v.pkl", "wb"))

pickle.dump(tfidf_matrix_holly, open("tfidf_holly.pkl", "wb"))
pickle.dump(knn_holly, open("knn_holly.pkl", "wb"))
pickle.dump(w2v_matrix_holly, open("w2v_matrix_holly.pkl", "wb"))
pickle.dump(knn_holly_w2v, open("knn_holly_w2v.pkl", "wb"))

print("Done! Models have been successfully retrained and exported.")
