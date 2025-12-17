# Movie Recommendation System

This project implements a content-based movie recommendation system using Machine Learning and Natural Language Processing techniques.  
The system recommends movies similar to a given movie by analyzing textual features such as movie overviews and genres.

The application supports both Bollywood and Hollywood movies and is deployed using a Flask web application with a dashboard-style interface.



## Features

- Content-based movie recommendations
- Supports Bollywood and Hollywood datasets
- TF-IDF based text vectorization
- K-Nearest Neighbors with cosine similarity
- Partial and fuzzy movie name matching
- Dashboard-style web interface using HTML and CSS
- Preserves user input and industry selection
- Handles invalid or misspelled movie names



## Technology Stack

- Python
- Flask
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- K-Nearest Neighbors
- HTML and CSS



## How the System Works:

1. Movie overviews and genres are cleaned and combined into a single text feature.
2. The combined text is transformed into numerical vectors using TF-IDF.
3. KNN with cosine similarity identifies the most similar movies.
4. The selected movie is excluded from the recommendation list.
5. The final recommendations are displayed through the Flask web interface.



## Project Structure:

MovieRecommendationSystem/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── data/
│   │
│   ├── tmdb_5000_movies.csv
│   │   Hollywood movie dataset
│   │
│   └── IMDB-Movie-Dataset(2023-1951).csv
│       Bollywood movie dataset
│
├── models/
│   │
|   |── knn.ipynb
|   |
│   ├── tfidf_bolly.pkl
│   │ 
│   ├── knn_bolly.pkl
│   │
│   ├── tfidf_holly.pkl
│   │
│   └── knn_holly.pkl
│
└── templates/
    │
    └── index.html


## Running the Project

1. Install the required dependencies:
   pip install -r requirements.txt

2. Run the Flask application:
   python app.py

3. Open the application in a browser:
   http://127.0.0.1:5000/



## Limitations:

- The system relies on textual similarity and does not use user ratings.
- Semantic understanding is approximated using TF-IDF, which may occasionally lead to weakly related recommendations.



## Author:

Shreya Jain  
B.Tech, Computer Science Engineering (AI and ML)
