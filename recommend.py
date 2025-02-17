import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import sys

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load Dataset
import os

def load_dataset(filepath="movies.csv"):
    return pd.read_csv(filepath, encoding="utf-8", quotechar='"', skip_blank_lines=True)



# Preprocess Text
def preprocess_text(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

# Recommend Movies
def recommend_movies(user_query, dataset, top_n=5):
    dataset["processed_description"] = dataset["Description"].apply(preprocess_text)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(dataset["processed_description"])
    
    # Vectorize the user query
    user_query_vector = tfidf.transform([preprocess_text(user_query)])
    
    # Compute Cosine Similarity
    similarities = cosine_similarity(user_query_vector, tfidf_matrix).flatten()
    
    # Get top N similar movies
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    recommendations = dataset.iloc[top_indices][["Title", "Description"]]
    
    return recommendations

if __name__ == "__main__":
    # User query from command line argument
    if len(sys.argv) < 2:
        print("Usage: python recommend.py 'Your movie preference text here'")
        sys.exit(1)
    
    user_input = sys.argv[1]
    
    movies_df = load_dataset()
    recommendations = recommend_movies(user_input, movies_df)
    
    print("\nTop Recommended Movies:")
    for index, row in recommendations.iterrows():
        print(f"{row['Title']}: {row['Description']}\n")
