
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import pickle
from summaries import summary
import numpy as np

app = FastAPI()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str


def preprocess_and_tokenize(description):
    tokens = word_tokenize(description.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def calculate_cosine_similarity(tfidf_matrix, query_vector):
    similarity_scores = cosine_similarity(tfidf_matrix, query_vector)
    return similarity_scores.flatten()

@app.get("/")
async def get_root():
    return 'Hello World'

@app.post("/items/")
async def create_item(item: Query):
    vectorizer = TfidfVectorizer()

    try:
        with open('vectors.pkl', 'rb') as f:
            summaries_vector = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as v:
                vectorizer = pickle.load(v)
    except FileNotFoundError:
        preprocessed_descriptions = [preprocess_and_tokenize(description) for description in summary]
        vectorizer = TfidfVectorizer()
        summaries_vector = vectorizer.fit_transform(preprocessed_descriptions)
        with open('vectors.pkl', 'wb') as f:
            pickle.dump(summaries_vector, f)
        with open('vectorizer.pkl', 'wb') as v:
            pickle.dump(vectorizer, v)

    # Compute the query vector
    query = item.query
    preprocessed_query = preprocess_and_tokenize(query)
    query_vector = vectorizer.transform([preprocessed_query])

    # Calculate similarity scores
    similarity_scores = calculate_cosine_similarity(summaries_vector, query_vector)
    top_indices = similarity_scores.argsort()[-5:][::-1]

    response = []

    for i, idx in enumerate(top_indices):
        response.append({"Index": int(idx) + 969, "Similarity Score": float(similarity_scores[idx])})
        
    return response
