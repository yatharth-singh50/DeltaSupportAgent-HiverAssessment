import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

print("Loading raw dataset and golden set...")
raw_df = pd.read_csv('data/twcs.csv')
golden_df = pd.read_csv('data/delta_golden_set_labeled.csv')

# Isolate Delta tweets and match customer to brand responses
delta_tweets = raw_df[raw_df['author_id'] == 'Delta']
conversations = pd.merge(
    raw_df[raw_df['inbound'] == True], 
    delta_tweets, 
    left_on='tweet_id', 
    right_on='in_response_to_tweet_id', 
    suffixes=('_customer', '_delta')
)[['text_customer', 'text_delta']].dropna()

# Prevent target leakage: Remove the 200 golden set rows from our training corpus
training_corpus = conversations[~conversations['text_customer'].isin(golden_df['text_customer'])]
print(f"Training corpus size: {len(training_corpus)} conversations")

# Build the TF-IDF Vectorizer and Nearest Neighbors model
print("Fitting TF-IDF Vectorizer...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorizer.fit_transform(training_corpus['text_customer'])

print("Fitting NearestNeighbors...")
nn_model = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# Serialize the models and the reference dataframe using joblib
os.makedirs('models', exist_ok=True)
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(nn_model, 'models/nn_model.pkl')
training_corpus.reset_index(drop=True).to_pickle('models/reference_corpus.pkl')

print("Retriever successfully built and saved to models/")