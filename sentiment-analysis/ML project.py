import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "abhi8923shriv/sentiment-analysis-dataset",
    path="test.csv",
    pandas_kwargs={"encoding": "latin1"}
)

# Handle missing values
df['text'] = df['text'].fillna('')
df['sentiment'] = df['sentiment'].fillna('neutral')

# Clean text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Convert sentiment labels to numerical values
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
df['sentiment'] = df['sentiment'].str.lower().map(sentiment_mapping)

# TF-IDF transformation
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['sentiment'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test)

# Model evaluation
print("Model Evaluation:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='macro'):.4f}")

# Function to predict user input sentiment
def predict_sentiment(text):
    clean_input = clean_text(text)
    input_features = tfidf.transform([clean_input]).toarray()
    prediction = lr_model.predict(input_features)[0]
    sentiment_labels = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return sentiment_labels[prediction]

if __name__ == "__main__":
    while True:
        user_input = input("Enter a message (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        print(f"Predicted Sentiment: {predict_sentiment(user_input)}\n")

