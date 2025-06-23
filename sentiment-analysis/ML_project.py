import re
import string
import logging
import random
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk

# # first run this
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#load dataset
def load_data():
    from kagglehub import KaggleDatasetAdapter
    import kagglehub
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "abhi8923shriv/sentiment-analysis-dataset",
        path="test.csv",
        pandas_kwargs={"encoding": "latin1"}
    )
    df['text'] = df['text'].fillna('')
    df['sentiment'] = df['sentiment'].fillna('neutral')
    return df

#clean data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = []
    for w in tokens:
        if w.isalpha() and w not in stop_words:
            lemma = lemmatizer.lemmatize(w)
            filtered_tokens.append(lemma)
    tokens = filtered_tokens
    return " ".join(tokens)


def augment_text(text, n=2):
    
    # this may cause trouble if wordnet is not downloaded.

    words = text.split()
    new_words = words.copy()
    # Find words that have synonyms in WordNet
    candidate_words = []
    for word in words:
        if wordnet.synsets(word):
            candidate_words.append(word)
    random.shuffle(candidate_words)
    num_replaced = 0
    for word in candidate_words:
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                if lemma_name != word:
                    synonyms.append(lemma_name)
        synonyms = list(set(synonyms))
        if synonyms:
            synonym = random.choice(synonyms)
            # Replace all occurrences of the word in new_words
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)


def build_pipeline(model):
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', model)
    ])


def train_model(X_train, y_train, model_type='logistic'):
    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'nb': MultinomialNB(),
        'svm': LinearSVC()
    }
    model = models.get(model_type, LogisticRegression(max_iter=1000))
    pipe = build_pipeline(model)

    if model_type == 'logistic':
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__solver': ['liblinear', 'lbfgs'],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        }
    elif model_type == 'nb':
        param_grid = {
            'clf__alpha': [0.5, 1.0],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        }
    elif model_type == 'svm':
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        }
    else:
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    logging.info("\n" + classification_report(y_test, preds))


def predict_sentiment(model, texts):
    return model.predict(texts)


def main(model_type='svm'):
    df = load_data()
    df['clean_text'] = df['text'].apply(clean_text)

    # augment training data for much needed diversity
    df_aug = df.copy()
    df_aug['clean_text'] = df_aug['clean_text'].apply(lambda x: augment_text(x))
    df = pd.concat([df, df_aug], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, model_type=model_type)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='svm', choices=['logistic', 'nb', 'svm'], help='Choose model type')
    args = parser.parse_args()
    main(model_type=args.model)
