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

# first run this
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
    return " ".join(tokens)


def augment_text(text, n=2):
    """
    Placeholder for EDA synonym replacement. Ensure wordnet is downloaded.
    """
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        synonym_words = list(set([lemma.name().replace("_", " ") for syn in synonyms for lemma in syn.lemmas() if lemma.name() != word]))
        if synonym_words:
            synonym = random.choice(synonym_words)
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

    param_grid = {
        'clf__C': [0.1, 1, 10] if model_type == 'logistic' else [1],
        'clf__solver': ['liblinear', 'lbfgs'] if model_type == 'logistic' else [''],
        'tfidf__ngram_range': [(1, 1), (1, 2)]
    }
    param_grid = {k: v for k, v in param_grid.items() if v != ['']}

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    logging.info(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    logging.info("\n" + classification_report(y_test, preds))


def predict_sentiment(model, texts):
    return model.predict(texts)


def main():
    df = load_data()
    df['clean_text'] = df['text'].apply(clean_text)

    # Optional: augment training data
    df_aug = df.copy()
    df_aug['clean_text'] = df_aug['clean_text'].apply(lambda x: augment_text(x))
    df = pd.concat([df, df_aug], ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, model_type='logistic')
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='logistic', choices=['logistic', 'nb', 'svm'], help='Choose model type')
    args = parser.parse_args()
    main()
