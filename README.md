# Advanced Sentiment Analyzer

An NLP project that classifies text sentiment as positive, neutral, or negative. This tool incorporates a text preprocessing pipeline, data augmentation to accomodate for a variety of words, and automated hyperparameter tuning to find the best-performing models.

---

## 🎥 Screenshot

<img width="969" height="390" alt="image" src="https://github.com/user-attachments/assets/5d7128a4-af3d-4358-99d9-bb17536f90e9" />


---

## ✨ Features

* **Text Preprocessing:** Cleans text by lowercasing, removing URLs and punctuation, tokenizing, removing stopwords, and applying lemmatization.
* **Data Augmentation:** Utilizes WordNet to replace words with their synonyms, creating a more diverse and rich training dataset.
* **Multiple Model Support:** Easily train and evaluate three different classic machine learning models:

  * Logistic Regression
  * Multinomial Naive Bayes
  * Linear Support Vector Machine (SVM)
* **Hyperparameter Tuning:** Implements GridSearchCV to automatically find the optimal hyperparameters for each model, maximizing performance (F1-score).
* **Efficient Pipelines:** Uses Scikit-learn Pipelines to streamline the workflow from text vectorization to model training.
* **Command-Line Interface:** Select your desired model to train via a simple command-line argument.

---

## 🛠️ Technologies & Libraries

**Python 3.8+**

**Data Manipulation & Analysis:**

* NumPy
* Pandas

**NLP & Text Processing:**

* NLTK (Natural Language Toolkit)

**Machine Learning:**

* Scikit-learn

**Dataset:**

* Kaggle Hub for data loading.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

* Python 3.8 or later
* Pip package manager

### 2. Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/NumeralTiger/Sentiment-analyzer.git
cd Sentiment-analyzer
cd sentiment-analysis
```

2. Create a `requirements.txt` file:

Create a file named `requirements.txt` with the following content:

```text
pandas
numpy
nltk
scikit-learn
kagglehub
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Download NLTK data:

Run the following Python commands once to download the necessary data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## ⚡ Usage

You can train the sentiment analysis model directly from your terminal. The script uses the `argparse` library to let you choose which model to train.

The main argument is `--model`, which can be one of `logistic`, `nb`, or `svm`.

### Examples

To train the **Linear SVM** model (Default & Best Performing):

```bash
python model.py --model svm
```

To train the **Logistic Regression** model:

```bash
python model.py --model logistic
```

To train the **Multinomial Naive Bayes** model:

```bash
python model.py --model nb
```

The script will output the best parameters found during the grid search and then print a detailed classification report for the model's performance on the test set.

### Output Example

```text
C:\Users\zaidm\OneDrive\Desktop\MyProjects\sentiment-analyzer\sentiment-analysis\model.py:35: DeprecationWarning: load_dataset is deprecated and will be removed in a future version.
  df = kagglehub.load_dataset(
2025-07-11 09:31:54,975 - INFO - Best Parameters: {'clf__C': 10, 'tfidf__ngram_range': (1, 2)}
2025-07-11 09:31:55,144 - INFO -
              precision    recall  f1-score   support

    negative       0.85      0.86      0.85       406
     neutral       0.93      0.91      0.92      1076
    positive       0.85      0.90      0.87       444

    accuracy                           0.90      1926
   macro avg       0.88      0.89      0.88      1926
weighted avg       0.90      0.90      0.90      1926
```

