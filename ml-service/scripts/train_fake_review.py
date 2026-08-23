import os
import re
import string
import joblib
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "fake_reviews",
    "fake_reviews.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fake_review_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Prepare labels
# --------------------------------------------------

df["label"] = df["label"].map({
    "CG": 1,
    "OR": 0
})

df = df[["text_", "label"]]

df.rename(
    columns={"text_": "review"},
    inplace=True
)


# --------------------------------------------------
# 4. Text preprocessing
# --------------------------------------------------

print("Preparing text preprocessing...")

import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


print("Cleaning reviews...")

df["clean_review"] = df["review"].apply(
    clean_text
)


# --------------------------------------------------
# 5. TF-IDF
# --------------------------------------------------

print("Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=5000
)

X = tfidf.fit_transform(
    df["clean_review"]
)

y = df["label"]

print("TF-IDF shape:", X.shape)


# --------------------------------------------------
# 6. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 7. Train Linear SVM
# --------------------------------------------------

print("Training Linear SVM...")

svm = LinearSVC(
    random_state=42
)

svm.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. Evaluate
# --------------------------------------------------

pred_svm = svm.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred_svm
)

precision = precision_score(
    y_test,
    pred_svm
)

recall = recall_score(
    y_test,
    pred_svm
)

f1 = f1_score(
    y_test,
    pred_svm
)


print("\n--- Linear SVM Results ---")

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall   : {recall:.6f}"
)

print(
    f"F1 Score : {f1:.6f}"
)


# --------------------------------------------------
# 9. Save artifacts
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print("\nSaving model...")

joblib.dump(
    svm,
    MODEL_PATH
)

print(
    "Saved:",
    MODEL_PATH
)


print("Saving TF-IDF vectorizer...")

joblib.dump(
    tfidf,
    VECTORIZER_PATH
)

print(
    "Saved:",
    VECTORIZER_PATH
)


# --------------------------------------------------
# 10. Verify files
# --------------------------------------------------

print("\n--- Verification ---")

print(
    "Model exists:",
    os.path.exists(MODEL_PATH)
)

print(
    "Vectorizer exists:",
    os.path.exists(VECTORIZER_PATH)
)

print(
    "\nTraining and artifact creation completed."
)