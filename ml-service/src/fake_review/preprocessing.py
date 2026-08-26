import re
import string

import nltk

def _ensure_nltk_data():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load preprocessing resources
stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Apply the same text preprocessing used during
    Fake Review model training.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    words = text.split()

    # Remove stopwords + lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)
