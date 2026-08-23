import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load saved artifacts
model = joblib.load("models/fake_review_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Same preprocessing used during training
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Test reviews
reviews = [
    "This product is absolutely amazing. I have been using it for two months and the quality is excellent.",
    "BUY THIS NOW!!! BEST PRODUCT EVER!!! FIVE STARS!!! YOU WILL LOVE IT!!!",
    "The phone arrived yesterday. The battery lasts around two days and the camera quality is decent.",
    "Terrible product. It stopped working after three days and customer support never responded."
]


for review in reviews:

    cleaned_review = clean_text(review)

    review_tfidf = tfidf.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_tfidf
    )[0]

    if prediction == 1:
        result = "Fake Review"
    else:
        result = "Genuine Review"

    print("\nReview:")
    print(review)

    print("Prediction:", result)