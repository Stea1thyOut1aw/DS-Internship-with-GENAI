import os
import re
import warnings
import joblib
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

print(f"\n📂 Script location: {BASE_DIR}")
print(f"📄 Data path: {DATA_PATH}")
print(f"💾 Model path: {MODEL_PATH}")


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update({"yonex", "mavis", "shuttlecock", "badminton"})
STEMMER = PorterStemmer()


def preprocess_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    words = [
        STEMMER.stem(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(words)


def load_or_create_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        print("📄 Loading existing data.csv")
        return pd.read_csv(DATA_PATH)

    print("📊 data.csv not found — generating synthetic dataset")

    np.random.seed(42)
    n_samples = 8518

    sentiments = np.random.choice([0, 1], size=n_samples, p=[0.28, 0.72])

    positive_reviews = [
    "excellent durability and long lasting performance",
    "good quality shuttlecock worth the price",
    "smooth flight and stable trajectory",
    "very reliable for regular badminton practice",
    "value for money and decent quality"
   ]

    negative_reviews = [
    "breaks easily and not durable",
    "poor quality product not worth buying",
    "inconsistent speed and bad control",
    "overpriced for such low quality",
    "very disappointing performance"
   ]

    reviews = [
        np.random.choice(negative_reviews if s == 0 else positive_reviews)
        for s in sentiments
    ]

    ratings = np.where(
        sentiments == 1,
        np.random.choice([4, 5], size=n_samples, p=[0.4, 0.6]),
        np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.2]),
    )

    df = pd.DataFrame(
        {
            "rating": ratings,
            "review_text": reviews,
            "sentiment": sentiments,
        }
    )

    df.to_csv(DATA_PATH, index=False)
    print("📁 data.csv created")

    return df


def train():
    df = load_or_create_data()

    print("🧹 Preprocessing reviews...")
    df["clean_review"] = df["review_text"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
    )

    X = vectorizer.fit_transform(df["clean_review"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print("\n🎯 MODEL PERFORMANCE")
    print(f"F1-Score: {f1:.3f}")
    print(classification_report(y_test, y_pred))


    print("\n💾 Saving model artifacts...")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("❌ sentiment_model.pkl was NOT created")

    if not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError("❌ tfidf_vectorizer.pkl was NOT created")

    print("✅ Model files successfully saved:")
    print(f"   → {MODEL_PATH}")
    print(f"   → {VECTORIZER_PATH}")


if __name__ == "__main__":
    print("\n🚀 Starting training pipeline...")
    train()
    print("🏁 Training complete\n")
