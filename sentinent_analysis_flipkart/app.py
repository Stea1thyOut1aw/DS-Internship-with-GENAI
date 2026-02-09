
import os
import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")


@st.cache_resource
def load_nltk():
    nltk.download("stopwords", quiet=True)

load_nltk()

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update({"yonex", "mavis", "shuttlecock", "badminton"})
STEMMER = PorterStemmer()


def preprocess_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    words = [
        STEMMER.stem(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(words)


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()


st.title("🧠 Product Review Sentiment Analyzer")
st.markdown(
    "Analyze whether a badminton product review is **positive** or **negative**."
)

review_text = st.text_area(
    "Enter your review:",
    placeholder="The shuttlecock quality is excellent and lasts long...",
    height=150,
)


if st.button("Analyze Sentiment"):
    if not review_text.strip():
        st.warning("⚠️ Please enter a review.")
    elif model is None or vectorizer is None:
        st.error("❌ Model files not found. Please train the model first.")
    else:
        clean_review = preprocess_text(review_text)
        X = vectorizer.transform([clean_review])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0].max()

        if prediction == 1:
            st.success(f"✅ Positive Review ({confidence:.2%} confidence)")
        else:
            st.error(f"❌ Negative Review ({confidence:.2%} confidence)")


st.markdown("---")
st.caption("Built with Streamlit · TF-IDF · Logistic Regression")