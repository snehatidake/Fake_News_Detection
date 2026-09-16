```python
import streamlit as st
import joblib
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# =========================================================
# NLTK RESOURCES
# =========================================================

@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


download_nltk_resources()


# =========================================================
# LOAD TRAINED MODEL FILES
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("fake_news_model (1).pkl")

    tfidf_vectorizer = joblib.load(
        "tfidf_vectorizer (1)(1).pkl"
    )

    label_encoder = joblib.load(
        "label_encoder (1).pkl"
    )

    return model, tfidf_vectorizer, label_encoder


try:
    model, tfidf_vectorizer, label_encoder = load_models()
except Exception as e:

    st.error("❌ Model files could not be loaded.")

    st.info(
        "Make sure these 3 files are in the same GitHub repository "
        "as App.py."
    )

    st.code(
        """
fake_news_model (1).pkl
tfidf_vectorizer (1)(1).pkl
label_encoder (1).pkl
        """
    )

    st.stop()


# =========================================================
# TEXT PREPROCESSING
# =========================================================

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    filtered_tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    # Stemming
    stemmed_tokens = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    # Convert list back to text
    processed_text = " ".join(stemmed_tokens)

    return processed_text


# =========================================================
# HEADER
# =========================================================

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below and the Machine Learning "
    "model will predict whether the news is Fake or Real."
)

st.divider()


# =========================================================
# NEWS INPUT
# =========================================================

news = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)


# =========================================================
# DETECT BUTTON
# =========================================================

if st.button("🔍 Detect News", use_container_width=True):

    if not news.strip():

        st.warning(
            "⚠️ Please enter a news article first."
        )

    else:

        try:

            # -------------------------------------------------
            # PREPROCESS TEXT
            # -------------------------------------------------

            processed_text = preprocess_text(news)

            # -------------------------------------------------
            # TF-IDF TRANSFORMATION
            # -------------------------------------------------

            vector = tfidf_vectorizer.transform(
                [processed_text]
            )

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            prediction = model.predict(vector)

            # Convert encoded value to original label
            predicted_class = label_encoder.inverse_transform(
                prediction
            )[0]

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            probabilities = model.predict_proba(vector)

            confidence = probabilities.max() * 100

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.divider()

            st.subheader("🎯 Prediction Result")

            if str(predicted_class).lower() == "fake":

                st.error("🚨 FAKE NEWS")

            else:

                st.success("✅ REAL NEWS")

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # -------------------------------------------------
            # PROGRESS BAR
            # -------------------------------------------------

            st.write("Confidence Level")

            st.progress(
                min(int(confidence), 100)
            )

            # -------------------------------------------------
            # ANALYSIS
            # -------------------------------------------------

            st.subheader("📊 Analysis")

            st.write(
                f"The model predicts this article as "
                f"**{predicted_class.upper()}** "
                f"with a confidence of "
                f"**{confidence:.2f}%**."
            )

            # -------------------------------------------------
            # PREPROCESSED TEXT
            # -------------------------------------------------

            with st.expander("🔎 View Processed Text"):

                st.write(processed_text)

        except Exception as e:

            st.error(
                "❌ An error occurred while processing the news."
            )

            st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Fake News Detection System | Machine Learning Project"
)

