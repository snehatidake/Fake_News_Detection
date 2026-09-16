import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# --------------------------------------------------
# Download NLTK Resources
# --------------------------------------------------

@st.cache_resource
def download_nltk_resources():
    nltk.download("stopwords", quiet=True)

download_nltk_resources()

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# --------------------------------------------------
# Load Trained ML Files
# --------------------------------------------------

@st.cache_resource
def load_models():

    model = joblib.load("fake_news_model (1).pkl")
    vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
    label_encoder = joblib.load("label_encoder (1).pkl")

    return model, vectorizer, label_encoder


try:
    model, vectorizer, label_encoder = load_models()
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error("⚠️ Model files could not be loaded.")
    st.info(
        "Please check that all three .pkl files are present "
        "in your GitHub repository."
    )

# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below to predict whether it is "
    "Fake or Real using Machine Learning."
)

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste your news article here:",
    height=220,
    placeholder="Enter news article text..."
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict News", use_container_width=True):

    if not model_loaded:
        st.error("Model is not loaded. Please check your .pkl files.")

    elif news_text.strip() == "":
        st.warning("⚠️ Please enter a news article first.")

    else:

        with st.spinner("Analyzing news..."):

            cleaned_text = preprocess_text(news_text)

            transformed_text = vectorizer.transform(
                [cleaned_text]
            )

            prediction = model.predict(transformed_text)

            predicted_label = label_encoder.inverse_transform(
                prediction
            )[0]

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        st.subheader("📊 Prediction Result")

        label = str(predicted_label).strip().lower()

        if label in ["fake", "1", "false"]:

            st.error("🚨 FAKE NEWS")

            st.write(
                "The model predicts that this news article "
                "is Fake."
            )

        else:

            st.success("✅ REAL NEWS")

            st.write(
                "The model predicts that this news article "
                "is Real."
            )

        # --------------------------------------------------
        # Confidence Score
        # --------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                transformed_text
            )[0]

            confidence = max(probabilities) * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        st.info(
            "Note: This prediction is based on the trained "
            "machine learning model and may not always be correct."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Fake News Detection Project | "
    "Python • NLP • TF-IDF • Machine Learning • Streamlit"
)
