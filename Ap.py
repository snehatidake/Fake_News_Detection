import streamlit as st
import joblib
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# -------------------------------
# Load trained ML files
# -------------------------------

model = joblib.load("fake_news_model (1).pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
label_encoder = joblib.load("label_encoder (1).pkl")


# -------------------------------
# Text preprocessing
# -------------------------------

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):

    tokens = word_tokenize(text.lower())

    filtered = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    stemmed = [
        stemmer.stem(word)
        for word in filtered
    ]

    processed_text = " ".join(stemmed)

    return processed_text


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below and the Machine Learning model "
    "will predict whether the news is Fake or Real."
)


st.divider()


# News input box

news = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)


# Predict button

if st.button("🔍 Detect News", use_container_width=True):

    if news.strip() == "":
        st.warning("⚠️ Please enter a news article first.")

    else:

        # Preprocess news
        processed_text = preprocess_text(news)

        # Convert text into TF-IDF numbers
        vector = tfidf_vectorizer.transform([processed_text])

        # Prediction
        prediction = model.predict(vector)

        # Convert encoded prediction back to Fake/Real
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        # Confidence
        probabilities = model.predict_proba(vector)
        confidence = probabilities.max() * 100


        st.divider()

        st.subheader("🎯 Prediction Result")


        # Display result

        if str(predicted_class).lower() == "fake":

            st.error("🚨 FAKE NEWS")

        else:

            st.success("✅ REAL NEWS")


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        st.subheader("📊 Analysis")

        st.write(
            f"The model predicts this article as **{predicted_class}** "
            f"with a confidence of **{confidence:.2f}%**."
        )