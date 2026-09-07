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

# Load trained files
model = joblib.load("fake_news_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Text preprocessing
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


# Streamlit UI
st.title("📰 Fake News Detection")
st.write("Enter a news article below to check whether it is Fake or Real.")

news_text = st.text_area(
    "Enter News Text",
    height=200
)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        processed_text = preprocess_text(news_text)

        text_vector = tfidf_vectorizer.transform([processed_text])

        prediction = model.predict(text_vector)

        predicted_label = label_encoder.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(text_vector)[0]
        confidence = probabilities.max() * 100

        if str(predicted_label).lower() == "fake":
            st.error(f"🚨 FAKE NEWS")
        else:
            st.success(f"✅ REAL NEWS")

        st.info(f"Confidence: {confidence:.2f}%")
