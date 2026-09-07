import streamlit as st
import joblib
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# =========================================================
# 1. NLTK RESOURCES
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# =========================================================
# 2. LOAD TRAINED ML FILES
# =========================================================

try:
    model = joblib.load("fake_news_model (1).pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
    label_encoder = joblib.load("label_encoder (1).pkl")

except FileNotFoundError:
    st.error(
        "❌ Model files not found. Make sure these files are in "
        "the same folder as app.py."
    )
    st.stop()


# =========================================================
# 3. TEXT PREPROCESSING
# =========================================================

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation, numbers and stopwords
    filtered_words = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    # Stemming
    stemmed_words = [
        stemmer.stem(word)
        for word in filtered_words
    ]

    # Convert list back to text
    processed_text = " ".join(stemmed_words)

    return processed_text


# =========================================================
# 4. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# =========================================================
# 5. TITLE
# =========================================================

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below and the Machine Learning "
    "model will predict whether the news is **Fake or Real**."
)

st.divider()


# =========================================================
# 6. NEWS INPUT
# =========================================================

news = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)


# =========================================================
# 7. PREDICTION BUTTON
# =========================================================

if st.button("🔍 Detect News", use_container_width=True):

    # Check empty input
    if not news.strip():

        st.warning("⚠️ Please enter a news article first.")

    else:

        # -------------------------------------------------
        # Step 1: Preprocess the news
        # -------------------------------------------------

        processed_text = preprocess_text(news)

        # Check if meaningful words remain
        if not processed_text.strip():

            st.warning(
                "⚠️ The article does not contain enough meaningful "
                "English words for prediction."
            )
            st.stop()

        # -------------------------------------------------
        # Step 2: Convert text into TF-IDF numerical vector
        # -------------------------------------------------

        vector = tfidf_vectorizer.transform(
            [processed_text]
        )

        # -------------------------------------------------
        # Step 3: Make prediction
        # -------------------------------------------------

        prediction = model.predict(vector)

        # -------------------------------------------------
        # Step 4: Convert encoded label back to Fake/Real
        # -------------------------------------------------

        predicted_class = label_encoder.inverse_transform(
            prediction
        )[0]

        # -------------------------------------------------
        # Step 5: Calculate confidence
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(vector)

            confidence = probabilities.max() * 100

        else:

            confidence = None

        # -------------------------------------------------
        # Step 6: Display result
        # -------------------------------------------------

        st.divider()

        st.subheader("🎯 Prediction Result")

        if str(predicted_class).lower() == "fake":

            st.error("🚨 FAKE NEWS")

        else:

            st.success("✅ REAL NEWS")

        # -------------------------------------------------
        # Step 7: Display confidence
        # -------------------------------------------------

        if confidence is not None:

            st.metric(
                label="🤖 Model Confidence",
                value=f"{confidence:.2f}%"
            )

        # -------------------------------------------------
        # Step 8: Analysis
        # -------------------------------------------------

        st.subheader("📊 Analysis")

        if confidence is not None:

            st.write(
                f"The Machine Learning model predicts this article "
                f"as **{predicted_class}** with a model confidence "
                f"of **{confidence:.2f}%**."
            )

        else:

            st.write(
                f"The Machine Learning model predicts this article "
                f"as **{predicted_class}**."
            )

        # -------------------------------------------------
        # Step 9: Show processed text
        # -------------------------------------------------

        with st.expander("🔎 View Preprocessed Text"):

            st.write(processed_text)
