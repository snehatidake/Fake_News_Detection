```python
import streamlit as st
import joblib
import nltk

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
# DOWNLOAD NLTK RESOURCES
# =========================================================

@st.cache_resource
def download_nltk_resources():

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


download_nltk_resources()


# =========================================================
# LOAD TRAINED ML FILES
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("fake_news_model (1).pkl")

    tfidf_vectorizer = joblib.load(
        "tfidf_vectorizer (1).pkl"
    )

    label_encoder = joblib.load(
        "label_encoder (1).pkl"
    )

    return model, tfidf_vectorizer, label_encoder


try:

    model, tfidf_vectorizer, label_encoder = load_models()

except Exception as e:

    st.error("❌ Unable to load the trained model files.")

    st.info(
        "Make sure the following files are in the same "
        "GitHub repository as app.py:"
    )

    st.code(
        """
fake_news_model (1).pkl
tfidf_vectorizer (1).pkl
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

    # Convert words back to text
    processed_text = " ".join(stemmed_words)

    return processed_text


# =========================================================
# HEADER
# =========================================================

st.title("📰 Fake News Detection System")

st.write(
    "### Detect whether a news article is **Fake or Real** "
    "using Machine Learning."
)

st.write(
    "Enter or paste a news article below and click "
    "**Detect News**."
)

st.divider()


# =========================================================
# NEWS INPUT
# =========================================================

news = st.text_area(
    "📝 Enter News Article",
    height=250,
    placeholder=(
        "Paste the complete news article here..."
    )
)


# =========================================================
# DETECT BUTTON
# =========================================================

if st.button(
    "🔍 Detect News",
    use_container_width=True
):

    # Check empty input
    if not news.strip():

        st.warning(
            "⚠️ Please enter a news article first."
        )

    else:

        # -------------------------------------------------
        # PREPROCESS TEXT
        # -------------------------------------------------

        processed_text = preprocess_text(news)

        # Check processed text
        if not processed_text.strip():

            st.warning(
                "⚠️ Please enter a meaningful news article."
            )

            st.stop()

        # -------------------------------------------------
        # TF-IDF CONVERSION
        # -------------------------------------------------

        vector = tfidf_vectorizer.transform(
            [processed_text]
        )

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(vector)

        # -------------------------------------------------
        # CONVERT ENCODED LABEL TO ORIGINAL LABEL
        # -------------------------------------------------

        predicted_class = label_encoder.inverse_transform(
            prediction
        )[0]

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(vector)

            confidence = probabilities.max() * 100


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("🎯 Prediction Result")


        # FAKE NEWS
        if str(predicted_class).lower() == "fake":

            st.error(
                "🚨 FAKE NEWS"
            )

            st.warning(
                "⚠️ The model has classified this article "
                "as Fake News."
            )


        # REAL NEWS
        else:

            st.success(
                "✅ REAL NEWS"
            )

            st.success(
                "The model has classified this article "
                "as Real News."
            )


        # =================================================
        # CONFIDENCE
        # =================================================

        if confidence is not None:

            st.metric(
                label="🤖 Model Confidence",
                value=f"{confidence:.2f}%"
            )


        # =================================================
        # ANALYSIS
        # =================================================

        st.subheader("📊 Analysis")

        if confidence is not None:

            st.write(
                f"The Machine Learning model predicts this "
                f"article as **{predicted_class}** with a "
                f"confidence of **{confidence:.2f}%**."
            )

        else:

            st.write(
                f"The Machine Learning model predicts this "
                f"article as **{predicted_class}**."
            )


        # =================================================
        # PREPROCESSED TEXT
        # =================================================

        with st.expander(
            "🔎 View Preprocessed Text"
        ):

            st.write(processed_text)
```
