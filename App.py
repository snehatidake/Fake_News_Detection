import streamlit as st
import joblib
import re
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>
.main {
    padding: 2rem;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<div class="title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered news classification using Machine Learning</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Model File Paths
# --------------------------------------------------

MODEL_FILE = "fake_news_model (1).pkl"
VECTORIZER_FILE = "tfidf_vectorizer (1).pkl"
ENCODER_FILE = "label_encoder (1).pkl"

# --------------------------------------------------
# Check Files
# --------------------------------------------------

missing_files = []

if not os.path.exists(MODEL_FILE):
    missing_files.append(MODEL_FILE)

if not os.path.exists(VECTORIZER_FILE):
    missing_files.append(VECTORIZER_FILE)

if not os.path.exists(ENCODER_FILE):
    missing_files.append(ENCODER_FILE)

if missing_files:

    st.error("❌ Required model files are missing.")

    st.write("Please make sure these files are uploaded to GitHub:")

    for file in missing_files:
        st.write("•", file)

    st.stop()

# --------------------------------------------------
# Load Model
# --------------------------------------------------

try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    label_encoder = joblib.load(ENCODER_FILE)

except Exception as e:

    st.error("❌ Error loading ML model files.")

    st.write("Please check that the `.pkl` files are valid.")

    st.stop()

# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------

def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --------------------------------------------------
# News Input
# --------------------------------------------------

st.subheader("Enter News Article")

news_text = st.text_area(
    "Paste the news article below:",
    height=220,
    placeholder="Enter or paste news article here..."
)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("🔍 Detect Fake News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article first.")

    else:

        try:

            # Preprocess
            cleaned_text = preprocess_text(news_text)

            # TF-IDF transformation
            transformed_text = vectorizer.transform([cleaned_text])

            # Prediction
            prediction = model.predict(transformed_text)

            # Convert prediction to original label
            try:
                result = label_encoder.inverse_transform(prediction)[0]
            except Exception:
                result = str(prediction[0])

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(transformed_text)

                confidence = max(probabilities[0]) * 100

            # --------------------------------------------------
            # Result
            # --------------------------------------------------

            result_text = str(result).lower()

            if "fake" in result_text:

                st.error(
                    "🚨 FAKE NEWS DETECTED"
                )

            elif "real" in result_text:

                st.success(
                    "✅ REAL NEWS DETECTED"
                )

            else:

                st.info(
                    f"Prediction: {result}"
                )

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            if confidence is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

        except Exception as e:

            st.error("❌ Prediction error occurred.")

            st.write(
                "Please make sure the preprocessing used during training "
                "matches the preprocessing used in this application."
            )
