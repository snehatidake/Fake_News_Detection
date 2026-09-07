```python
import streamlit as st
import joblib
import re

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------------------------
# LOAD SAVED FILES
# ---------------------------------
try:
    model = joblib.load("fake_news_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

except Exception as e:
    st.error("Error loading model files.")
    st.write(e)
    st.stop()


# ---------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------
def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------
# TITLE
# ---------------------------------
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below and the machine learning model "
    "will predict whether the news is **Fake** or **Real**."
)

st.divider()


# ---------------------------------
# NEWS INPUT
# ---------------------------------
news_text = st.text_area(
    "✍️ Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)


# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------
if st.button("🔍 Detect News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article first.")

    else:

        # Clean input
        cleaned_news = clean_text(news_text)

        # Convert text into TF-IDF numerical features
        news_vector = tfidf_vectorizer.transform([cleaned_news])

        # Make prediction
        prediction = model.predict(news_vector)

        # Convert encoded prediction back to original label
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # ---------------------------------
        # CONFIDENCE
        # ---------------------------------
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(news_vector)

            confidence = probabilities.max() * 100

        else:
            confidence = None


        st.divider()

        # ---------------------------------
        # RESULT
        # ---------------------------------
        st.subheader("📊 Prediction Result")

        if str(predicted_label).lower() == "fake":

            st.error("🚨 FAKE NEWS")

        else:

            st.success("✅ REAL NEWS")


        # ---------------------------------
        # CONFIDENCE SCORE
        # ---------------------------------
        if confidence is not None:

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )


        # ---------------------------------
        # INFORMATION
        # ---------------------------------
        st.info(
            "The prediction is generated using the trained "
            "machine learning model and TF-IDF text features."
        )


# ---------------------------------
# SIDEBAR
# ---------------------------------
with st.sidebar:

    st.header("ℹ️ About")

    st.write(
        "This Fake News Detection System uses "
        "Machine Learning to classify news articles."
    )

    st.write("### Technology Used")

    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    - Streamlit
    """)

    st.write("### Output")

    st.write("""
    **Fake News** 🚨

    **Real News** ✅

    **Confidence Score** 📊
    """)
```
