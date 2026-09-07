```python
import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load trained files
try:
    model = joblib.load("fake_news_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

except Exception as e:
    st.error("Model files could not be loaded.")
    st.write("Make sure these 3 files are in the same folder as App.py:")
    st.write("1. fake_news_model.pkl")
    st.write("2. tfidf_vectorizer.pkl")
    st.write("3. label_encoder.pkl")
    st.write("Error:", e)
    st.stop()


# Text cleaning function
def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Title
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below to check whether it is "
    "likely to be Fake or Real."
)

st.divider()


# News input
news_text = st.text_area(
    "✍️ Enter News Article",
    height=250,
    placeholder="Paste your news article here..."
)


# Prediction button
if st.button("🔍 Detect News", use_container_width=True):

    if news_text.strip() == "":
        st.warning("⚠️ Please enter a news article.")

    else:

        # Clean news
        cleaned_text = clean_text(news_text)

        # Convert text into TF-IDF
        news_vector = tfidf_vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(news_vector)

        # Convert encoded value back to label
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Confidence
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(news_vector)
            confidence = probabilities.max() * 100

        st.divider()

        st.subheader("📊 Prediction Result")

        # Display result
        if str(predicted_label).lower() == "fake":
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        # Display confidence
        if confidence is not None:

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(int(confidence), 100)
            )

        st.info(
            "This prediction is generated using your trained "
            "Machine Learning model and TF-IDF vectorizer."
        )


# Sidebar
with st.sidebar:

    st.header("ℹ️ About This Project")

    st.write(
        "This application uses Machine Learning to classify "
        "news articles as Fake or Real."
    )

    st.subheader("Technology Used")

    st.write("""
    • Python  
    • Scikit-learn  
    • TF-IDF  
    • Logistic Regression  
    • Streamlit  
    """)

    st.subheader("Output")

    st.write("""
    🚨 Fake News

    ✅ Real News

    📊 Confidence Score
    """)
```
