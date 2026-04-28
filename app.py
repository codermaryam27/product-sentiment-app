import streamlit as st
from transformers import pipeline
import requests
import pandas as pd
from PIL import Image

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="ARSA - Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# -------------------------------
# LOGO + TITLE (TOP)
# -------------------------------
col1, col2 = st.columns([1,4])

with col1:
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=80)
    except:
        pass

with col2:
    st.markdown("## 🛍️ ARSA")
    st.caption("Amazon Review Sentiment Analyzer (AI-powered using BERT)")

# -------------------------------
# LOAD BERT MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

model = load_model()

# -------------------------------
# SENTIMENT FUNCTION
# -------------------------------
def get_sentiment(text):
    result = model(text[:512])[0]
    stars = int(result['label'][0])

    if stars >= 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    else:
        return "negative"

# -------------------------------
# REVIEW FETCH (DEMO API)
# -------------------------------
def get_reviews(url):
    api_url = "https://dummyjson.com/comments"
    res = requests.get(api_url)
    data = res.json()
    reviews = [item["body"] for item in data["comments"]]
    return reviews

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("🔗 Enter Amazon Product URL")
url = st.text_input("Paste product link here")

# -------------------------------
# ANALYZE BUTTON
# -------------------------------
if st.button("🚀 Analyze Reviews"):

    if url.strip() == "":
        st.warning("⚠️ Please enter a product URL")
    else:
        with st.spinner("Analyzing reviews using AI..."):

            reviews = get_reviews(url)

            if len(reviews) == 0:
                st.error("No reviews found!")
            else:
                sentiments = [get_sentiment(r) for r in reviews]

                pos = sentiments.count("positive")
                neg = sentiments.count("negative")
                neu = sentiments.count("neutral")

                # -------------------------------
                # METRICS
                # -------------------------------
                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Positive", pos)
                col2.metric("❌ Negative", neg)
                col3.metric("⚖️ Neutral", neu)

                # -------------------------------
                # CHART
                # -------------------------------
                df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [pos, neg, neu]
                })

                st.subheader("📊 Sentiment Distribution")
                st.bar_chart(df.set_index("Sentiment"))

                # -------------------------------
                # SAMPLE REVIEWS
                # -------------------------------
                st.subheader("📝 Sample Reviews with Prediction")

                for i in range(min(5, len(reviews))):
                    st.write(f"**Review:** {reviews[i]}")
                    st.write(f"**Sentiment:** {sentiments[i]}")
                    st.write("---")

# -------------------------------
# SIDEBAR (LOGO + ABOUT)
# -------------------------------
try:
    sidebar_logo = Image.open("logo.png")
    st.sidebar.image(sidebar_logo, width=120)
except:
    pass

st.sidebar.markdown("## 🛍️ ARSA")

st.sidebar.info("""
**ARSA - Amazon Review Sentiment Analyzer**

This is an AI-powered web app that analyzes sentiment of product reviews.

🔹 Model: BERT (Transformer)
🔹 Input: Amazon Product URL
🔹 Output: Positive / Negative / Neutral sentiment

✨ Features:
- Real-time sentiment prediction
- Interactive charts
- Clean UI

📌 Note:
Currently using demo API for reviews (can be replaced with real Amazon API).
""")

st.sidebar.caption("Version 2.0 | Powered by AI")
st.sidebar.success("Made with ❤️ by Maryam Nauman")
