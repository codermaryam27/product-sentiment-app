import streamlit as st
from transformers import pipeline
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="ARSA - Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# -------------------------------
# LOGO
# -------------------------------
try:
    st.image("logo.png", width=120)
except:
    st.warning("Logo not found")
    
# -------------------------------
# TITLE
# -------------------------------
st.markdown("## 🛍️ ARSA - Amazon Review Sentiment Analyzer (BERT Powered)")
st.caption("Analyze reviews or text using AI (No scraping, fully safe & stable)")

# -------------------------------
# LOAD MODEL
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
    try:
        result = model(text[:512])[0]
        stars = int(result["label"][0])

        if stars >= 4:
            return "Positive"
        elif stars == 3:
            return "Neutral"
        else:
            return "Negative"
    except:
        return "Neutral"

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("✍️ Enter Product Reviews / Text")

text_input = st.text_area(
    "Write or paste multiple reviews (one per line)",
    height=200
)

# -------------------------------
# ANALYZE BUTTON
# -------------------------------
if st.button("🚀 Analyze Sentiment"):

    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        reviews = [r.strip() for r in text_input.split("\n") if r.strip()]

        sentiments = [get_sentiment(r) for r in reviews]

        # -----------------------
        # COUNT
        # -----------------------
        pos = sentiments.count("Positive")
        neg = sentiments.count("Negative")
        neu = sentiments.count("Neutral")

        # -----------------------
        # METRICS
        # -----------------------
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", pos)
        c2.metric("Negative", neg)
        c3.metric("Neutral", neu)

        # -----------------------
        # CHART
        # -----------------------
        df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Count": [pos, neg, neu]
        })

        st.bar_chart(df.set_index("Sentiment"))

        # -----------------------
        # RESULTS
        # -----------------------
        st.subheader("📝 Results")

        for i, review in enumerate(reviews):
            st.write("💬", review)
            st.write("➡ Sentiment:", sentiments[i])
            st.write("---")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🛍️ ARSA")

st.sidebar.info("""
✔ AI Model: BERT (nlptown)  
✔ Input: Manual text/reviews  
✔ Output: Positive / Negative / Neutral  

⚡ Fully stable (no Amazon blocking issues)
""")

st.sidebar.success("Made with ❤️ by Maryam Nauman")
