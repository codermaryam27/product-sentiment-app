import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image
import random
import time
import re

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="ARSA - Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# -------------------------------
# LOGO + TITLE
# -------------------------------
col1, col2 = st.columns([1, 4])

with col1:
    try:
        st.image("logo.png", width=80)
    except:
        pass

with col2:
    st.markdown("## 🛍️ ARSA")
    st.caption("Amazon Review Sentiment Analyzer (Powered by BERT)")

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
# BERT SENTIMENT FUNCTION
# -------------------------------
def get_sentiment(text):
    try:
        result = model(text[:512])[0]
        stars = int(result['label'][0])

        if stars >= 4:
            return "positive"
        elif stars == 3:
            return "neutral"
        else:
            return "negative"
    except:
        return "neutral"

# -------------------------------
# SAFE AMAZON SCRAPER (ANTI-BLOCK)
# -------------------------------
HEADERS = [
    {"User-Agent": "Mozilla/5.0"},
    {"User-Agent": "Chrome/120.0"},
    {"User-Agent": "Safari/537.36"}
]

def extract_asin(url):
    match = re.search(r'/dp/([A-Z0-9]{10})', url)
    return match.group(1) if match else None

def scrape_reviews(asin, pages=2):
    reviews = []

    for page in range(1, pages + 1):
        url = f"https://www.amazon.com/product-reviews/{asin}/?pageNumber={page}"
        headers = random.choice(HEADERS)

        try:
            time.sleep(random.uniform(1.5, 3))
            r = requests.get(url, headers=headers, timeout=10)

            # If blocked
            if r.status_code != 200:
                break

            soup = BeautifulSoup(r.text, "html.parser")
            blocks = soup.select("[data-hook='review']")

            if not blocks:
                break

            for b in blocks:
                text = b.select_one("[data-hook='review-body']")
                if text:
                    reviews.append(text.text.strip())

        except:
            break

    return reviews

# -------------------------------
# UI INPUT
# -------------------------------
st.subheader("🔗 Enter Amazon Product URL")
url = st.text_input("Paste Amazon product link")

pages = st.slider("Pages to scrape", 1, 5, 2)

# -------------------------------
# ANALYZE BUTTON
# -------------------------------
if st.button("🚀 Analyze Reviews"):

    if not url.strip():
        st.warning("Please enter a URL")
    else:
        asin = extract_asin(url)

        if not asin:
            st.error("Invalid Amazon URL")
        else:
            with st.spinner("Scraping reviews safely..."):

                reviews = scrape_reviews(asin, pages)

            if len(reviews) == 0:
                st.error("⚠️ Amazon blocked scraping OR no reviews found.")
                st.info("Try again later or use a different product.")
            else:

                sentiments = [get_sentiment(r) for r in reviews]

                pos = sentiments.count("positive")
                neg = sentiments.count("negative")
                neu = sentiments.count("neutral")

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
                # SAMPLE REVIEWS
                # -----------------------
                st.subheader("📝 Sample Reviews")

                for i in range(min(5, len(reviews))):
                    st.write("💬", reviews[i])
                    st.write("➡ Sentiment:", sentiments[i])
                    st.write("---")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🛍️ ARSA")

st.sidebar.info("""
ARSA uses **BERT Transformer AI** to analyze Amazon reviews.

✔ Input: Amazon Product URL  
✔ Output: Positive / Negative / Neutral  
✔ AI Model: BERT (nlptown)

⚠ If Amazon blocks scraping, try again later.
""")

st.sidebar.success("Made with ❤️ by Maryam Nauman")
