import streamlit as st
import pandas as pd
import re
import nltk
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import random

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiScope — Amazon Review Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Syne:wght@800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.logo-block {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.2rem 0 0.4rem;
}
.logo-icon {
    background: linear-gradient(135deg, #FF6B35 0%, #F7C59F 100%);
    border-radius: 16px;
    width: 56px; height: 56px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    box-shadow: 0 4px 18px rgba(255,107,53,0.35);
}
.logo-text-main {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF6B35, #c44d09);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 0;
}
.logo-text-sub {
    font-size: 0.78rem;
    color: #888;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

.stat-card {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.stat-number { font-size: 2.2rem; font-weight: 700; line-height: 1; }
.stat-label  { font-size: 0.8rem; color: #aaa; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }

.pos-color { color: #4ade80; }
.neg-color { color: #f87171; }
.neu-color { color: #94a3b8; }

.review-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.pos-pill { background: #14532d; color: #4ade80; }
.neg-pill { background: #7f1d1d; color: #f87171; }

.tip-box {
    background: rgba(255,107,53,0.08);
    border-left: 3px solid #FF6B35;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #ccc;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── NLTK Downloads ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    for pkg in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True

download_nltk()

# ── Text Processing ───────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return ' '.join(w for w in tokens if w not in STOP_WORDS)

def preprocess(text: str) -> str:
    return remove_stopwords(clean_text(text))

# ── Model Training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Train model fresh — avoids sklearn pickle version mismatch.
    Uses a balanced synthetic seed dataset + loads user CSV if present.
    """
    import os

    seed_reviews = [
        # Positive
        ("absolutely love this product works perfectly every time highly recommend", "positive"),
        ("great quality fast shipping exceeded my expectations will buy again", "positive"),
        ("amazing product exactly as described very happy with my purchase", "positive"),
        ("excellent build quality durable and works as advertised five stars", "positive"),
        ("best purchase ever outstanding quality love it so much", "positive"),
        ("good product value for money arrived quickly well packaged", "positive"),
        ("works great as expected solid quality nice design happy", "positive"),
        ("perfect fit great material easy to use very satisfied", "positive"),
        ("superb product top notch quality would definitely recommend friends", "positive"),
        ("brilliant product does exactly what it says love it", "positive"),
        ("nice product good quality satisfied with delivery time", "positive"),
        ("product is decent does the job good for the price", "positive"),
        # Negative
        ("terrible quality broke after one day complete waste of money", "negative"),
        ("worst product ever do not buy total garbage returned immediately", "negative"),
        ("stopped working after a week very disappointed bad quality", "negative"),
        ("arrived damaged poor packaging customer service unhelpful avoid this", "negative"),
        ("nothing like the description misleading product false advertising", "negative"),
        ("cheap flimsy material feels like it will break any moment", "negative"),
        ("does not work at all useless product very frustrating", "negative"),
        ("absolute rubbish broke in two days waste of money", "negative"),
        ("poor quality not worth the price very unsatisfied", "negative"),
        ("horrible experience product defective seller unresponsive stay away", "negative"),
        ("disappointing quality not as pictured returned for refund", "negative"),
        ("bad product fell apart immediately extremely dissatisfied", "negative"),
    ]

    texts  = [preprocess(r[0]) for r in seed_reviews]
    labels = [r[1] for r in seed_reviews]

    # Try to load user's CSV for better accuracy
    for csv_name in ["amazon_dataset.csv", "amazon.csv"]:
        if os.path.exists(csv_name):
            try:
                df = pd.read_csv(csv_name)
                rating_col = next(
                    (c for c in df.columns if 'rating' in c.lower()), None
                )
                text_col = next(
                    (c for c in df.columns if 'text' in c.lower() or 'review' in c.lower()), None
                )
                if rating_col and text_col:
                    df = df.dropna(subset=[rating_col, text_col])
                    # FIXED: >= 4 is positive (not >= 5)
                    df['sentiment'] = df[rating_col].apply(
                        lambda x: 'positive' if float(x) >= 4 else 'negative'
                    )
                    sample = df.sample(min(3000, len(df)), random_state=42)
                    texts  += sample[text_col].apply(preprocess).tolist()
                    labels += sample['sentiment'].tolist()
                    st.sidebar.success(f"✅ Loaded {len(sample)} real reviews from {csv_name}")
            except Exception as e:
                st.sidebar.warning(f"Could not load {csv_name}: {e}")
            break

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        random_state=42,
    )
    model.fit(X, labels)
    return tfidf, model

# ── Amazon Scraper ────────────────────────────────────────────────────────────
HEADERS_LIST = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                      "Version/17.3 Safari/605.1.15",
        "Accept-Language": "en-US,en;q=0.8",
    },
]

def extract_asin(url: str) -> str | None:
    match = re.search(r'/(?:dp|product|gp/product)/([A-Z0-9]{10})', url)
    return match.group(1) if match else None

def scrape_amazon_reviews(asin: str, pages: int = 3) -> list[dict]:
    reviews = []
    base = "https://www.amazon.com/product-reviews/{asin}/?pageNumber={page}"
    session = requests.Session()

    for page in range(1, pages + 1):
        url = base.format(asin=asin, page=page)
        headers = random.choice(HEADERS_LIST)
        try:
            time.sleep(random.uniform(1.5, 3.0))
            resp = session.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, 'html.parser')
            items = soup.select('[data-hook="review"]')
            if not items:
                break
            for item in items:
                title_el  = item.select_one('[data-hook="review-title"]')
                body_el   = item.select_one('[data-hook="review-body"]')
                rating_el = item.select_one('[data-hook="review-star-rating"]')
                if body_el:
                    rating_text = rating_el.get_text(strip=True) if rating_el else "0"
                    try:
                        stars = float(rating_text.split()[0])
                    except Exception:
                        stars = 0.0
                    reviews.append({
                        "title":  title_el.get_text(strip=True) if title_el else "",
                        "review": body_el.get_text(strip=True),
                        "stars":  stars,
                    })
        except Exception as e:
            st.warning(f"Page {page} scrape failed: {e}")
            break

    return reviews

def predict_sentiment(texts: list[str], tfidf, model) -> list[str]:
    processed = [preprocess(t) for t in texts]
    X = tfidf.transform(processed)
    return model.predict(X).tolist()

# ── Donut Chart ───────────────────────────────────────────────────────────────
def donut_chart(pos: int, neg: int):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
    sizes  = [pos, neg]
    colors = ['#4ade80', '#f87171']
    wedge_props = dict(width=0.45, edgecolor='#1a1a2e', linewidth=2.5)
    ax.pie(sizes, colors=colors, wedgeprops=wedge_props, startangle=90)
    total = pos + neg or 1
    pct = f"{pos/total*100:.0f}%"
    ax.text(0, 0, pct, ha='center', va='center',
            fontsize=22, fontweight='bold', color='#4ade80')
    ax.set_title("Positive", color='#aaa', fontsize=11, pad=8)
    fig.patch.set_alpha(0)
    return fig

# ── Logo ──────────────────────────────────────────────────────────────────────
def render_logo():
    st.markdown("""
    <div class="logo-block">
        <div class="logo-icon">🔬</div>
        <div>
            <p class="logo-text-main">SentiScope</p>
            <p class="logo-text-sub">Amazon Review Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

render_logo()
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    pages_to_scrape = st.slider("Review pages to scrape", 1, 5, 2)
    show_raw = st.checkbox("Show raw review table", value=False)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **SentiScope** uses TF-IDF + Logistic Regression 
    trained on Amazon product reviews.
    
    - ⭐⭐⭐⭐⭐ = Positive
    - ⭐⭐⭐⭐ = Positive
    - ⭐⭐⭐ or below = Negative
    """)

# Load model
with st.spinner("Loading sentiment model…"):
    tfidf, model = load_model()

st.success("✅ Sentiment model ready.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔗  Analyze by Amazon URL", "✏️  Analyze Custom Text"])

# ── Tab 1: URL ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Paste an Amazon product URL")
    st.markdown("""
    <div class="tip-box">
        Paste any Amazon product URL, e.g.<br>
        <code>https://www.amazon.com/dp/B08N5WRWNW</code>
    </div>
    """, unsafe_allow_html=True)

    url_input = st.text_input(
        label="Amazon URL",
        placeholder="https://www.amazon.com/dp/B08N5WRWNW",
        label_visibility="collapsed",
    )

    if st.button("🔍  Scrape & Analyze", use_container_width=True, key="scrape_btn"):
        if not url_input.strip():
            st.error("Please enter a URL.")
        else:
            asin = extract_asin(url_input)
            if not asin:
                st.error("Could not find ASIN in URL. Make sure it contains `/dp/XXXXXXXXXX`.")
            else:
                st.info(f"Product ASIN detected: **{asin}** — scraping {pages_to_scrape} page(s)…")
                with st.spinner("Scraping Amazon reviews… (this may take 10–20 seconds)"):
                    reviews = scrape_amazon_reviews(asin, pages=pages_to_scrape)

                if not reviews:
                    st.warning(
                        "⚠️ No reviews scraped. Amazon may be blocking the request. "
                        "Try again or use the manual text tab."
                    )
                else:
                    df_reviews = pd.DataFrame(reviews)
                    texts_to_analyze = (
                        df_reviews['title'] + " " + df_reviews['review']
                    ).tolist()
                    df_reviews['sentiment'] = predict_sentiment(texts_to_analyze, tfidf, model)

                    pos_count = (df_reviews['sentiment'] == 'positive').sum()
                    neg_count = (df_reviews['sentiment'] == 'negative').sum()
                    total     = len(df_reviews)

                    st.markdown("### 📊 Results")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number neu-color">{total}</div>
                            <div class="stat-label">Total Reviews</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number pos-color">{pos_count}</div>
                            <div class="stat-label">Positive 👍</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number neg-color">{neg_count}</div>
                            <div class="stat-label">Negative 👎</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("")
                    col_chart, col_info = st.columns([1, 2])
                    with col_chart:
                        fig = donut_chart(pos_count, neg_count)
                        st.pyplot(fig, use_container_width=True)
                    with col_info:
                        pos_pct = pos_count / total * 100 if total else 0
                        neg_pct = 100 - pos_pct
                        verdict = "🟢 Mostly Positive" if pos_pct > 65 else (
                                  "🔴 Mostly Negative" if neg_pct > 65 else
                                  "🟡 Mixed Reviews")
                        st.markdown(f"### Verdict: {verdict}")
                        st.progress(int(pos_pct))
                        st.caption(f"{pos_pct:.1f}% positive · {neg_pct:.1f}% negative")
                        st.markdown(
                            "*Based on NLP analysis of scraped review text.*"
                        )

                    if show_raw:
                        st.markdown("### 📋 Raw Reviews")
                        st.dataframe(
                            df_reviews[['stars', 'title', 'review', 'sentiment']],
                            use_container_width=True,
                        )

                    # Sample reviews
                    st.markdown("### 💬 Sample Reviews")
                    pos_samples = df_reviews[df_reviews['sentiment'] == 'positive'].head(3)
                    neg_samples = df_reviews[df_reviews['sentiment'] == 'negative'].head(3)

                    for _, row in pd.concat([pos_samples, neg_samples]).iterrows():
                        pill_cls = "pos-pill" if row['sentiment'] == 'positive' else "neg-pill"
                        pill_txt = "Positive" if row['sentiment'] == 'positive' else "Negative"
                        st.markdown(f"""
                        <div style="background:#111827;border-radius:12px;padding:1rem 1.2rem;margin:6px 0;border:1px solid #2a2a3a;">
                            <span class="review-pill {pill_cls}">{pill_txt}</span>
                            <span style="font-size:0.8rem;color:#888;margin-left:8px;">⭐ {row['stars']:.0f}</span>
                            <p style="margin:0.5rem 0 0;font-size:0.92rem;color:#ddd;">{row['review'][:250]}{'…' if len(row['review'])>250 else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)

# ── Tab 2: Custom Text ────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Type or paste any review text")
    custom_text = st.text_area(
        label="Review text",
        placeholder="e.g. This product is absolutely amazing! The build quality is fantastic.",
        height=160,
        label_visibility="collapsed",
    )

    if st.button("🧠  Predict Sentiment", use_container_width=True, key="predict_btn"):
        if not custom_text.strip():
            st.error("Please enter some text.")
        else:
            pred = predict_sentiment([custom_text], tfidf, model)[0]
            proba = model.predict_proba(tfidf.transform([preprocess(custom_text)]))[0]
            classes = model.classes_.tolist()
            pos_score = proba[classes.index('positive')] if 'positive' in classes else 0

            color = "#4ade80" if pred == "positive" else "#f87171"
            emoji = "😊" if pred == "positive" else "😞"

            st.markdown(f"""
            <div style="background:#111827;border-radius:16px;padding:1.5rem 2rem;
                        border:1px solid {color}44;margin-top:1rem;text-align:center;">
                <div style="font-size:3rem;">{emoji}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};margin:0.3rem 0;">
                    {pred.upper()}
                </div>
                <div style="font-size:0.9rem;color:#888;">
                    Confidence: {max(proba)*100:.1f}%
                </div>
                <div style="margin-top:1rem;">
                    <div style="height:8px;background:#1f2937;border-radius:999px;overflow:hidden;">
                        <div style="height:100%;width:{pos_score*100:.0f}%;
                                    background:linear-gradient(90deg,#4ade80,#22c55e);
                                    border-radius:999px;transition:width 0.5s;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.75rem;color:#666;margin-top:4px;">
                        <span>Negative</span><span>Positive</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
