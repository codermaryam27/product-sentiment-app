import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Page settings
st.set_page_config(page_title="Product Review Sentiment Analyzer", page_icon="📦", layout="centered")

st.title("📦 Product Review Sentiment Analyzer")
st.markdown("### Enter a product review and get instant sentiment prediction (Positive / Negative)")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('best_logistic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, tfidf = load_model()

# Preprocessing functions (same as Week 2)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# User input
review = st.text_area("Paste your Amazon product review here:", 
                      height=150, 
                      placeholder="Example: This phone is amazing! Battery lasts very long and camera quality is superb.")

if st.button("🔍 Analyze Sentiment", type="primary"):
    if review.strip():
        with st.spinner("Analyzing your review..."):
            # Preprocess
            cleaned = clean_text(review)
            processed = remove_stopwords(cleaned)
            
            # Transform with TF-IDF
            vector = tfidf.transform([processed])
            
            # Predict
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]
            
            # Show result
            if prediction == 'positive':
                st.success(f"✅ **Positive Sentiment** ({probability[1]*100:.1f}% confidence)")
                st.balloons()
            else:
                st.error(f"❌ **Negative Sentiment** ({probability[0]*100:.1f}% confidence)")
            
            st.subheader("Processed Review (for understanding):")
            st.write(processed)
            
            # Confidence bar
            st.progress(int(max(probability)*100))
            
    else:
        st.warning("⚠️ Please enter a review first!")

# Sidebar extra info
st.sidebar.header("About this App")
st.sidebar.info("""
This is your **Product Review Sentiment Analyzer** built during the 5-week project.

- Data: Amazon Product Reviews (Kaggle)
- Preprocessing: Cleaning + Stopwords removal
- Features: TF-IDF
- Best Model: Logistic Regression
""")

st.sidebar.success("Made with ❤️ by Maryam")