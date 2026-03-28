import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
import nltk
import PyPDF2
import time

# Load stopwords safely
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))



# Page config
st.set_page_config(page_title="AI News Classifier", page_icon="📰", layout="centered")

# CSS Styling
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1504711434969-e33886168f5c");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6);
    z-index: -1;
}

/* Glass UI */
.main {
    background: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 18px;
}

/* Text styles */
h1, h2, h3, h4, h5, h6 {
    color: #000 !important;
    font-weight: 800 !important;
}

p, label, span, div {
    color: #111 !important;
    font-weight: 600 !important;
}

/* Text area */
textarea {
    background-color: #ffffff !important;
    color: #000 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

/* Button animation */
.stButton>button {
    background: linear-gradient(270deg, #667eea, #764ba2, #667eea);
    background-size: 400% 400%;
    animation: buttonMove 5s ease infinite;
    color: white;
    border-radius: 25px;
    padding: 12px 30px;
    font-size: 18px;
    border: none;
}

@keyframes buttonMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Result glow */
.result-box {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #ff758c, #ff7eb3);
    color: white;
    text-align: center;
    font-size: 28px;
    font-weight: 900;
    margin-top: 20px;
    box-shadow: 0 0 20px rgba(255, 118, 136, 0.8);
    animation: glow 1.5s infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 0 10px rgba(255, 118, 136, 0.6); }
    to { box-shadow: 0 0 25px rgba(255, 118, 136, 1); }
}

/* Title */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 900;
    color: white;
    text-shadow: 0px 0px 15px rgba(255,255,255,1);
}

.subtitle {
    text-align: center;
    color: #ddd;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# 📰 NEWS TICKER (correct placement)
st.markdown("""
<marquee behavior="scroll" direction="left" style="color:white; font-size:18px;">
Breaking: AI is transforming the world | Markets fluctuate globally | Big match tonight | New tech innovations released!
</marquee>
""", unsafe_allow_html=True)

# Functions
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict_news(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    
    categories = ['🌍 World', '🏏 Sports', '💼 Business', '🤖 Sci/Tech']
    
    return categories[pred], round(prob * 100, 2)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# UI
st.markdown('<div class="title">📰 AI News Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or type news and let AI classify it</div>', unsafe_allow_html=True)

st.info("💡 Tip: Paste a news article or upload a file!")

# Input
option = st.radio("Choose input method:", ["✍ Type Text", "📂 Upload File"])

text_input = ""

if option == "✍ Type Text":
    text_input = st.text_area("Enter News Text:", height=150)

elif option == "📂 Upload File":
    uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text_input = extract_text_from_pdf(uploaded_file)
        else:
            text_input = uploaded_file.read().decode("utf-8")

# Prediction
if st.button("🚀 Predict Category"):
    if text_input:
        with st.spinner("🧠 Analyzing news... Please wait..."):
            time.sleep(2)
            result, confidence = predict_news(text_input)

        st.markdown(
            f"""
            <div class="result-box">
                {result}<br>
                <span style="font-size:18px;">Confidence: {confidence}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.warning("⚠ Please provide input text or file")
        st.markdown("""
<hr>
<center style="color:gray;">Built with ❤️ using NLP & Streamlit</center>
""", unsafe_allow_html=True)
        
