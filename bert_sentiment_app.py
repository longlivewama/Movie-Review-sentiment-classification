import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import base64

# ============ Page Config ============ #
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# ---------- Background ---------- #
BG_PATH = r"D:\NLP\Movie Review sentiment classification\IMDB.avif"

def add_bg_from_local(image_path):
    with open(image_path, "rb") as file:
        data = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/avif;base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.65); 
            z-index: 0;
        }}
        .stMarkdown, .stTextInput, .stTextArea, .stButton, .stHeader, .stTitle, .stSubheader {{
            color: white !important;
            z-index: 1;
            position: relative;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local(BG_PATH)

# ---------- Load Model ---------- #
MODEL_PATH = r"D:\NLP\Movie Review sentiment classification\fine-tuned-bert-imdb"
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

# ============ Main UI ============ #
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review and check if it's positive or negative ✨")

# ---------- Input Text ---------- #
if "text" not in st.session_state:
    st.session_state.text = ""

text = st.text_area("✍️ Write your review here:", st.session_state.text, height=150, key="text")

# ---------- Buttons ----------- #
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Analyze"):
        if text.strip():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                label = torch.argmax(predictions).item()
                confidence = torch.max(predictions).item() * 100

            sentiment = "😊 Positive" if label == 1 else "😡 Negative"
            color = "green" if label == 1 else "red"

            st.subheader("✅ Prediction Result")
            st.markdown(
                f"<h3 style='color:{color};text-align:center;'>{sentiment}</h3>", 
                unsafe_allow_html=True
            )

            # Custom Colored Progress Bar
            st.markdown(
                f"""
                <div style="background-color:#ddd; border-radius:20px; width:100%; height:25px;">
                    <div style="background-color:{color}; width:{confidence}%; height:100%; border-radius:20px; text-align:center; color:white; font-weight:bold;">
                        {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning("⚠️ Please enter a review first.")

with col2:
    if st.button("🔄 Reset"):
        st.session_state.text = ""   # clear textarea
        st.rerun()                   # reload page

# ---------- Sidebar Examples ---------- #
st.sidebar.title("📌 Try Example Reviews")
if st.sidebar.button("⭐ Positive Example"):
    st.session_state.text = "This movie was absolutely fantastic! The acting was brilliant and the story was engaging."
    st.rerun()

if st.sidebar.button("💔 Negative Example"):
    st.session_state.text = "The film was boring and way too long. I regret watching it."
    st.rerun()
