# ------------------ Imports ------------------
import streamlit as st
from transformers import pipeline
import os
from dotenv import load_dotenv

# ------------------ Load .env ------------------
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# ------------------ Check Token ------------------
if not HF_API_TOKEN:
    st.error("Hugging Face API token not found. Add it to .env or Streamlit Secrets.")
    st.stop()

# ------------------ Load Model ------------------
@st.cache_resource(show_spinner=True)
def load_generator():
    generator = pipeline(
        "text2text-generation",  # FLAN-T5 is seq2seq
        model="google/flan-t5-small",
        device=-1,               # CPU
        use_auth_token=HF_API_TOKEN
    )
    return generator

generator = load_generator()

# ------------------ Helper Function ------------------
def get_response(prompt):
    result = generator(prompt, max_new_tokens=200)
    return result[0]["generated_text"]

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Page Config ------------------
st.set_page_config(page_title="ApkaApna AI Chatbot", page_icon="🤖", layout="wide")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.title("💬 ApkaApna AI Chatbot")
    st.markdown("---")
    st.subheader("⚡ About")
    st.write(
        "Welcome to **ApkaApna AI Chatbot**, your intelligent virtual assistant "
        "designed to provide instant, accurate, and engaging responses."
    )
    st.subheader("✨ Key Highlights")
    st.markdown(
        """
        ✅ **Smart & Reliable** – Powered by FLAN-T5-Small  
        💬 **Human-like Chat** – Natural conversations  
        ⚡ **Fast & Responsive** – CPU-friendly  
        🎯 **Personalized Help** – Tailored responses  
        🔒 **Secure & Private** – Chats stay confidential  
        🌐 **Always Available** – 24/7 assistance  
        """
    )
    st.subheader("🎨 Theme")
    theme = st.radio("Choose Theme:", ["Light", "Dark"], index=0)
    st.subheader("🛠 Options")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
    st.markdown("---")
    st.caption("🚀 Developed by Ashish")

# ------------------ CSS ------------------
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.chat-container {max-width:800px;margin:auto;padding:20px;}
.msg-row {display:flex;margin:12px 0;}
.msg-row.user {justify-content:flex-end;}
.msg-row.bot {justify-content:flex-start;}
.user-msg,.bot-msg {padding:12px 16px;border-radius:18px;max-width:70%;font-size:16px;line-height:1.4;word-wrap:break-word;box-shadow:0 2px 6px rgba(0,0,0,0.1);}
.user-msg {background-color:#0d6efd;color:white;border-bottom-right-radius:5px;}
.bot-msg {background-color:#e9ecef;color:#212529;border-bottom-left-radius:5px;}
.title {text-align:center;font-size:32px;font-weight:bold;color:#0d47a1;margin-bottom:4px;}
.tagline {text-align:center;font-size:14px;color:#6c757d;margin-bottom:25px;}
.input-container {position:fixed;bottom:15px;width:80%;left:50%;transform:translateX(-50%);background:white;padding:10px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);display:flex;gap:10px;z-index:999;}
.stTextInput {flex:1;}
.stButton > button {background-color:#0d6efd;color:white;padding:0.6rem 1rem;border-radius:8px;border:none;cursor:pointer;font-weight:bold;}
.stButton > button:hover {background-color:#0b5ed7;}
</style>
""", unsafe_allow_html=True)

# ------------------ Main Chat ------------------
st.markdown('<div class="title">🤖 ApkaApna AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">“Ask anything, get instant answers – powered by AI & Developed by ABSingh”</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="msg-row user"><div class="user-msg">{msg}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-row bot"><div class="bot-msg">{msg}</div></div>', unsafe_allow_html=True)

# ------------------ Input Form ------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("💭 Type your message:", placeholder="Send a message...")
    submit_button = st.form_submit_button("Ask")

    if submit_button and user_input.strip():
        st.session_state.chat_history.append(("user", user_input))
        response = get_response(user_input)
        st.session_state.chat_history.append(("bot", response))
        st.experimental_rerun()
