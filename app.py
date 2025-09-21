# ------------------ Imports ------------------
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ------------------ Model Setup ------------------
MODEL_NAME = "tiiuae/falcon-7b-instruct"  # CPU-friendly

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",          # CPU mode
        torch_dtype=torch.float16,  # lower RAM usage
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200
    )
    return generator

generator = load_model()

# ------------------ Helper Function ------------------
def get_response(prompt):
    result = generator(prompt)
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
        ✅ **Smart & Reliable** – Accurate answers powered by Falcon-7B  
        💬 **Human-like Chat** – Natural and engaging conversations  
        ⚡ **Fast & Responsive** – Quick replies (~1-2s per query)  
        🎯 **Personalized Help** – Tailored responses just for you  
        🔒 **Secure & Private** – Your chats stay safe and confidential  
        🌐 **Always Available** – 24/7 assistance, anytime you need  
        """
    )
    st.subheader("🎨 Theme")
    theme = st.radio("Choose Theme:", ["Light", "Dark"], index=0)
    st.subheader("🛠 Options")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
    st.markdown("---")
    st.caption("🚀 Developed by Ashish")

# ------------------ CSS for Chat Style ------------------
if theme == "Light":
    st.markdown("""
    <style>
    body { background-color: #f7f8fa; font-family: 'Segoe UI', sans-serif; }
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
else:
    st.markdown("""
    <style>
    body { background-color: #121212; font-family: 'Segoe UI', sans-serif; color:white; }
    .chat-container {max-width:800px;margin:auto;padding:20px;}
    .msg-row {display:flex;margin:12px 0;}
    .msg-row.user {justify-content:flex-end;}
    .msg-row.bot {justify-content:flex-start;}
    .user-msg,.bot-msg {padding:12px 16px;border-radius:18px;max-width:70%;font-size:16px;line-height:1.4;word-wrap:break-word;box-shadow:0 2px 6px rgba(0,0,0,0.4);}
    .user-msg {background-color:#00b894;color:white;border-bottom-right-radius:5px;}
    .bot-msg {background-color:#2d2d2d;color:white;border-bottom-left-radius:5px;}
    .title {text-align:center;font-size:32px;font-weight:bold;color:#00e5ff;margin-bottom:4px;}
    .tagline {text-align:center;font-size:14px;color:#b0bec5;margin-bottom:25px;}
    .input-container {position:fixed;bottom:15px;width:80%;left:50%;transform:translateX(-50%);background:#1e1e2f;padding:10px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.5);display:flex;gap:10px;z-index:999;}
    .stTextInput {flex:1;color:white;}
    .stButton > button {background-color:#00b894;color:white;padding:0.6rem 1rem;border-radius:8px;border:none;cursor:pointer;font-weight:bold;}
    .stButton > button:hover {background-color:#00997b;}
    </style>
    """, unsafe_allow_html=True)

# ------------------ Main Chat Area ------------------
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
