import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    st.error("Hugging Face API token not found. Add it to .env or Streamlit Secrets.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    return response.json()[0]["generated_text"]

# ------------------ Session ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="ApkaApna AI Chatbot", page_icon="🤖", layout="wide")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.title("💬 ApkaApna AI Chatbot")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# ------------------ Chat ------------------
st.markdown("## 🤖 ApkaApna AI Chatbot")
st.markdown("“Ask anything, get instant answers – powered by AI”")

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

user_input = st.text_input("💭 Type your message:", placeholder="Send a message...")
if st.button("Ask") and user_input.strip():
    st.session_state.chat_history.append(("user", user_input))
    try:
        response = query({"inputs": user_input})
    except Exception as e:
        response = "⚠️ Error generating response."
    st.session_state.chat_history.append(("bot", response))
    st.experimental_rerun()
