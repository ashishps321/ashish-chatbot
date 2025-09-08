# main imports
import os
import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# ------------------ Model Setup ------------------
REPO_ID = "TheBloke/llama-2-7b-GGML"   # Change this if you want another model
MODEL_FILE = "llama-2-7b.ggmlv3.q4_0.bin"

st.info("⏳ Downloading LLaMA model from Hugging Face Hub (first run may take a while)...")

try:
    MODEL_PATH = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE
    )
    st.success("✅ Model downloaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to download model from Hugging Face Hub.\nError: {e}")
    st.stop()

# ------------------ Initialize LLaMA ------------------
llm = Llama(model_path=MODEL_PATH)

def get_llama_response(prompt: str):
    response = llm(prompt, max_tokens=256)
    return response['choices'][0]['text']

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="AI Chatbot by ASHISH", page_icon="🤖", layout="wide")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)
    st.title("💬 Ashish's AI Chatbot")
    st.markdown("---")
    st.subheader("⚡ About")
    st.write(
        """
        Welcome to **Ashish’s AI Chatbot**, powered by **LLaMA** model from Hugging Face Hub.  
        Instant, accurate, and private AI responses.  

        **Key Highlights**  
        - 💡 Insightful Responses  
        - 🗣️ Natural Conversation  
        - 🎯 Personalized Interaction  
        - ⚡ Fast & Efficient  
        """
    )
    st.subheader("🎨 Theme")
    theme = st.radio("Choose theme:", ["Light", "Dark"], index=0)
    st.subheader("🛠 Options")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    st.markdown("---")
    st.caption("Developed by Ashish")

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Main Chat Area ------------------
st.markdown('<div class="title">🤖 Ashish\'s AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">“Your AI companion for knowledge and conversation.”</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"""
                <div class="msg-row user">
                    <div class="user-msg">{msg}</div>
                    <img src="https://cdn-icons-png.flaticon.com/512/1946/1946429.png" class="msg-avatar">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="msg-row bot">
                    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" class="msg-avatar">
                    <div class="bot-msg">{msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Input at bottom
    user_input = st.text_input("💭 Type your message:", key="input", placeholder="Ask me anything...")
    if st.button(" Submit ", use_container_width=True):
        if user_input.strip():
            st.session_state.chat_history.append(("user", user_input))
            response = get_llama_response(user_input)
            st.session_state.chat_history.append(("bot", response))
            st.experimental_rerun()
        else:
            st.warning("⚠️ Please enter a question.")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Apply Themes ------------------
if theme == "Light":
    st.markdown(
        """
        <style>
        body { background: linear-gradient(135deg, #f9f9f9 0%, #e3f2fd 100%); font-family: 'Segoe UI', sans-serif; }
        .chat-container { max-width: 750px; margin: auto; padding: 20px; }
        .user-msg, .bot-msg { padding: 12px 16px; border-radius: 15px; margin: 10px 0; font-size: 16px; display: inline-block; max-width: 80%; backdrop-filter: blur(8px); box-shadow: 0px 4px 12px rgba(0,0,0,0.1); transition: transform 0.2s ease-in-out; }
        .user-msg:hover, .bot-msg:hover { transform: scale(1.02); }
        .user-msg { background: rgba(72, 187, 120, 0.9); color: white; text-align: right; }
        .bot-msg { background: rgba(255, 255, 255, 0.7); color: #2c3e50; text-align: left; }
        .msg-row { display: flex; align-items: flex-start; margin-bottom: 10px; }
        .msg-row.user { justify-content: flex-end; }
        .msg-avatar { width: 42px; height: 42px; border-radius: 50%; margin: 0 8px; border: 2px solid #ddd; }
        .title { text-align: center; font-size: 34px; font-weight: bold; color: #0d47a1; margin-bottom: 5px; text-shadow: 1px 1px 2px rgba(0,0,0,0.15); }
        .tagline { text-align: center; font-size: 16px; font-style: italic; color: #546e7a; margin-bottom: 25px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body { background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%); font-family: 'Segoe UI', sans-serif; color: white; }
        .chat-container { max-width: 750px; margin: auto; padding: 20px; }
        .user-msg, .bot-msg { padding: 12px 16px; border-radius: 15px; margin: 10px 0; font-size: 16px; display: inline-block; max-width: 80%; backdrop-filter: blur(8px); box-shadow: 0px 4px 12px rgba(0,0,0,0.4); transition: transform 0.2s ease-in-out; }
        .user-msg:hover, .bot-msg:hover { transform: scale(1.02); }
        .user-msg { background: rgba(0, 200, 83, 0.9); color: white; text-align: right; }
        .bot-msg { background: rgba(33, 33, 33, 0.8); color: #f1f1f1; text-align: left; }
        .msg-row { display: flex; align-items: flex-start; margin-bottom: 10px; }
        .msg-row.user { justify-content: flex-end; }
        .msg-avatar { width: 42px; height: 42px; border-radius: 50%; margin: 0 8px; border: 2px solid #444; }
        .title { text-align: center; font-size: 34px; font-weight: bold; color: #00e5ff; margin-bottom: 5px; text-shadow: 1px 1px 3px rgba(0,0,0,0.6); }
        .tagline { text-align: center; font-size: 16px; font-style: italic; color: #b0bec5; margin-bottom: 25px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
