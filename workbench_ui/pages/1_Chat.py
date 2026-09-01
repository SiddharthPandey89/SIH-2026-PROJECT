import streamlit as st
from components.chat_widget import render_chat_message

st.set_page_config(page_title="1. Chat | Sovereign AI", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0c0e14; color: #f1f5f9; }
    .user-bubble { background-color: #181c28; border: 1px solid #23293a; padding: 14px; border-radius: 10px; margin-bottom: 12px; }
    .ai-bubble { background-color: #121621; border: 1px solid #1e2433; padding: 14px; border-radius: 10px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.title("1. Chat")
    st.caption("Chat with your AI assistant securely on-premises.")
with col_h2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {"role": "user", "time": "10:30 AM", "content": "Explain the architecture of a transformer model in simple terms."},
        {"role": "assistant", "time": "10:30 AM", "content": "A transformer model is a type of neural network architecture that relies on self-attention mechanisms to process input data. It consists of an encoder and a decoder. The encoder processes the input sequence and creates contextual embeddings, while the decoder generates output sequences. Transformers are highly parallelizable and effective for tasks like translation, summarization, and more."}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <span style="color: #6366f1; font-weight: bold;">🧑‍💻 You</span> <span style="float: right; color: #64748b; font-size: 12px;">{msg.get('time', '')}</span><br><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-bubble">
            <span style="color: #38bdf8; font-weight: bold;">🤖 AI Assistant</span> <span style="float: right; color: #64748b; font-size: 12px;">{msg.get('time', '')}</span><br><br>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "time": "Just now", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "time": "Just now", "content": f"Processed locally on-device: '{user_input}'"})
    st.rerun()