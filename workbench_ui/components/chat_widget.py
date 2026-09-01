import streamlit as st

def render_chat_message(role, content, model_used=None):
    """User aur AI assistant ke messages ko stylish styling me render karta hai."""
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            if model_used:
                st.caption(f"🤖 Powered by: **{model_used}** | 🔒 *Air-Gapped Local Model*")
            st.write(content)