import streamlit as st

st.set_page_config(page_title="4. Knowledge Base | Sovereign AI", layout="wide")

st.title("4. Knowledge Base")
st.caption("Browse and search your indexed documents.")

st.text_input("🔍 Search documents...", placeholder="Type keywords to query local vector store...")

docs = [
    {"title": "AI_Research_Paper.pdf", "desc": "This paper discusses the advancements in AI...", "tag": "AI", "time": "PDF • Uploaded Just now"},
    {"title": "Handwritten_Notes.jpg", "desc": "These are the notes from the lecture on...", "tag": "Notes", "time": "JPG • Uploaded 5 mins ago"},
    {"title": "Meeting_Summary.txt", "desc": "Summary of the project meeting held on...", "tag": "Meeting", "time": "TXT • Uploaded 1 hour ago"},
    {"title": "Project_Overview.pdf", "desc": "This document provides an overview of the...", "tag": "Project", "time": "PDF • Uploaded yesterday"},
    {"title": "Design_Diagram.png", "desc": "System architecture diagram.", "tag": "Design", "time": "PNG • Uploaded yesterday"}
]

for d in docs:
    st.markdown(f"""
    <div style="background: #181c28; border: 1px solid #23293a; border-radius: 10px; padding: 14px; margin-bottom: 10px;">
        <span style="color: #6366f1; font-weight: bold;">📑 {d['title']}</span>
        <span style="float: right; background: #6366f122; color: #818cf8; padding: 2px 8px; border-radius: 10px; font-size: 11px;">{d['tag']}</span>
        <p style="color: #94a3b8; font-size: 13px; margin: 4px 0;">{d['desc']}</p>
        <small style="color: #64748b;">{d['time']}</small>
    </div>
    """, unsafe_allow_html=True)