import streamlit as st

st.set_page_config(page_title="3. Document Upload | Sovereign AI", layout="wide")

st.title("3. Document Upload")
st.caption("Upload documents to build your knowledge base.")

c1, c2 = st.columns([2, 1])

with c1:
    uploaded = st.file_uploader("Drag & drop files here or browse", type=["pdf", "png", "jpg", "txt", "md"], accept_multiple_files=True)
    st.caption("Supported formats: PDF, PNG, JPG, JPEG, TXT, MD | Max file size: 100MB")

with c2:
    st.subheader("Recent Uploads")
    recent = [
        ("AI_Research_Paper.pdf", "2.4 MB • PDF", "Just now"),
        ("Handwritten_Notes.jpg", "1.8 MB • JPG", "5 mins ago"),
        ("Meeting_Summary.txt", "0.6 MB • TXT", "1 hour ago"),
        ("Project_Overview.pdf", "3.1 MB • PDF", "Yesterday"),
        ("Design_Diagram.png", "2.7 MB • PNG", "Yesterday")
    ]
    for r in recent:
        st.markdown(f"""
        <div style="background: #181c28; border: 1px solid #23293a; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
            <b style="color: #f1f5f9; font-size: 13px;">📄 {r[0]}</b><br>
            <span style="color: #64748b; font-size: 11px;">{r[1]}</span>
            <span style="float: right; color: #64748b; font-size: 11px;">{r[2]}</span>
        </div>
        """, unsafe_allow_html=True)