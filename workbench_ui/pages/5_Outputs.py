import streamlit as st

st.set_page_config(page_title="5. Outputs | Sovereign AI", layout="wide")

st.title("5. Outputs")
st.caption("View and download AI-generated outputs.")

st.tabs(["All Outputs", "Reports", "Summaries", "Presentations", "Others"])

outputs = [
    {"file": "Market_Trends_Report.pdf", "desc": "Comprehensive report on market trends in AI.", "meta": "PDF • Generated today, 10:45 AM", "size": "2.3 MB"},
    {"file": "Document_Summary.txt", "desc": "Summary of uploaded documents.", "meta": "TXT • Generated today, 09:30 AM", "size": "0.8 MB"},
    {"file": "Key_Notes_Extracted.txt", "desc": "Extracted key notes from handwritten documents.", "meta": "TXT • Generated yesterday, 04:20 PM", "size": "1.1 MB"},
    {"file": "Project_Presentation.pptx", "desc": "Presentation based on knowledge base content.", "meta": "PPTX • Generated yesterday, 02:15 PM", "size": "5.7 MB"},
    {"file": "Analysis_Report.pdf", "desc": "In-depth analysis report.", "meta": "PDF • Generated 2 days ago, 11:00 AM", "size": "3.4 MB"}
]

for o in outputs:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"""
        <div style="background: #181c28; border: 1px solid #23293a; border-radius: 10px; padding: 12px; margin-bottom: 8px;">
            <b style="color: #f1f5f9;">📥 {o['file']}</b> — <span style="color: #64748b; font-size: 12px;">{o['size']}</span><br>
            <span style="color: #94a3b8; font-size: 13px;">{o['desc']}</span><br>
            <small style="color: #64748b;">{o['meta']}</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.button("Download", key=o['file'], use_container_width=True)