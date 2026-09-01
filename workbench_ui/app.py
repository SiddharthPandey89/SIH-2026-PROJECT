import streamlit as st

st.set_page_config(
    page_title="Sovereign AI Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global Dark Theme Matching Mockup Design
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0c0e14 !important;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Sleek Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121621 !important;
        border-right: 1px solid #1e2433;
    }

    /* Hide Default Header & Streamlit Navigation Artifacts */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Custom Input Styling */
    .stTextInput input, .stTextArea textarea {
        background-color: #181c28 !important;
        border: 1px solid #23293a !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Navigation Control (Hides extra "app" page automatically)
pages = [
    st.Page("pages/1_Chat.py", title="1. Chat", icon="💬", default=True),
    st.Page("pages/2_Agent_Tasks.py", title="2. Agent Tasks", icon="👥"),
    st.Page("pages/3_Document_Upload.py", title="3. Document Upload", icon="📄"),
    st.Page("pages/4_Knowledge_Base.py", title="4. Knowledge Base", icon="📚"),
    st.Page("pages/5_Outputs.py", title="5. Outputs", icon="📂"),
    st.Page("pages/6_System_Monitor.py", title="6. System Monitor", icon="🖥️"),
]

pg = st.navigation(pages)

# Custom Sidebar Footer (Matching exact bottom UI in photo)
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #181c28; padding: 14px; border-radius: 10px; border: 1px solid #23293a; margin-top: auto;">
        <span style="color: #10b981; font-weight: bold; font-size: 13px;">Local Mode ●</span>
        <p style="color: #94a3b8; font-size: 11px; margin: 4px 0 0 0; line-height: 1.3;">
            All processing happens on this machine. No data leaves your system.
        </p>
    </div>
    """, unsafe_allow_html=True)

pg.run()