import streamlit as st

st.set_page_config(page_title="6. System Monitor | Sovereign AI", layout="wide")

st.title("6. System Monitor")
st.caption("Monitor system resources and network activity in real-time.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("CPU Usage", "43%", "Normal")
c2.metric("RAM Usage", "58%", "Stable")
c3.metric("GPU Usage", "41%", "Active")
c4.metric("Disk Usage", "55%", "Healthy")

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Network Connections")
    st.markdown("""
    | Local Address | Remote Address | Status | Protocol |
    | :--- | :--- | :--- | :--- |
    | `127.0.0.1:50532` | `127.0.0.1:8000` | <span style='color:#10b981;'>ESTABLISHED</span> | TCP |
    | `127.0.0.1:50533` | `127.0.0.1:5432` | <span style='color:#10b981;'>ESTABLISHED</span> | TCP |
    | `127.0.0.1:50534` | `127.0.0.1:6379` | <span style='color:#10b981;'>ESTABLISHED</span> | TCP |
    | `127.0.0.1:50535` | `127.0.0.1:5000` | <span style='color:#3b82f6;'>LISTENING</span> | TCP |
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("System Information")
    st.markdown("""
    - **Hostname**: `Sovereign-AI-Local`
    - **OS**: `Windows 11 Pro / Linux`
    - **Python Version**: `3.11.6`
    - **Uptime**: `2h 45m 33s`
    - **Local IP**: `127.0.0.1`
    """)