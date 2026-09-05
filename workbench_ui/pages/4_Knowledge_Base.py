import streamlit as st
import requests

st.set_page_config(page_title="Knowledge Base", layout="wide")

st.title("📚 Knowledge Base Management")
st.markdown("View and manage indexed documents in the local Vector Database.")

st.subheader("Indexed Documents")

# Backend se documents fetch karne ka layout
try:
    response = requests.get("http://localhost:8000/api/documents")
    if response.status_code == 200:
        docs = response.json()
        if docs:
            st.dataframe(docs, use_container_width=True)
        else:
            st.info("No documents indexed yet. Upload files from the 'Document Upload' page.")
    else:
        st.info("Knowledge Base ready. Upload documents to populate vector indices.")
except Exception as e:
    st.warning("Could not fetch document list from backend. Showing local status.")

st.divider()
st.subheader("Vector Store Info")
col1, col2, col3 = st.columns(3)
col1.metric(label="Embedding Model", value="BAAI/bge-small-en-v1.5")
col2.metric(label="Vector Store", value="ChromaDB (Local)")
col3.metric(label="Chunk Size", value="512 tokens")