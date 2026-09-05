import streamlit as st
import requests

st.set_page_config(page_title="Document Upload", layout="wide")

st.title("📄 Document Upload & RAG Indexing")
st.markdown("Upload your PDF, TXT, or DOCX files to process and index them into the Knowledge Base.")

# File Uploader
uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt", "docx", "csv"])

if uploaded_file is not None:
    st.info(f"File Selected: **{uploaded_file.name}** ({uploaded_file.size} bytes)")
    
    if st.button("Upload & Process Document"):
        with st.spinner("Uploading and indexing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post("http://localhost:8000/api/upload", files=files)
                
                # 200 (OK) aur 201 (Created) dono success codes accept karein
                if response.status_code in [200, 201]:
                    st.success("✅ File uploaded and processed successfully!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Upload failed with status code {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"⚠️ Connection error: Could not reach backend server ({e})")