import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="2. Agent Tasks | Sovereign AI Workbench",
    page_icon="🤖",
    layout="wide"
)

# Header Section
st.title("2. Agent Tasks")
st.caption("Create and manage multi-step tasks for your AI agent.")

st.markdown("---")

# Section 1: Create New Task Form
with st.expander("➕ **Create New Multi-Step Task**", expanded=False):
    with st.form("new_task_form"):
        task_title = st.text_input("Task Title", placeholder="e.g., Analyze Maintenance Report & Draft Summary")
        task_description = st.text_area("Task Description & Instructions", placeholder="Describe step-by-step instructions for the agent...")
        
        col1, col2 = st.columns(2)
        with col1:
            task_priority = st.selectbox("Priority", ["Normal", "High", "Critical"])
        with col2:
            output_format = st.selectbox("Expected Output Format", ["Word Document (.docx)", "PowerPoint (.pptx)", "Excel Sheet (.xlsx)", "Plain Text"])
            
        submit_btn = st.form_submit_button("🚀 Submit Task to Agent")
        
        if submit_btn:
            if task_title and task_description:
                st.success(f"Task '{task_title}' submitted successfully! Agent execution started.")
            else:
                st.warning("Please fill in both the task title and description.")

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Task Status Tabs
tab_active, tab_completed, tab_failed = st.tabs(["⏳ Active Tasks", "✅ Completed", "❌ Failed"])

# ----------------------------
# 1. ACTIVE TASKS TAB
# ----------------------------
with tab_active:
    st.markdown("### ⏳ Currently Running & Pending Tasks")
    st.caption("Tasks currently being processed by the local AI Agent.")
    
    # Active Task 1 Card
    with st.container(border=True):
        col_title, col_status = st.columns([4, 1])
        with col_title:
            st.markdown("### 📋 Research Market Trends")
            st.write("Search for the latest market trends in AI and generate a summary report.")
        with col_status:
            st.warning("🔄 In Progress")
            
        with st.expander("🔍 View Live Agent Logs & Steps"):
            st.code("""
[10:14:02] Step 1: Initializing Planner Engine (Mistral-7B)...
[10:14:05] Step 2: Searching local Knowledge Base for past market reports...
[10:14:12] Step 3: Extracting key findings from 4 matching documents...
[10:14:20] Step 4: Generating draft summary report (.docx)...
            """, language="text")
            st.progress(75, text="Execution Progress: 75%")

    # Active Task 2 Card
    with st.container(border=True):
        col_title, col_status = st.columns([4, 1])
        with col_title:
            st.markdown("### 📋 Summarize Uploaded Documents")
            st.write("Read uploaded inspection PDFs and create a concise approval summary note.")
        with col_status:
            st.info("🕒 Pending")
            
        with st.expander("🔍 View Live Agent Logs & Steps"):
            st.code("""
[10:20:00] Step 1: Task queued in local database.
[10:20:01] Waiting for GPU resources...
            """, language="text")
            st.progress(10, text="Execution Progress: 10%")

# ----------------------------
# 2. COMPLETED TASKS TAB
# ----------------------------
with tab_completed:
    st.markdown("### ✅ Successfully Completed Tasks")
    st.caption("Tasks finished by the agent with output files generated.")
    
    # Completed Task 1 Card
    with st.container(border=True):
        col_title, col_status = st.columns([4, 1])
        with col_title:
            st.markdown("### 📄 Generate Board Presentation")
            st.write("Extract quarterly safety statistics and generate a 10-slide PowerPoint presentation.")
        with col_status:
            st.success("✅ Completed")
            
        with st.expander("📥 Download Output File & Summary"):
            st.write("**Completed On:** Today at 09:45 AM")
            st.write("**Model Used:** Mistral-7B + python-pptx")
            st.download_button(
                label="📥 Download presentation_q3.pptx",
                data=b"Dummy content for presentation",
                file_name="presentation_q3.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )

    # Completed Task 2 Card
    with st.container(border=True):
        col_title, col_status = st.columns([4, 1])
        with col_title:
            st.markdown("### 📝 Extract Key Notes & Hand-written Logs")
            st.write("Extract handwritten notes from scanned maintenance log sheets and save as structured text.")
        with col_status:
            st.success("✅ Completed")
            
        with st.expander("📥 Download Output File & Summary"):
            st.write("**Completed On:** Yesterday at 04:30 PM")
            st.write("**Model Used:** Surya OCR + DeepSeek-Coder")
            st.download_button(
                label="📥 Download extracted_notes.docx",
                data=b"Dummy content for docx",
                file_name="extracted_notes.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# ----------------------------
# 3. FAILED TASKS TAB
# ----------------------------
with tab_failed:
    st.markdown("### ❌ Failed or Interrupted Tasks")
    st.caption("Tasks that encountered errors during execution.")
    
    # Failed Task 1 Card
    with st.container(border=True):
        col_title, col_status = st.columns([4, 1])
        with col_title:
            st.markdown("### ⚠️ External API Scraping")
            st.write("Attempted to connect to external web server for updated specs.")
        with col_status:
            st.error("❌ Failed")
            
        with st.expander("🛠️ View Failure Reason & Error Log"):
            st.error("Error: Air-gapped isolation policy active. External network connections are strictly blocked (127.0.0.1 enforced).")
            st.button("🔄 Retry Task Offline", key="retry_btn_1")