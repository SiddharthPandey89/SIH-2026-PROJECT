# SIH Project Data Report

## 1. Project Overview

This project is a local, offline-first AI workbench designed for confidential industrial and enterprise use. The system supports:

- chat-based interaction with local models
- document upload and storage
- knowledge-base ingestion and retrieval
- conversation history persistence
- local model routing and task classification
- secure/offline operation assumptions

The project is structured around a backend service, document storage, local model assets, and UI automation for monitoring and interactions.

---

## 2. Data Inventory

### 2.1 Conversation and Chat Data

The backend stores user and assistant chat history in a local SQL database.

Primary storage:
- `data/workbench.db`

ORM models are defined in:
- `backend/database/db_models.py`

This database includes:
- `conversations`: conversation metadata such as id, title, timestamps
- `messages`: message text, sender role, model metadata, task type, and optional extra metadata

Data captured in chat history includes:
- user queries
- assistant responses
- conversation IDs and timestamps
- model used for generation
- task classification labels such as `chat`, `document_qa`, `summarization`

### 2.2 Uploaded Documents

Uploaded files are stored locally under:
- `data/uploads/`

Subfolders include:
- `data/uploads/scanned_pdfs/`
- `data/uploads/images/`
- `data/uploads/drawings/`

Uploaded files are kept with metadata sidecar JSON files that record:
- file ID
- original filename
- content type
- category
- size in bytes
- upload timestamp

This is implemented through:
- `backend/api/routes_upload.py`

### 2.3 Knowledge Base Documents

Curated, searchable local documents are stored under:
- `data/documents/manuals/`
- `data/documents/sops/`
- `data/documents/past_reports/`

These are used for retrieval-augmented generation (RAG) and local knowledge search.

### 2.4 Local Vector and Semantic Search Data

The project includes a local vector knowledge store and embedding environment:
- `data/knowledge_base/`
- `models/embedding/bge-m3/`

This supports:
- chunked document indexing
- semantic similarity search
- grounding for model responses using relevant document snippets

### 2.5 Model Assets

The system relies on local model artifacts stored under:
- `models/llm/`
- `models/embedding/`
- `models/vision/`

The model areas include:
- Mistral 7B
- BGE-M3 embedding models
- vision/OCR support components

These are local assets and are intended to avoid dependency on external cloud APIs.

### 2.6 Generated Outputs

The project includes output directories for generated artifacts:
- `data/outputs/`
- `sandbox/temp_outputs/`

These likely hold generated reports, intermediate files, or sandbox results.

---

## 3. Data Lifecycle

### 3.1 Upload Flow
1. A file is uploaded through the upload API.
2. The file is validated for type and size.
3. The file is stored under the correct upload subfolder.
4. Metadata is written to a companion `.meta.json` file.
5. The file becomes available for later ingestion or indexing.

### 3.2 Knowledge Ingestion Flow
1. A file path or upload file ID is supplied.
2. The file is validated against allowed local roots.
3. Content is extracted depending on file type.
4. Text is normalized and chunked.
5. Chunked content is prepared for retrieval or vector indexing.

### 3.3 Chat Flow
1. User prompt is received from the API.
2. The task is classified.
3. Relevant context is retrieved from documents or uploads when needed.
4. The local model generates a response.
5. The message is persisted in conversation history.
6. The answer and source references are returned to the user.

### 3.4 Storage and Retrieval Pattern

The project uses a hybrid pattern:
- SQL database for chat metadata/history
- filesystem storage for documents and uploads
- local vector store for semantic search
- keyword fallback retrieval when vector backend is unavailable

---

## 4. Data Types and Sensitivity

### 4.1 Structured Data
- conversation metadata
- message records
- model names and task labels
- source references
- uploaded file metadata

### 4.2 Unstructured Data
- PDFs
- images
- text documents
- markdown files
- scanned documents
- documents from manuals and reports

### 4.3 Sensitive Data Considerations

Because this is intended for confidential industrial use, the data may include:
- internal operation procedures
- safety documentation
- maintenance records
- manuals and plant process information
- uploaded company documents
- possibly proprietary knowledge

This makes local storage, access control, and offline processing critical design principles.

---

## 5. Data Security and Compliance Observations

The project itself strongly emphasizes local/offline processing and avoids cloud dependence.

Observed controls and patterns:
- local SQLite database configuration by default
- local file storage instead of external services
- model loading from local folders
- environment-variable configuration for directories and endpoints
- validation of upload type and size
- path restrictions for document ingestion
- local vector store usage for RAG

Important security considerations:
- access control is not fully described in the visible project code
- uploaded files may contain confidential or sensitive business information
- if the system is deployed in a shared environment, document access should be restricted
- retention and deletion policies should be explicitly defined

---

## 6. Data Storage Map

| Area | Purpose | Data Type |
|---|---|---|
| `data/workbench.db` | chat/conversation persistence | structured database |
| `data/uploads/` | uploaded files | documents/images/PDFs |
| `data/documents/` | curated knowledge base | manuals, SOPs, reports |
| `data/knowledge_base/` | vector storage | embeddings and searchable content |
| `models/` | local AI model files | binaries and weights |
| `data/outputs/` | generated results | reports/artifacts |
| `sandbox/temp_outputs/` | temporary generated outputs | ephemeral artifacts |

---

## 7. Key Data Risks

1. Sensitive internal documents may be stored without explicit retention enforcement.
2. Unstructured uploaded files may contain unsupported or malicious content if validation is insufficient.
3. A local database and filesystem model can still expose data if the host environment is shared.
4. Retrieval and indexing can include stale or outdated information if document updates are not managed.
5. Long-term data governance is not clearly documented in the visible repository.

---

## 8. Recommendations

- Define a formal data retention policy for uploads and generated outputs.
- Add clear user access rules for confidential documents.
- Maintain a document versioning strategy for `data/documents/` and uploads.
- Track document lineage between uploads, ingestion, and retrieval events.
- Add audit logs for who accessed or indexed which document.
- Define deletion and archival rules for outdated knowledge content.
- Document the exact data categories and owner approvals for sensitive use cases.

---

## 9. Summary

The SIH project is a local AI workbench with a clear, practical data architecture built around:

- SQLite-based conversation storage
- local filesystem document storage
- local knowledge retrieval and vector search
- offline model execution from local model folders
- ingestion pipelines for uploaded and curated documents

It is designed for privacy-first operation and is suitable for secure internal deployment, but it would benefit from a more explicit data governance, retention, and access-control policy.

---

## 10. Report Source Basis

This report was compiled from the project structure and the visible backend implementation, including:

- `README.md`
- `backend/main.py`
- `backend/database/db_models.py`
- `backend/database/crud.py`
- `backend/database/init_db.py`
- `backend/api/routes_upload.py`
- `backend/api/routes_chat.py`
- `backend/api/routes_knowledge.py`
- `backend/knowledge_base/retriever.py`
- `backend/knowledge_base/ingestion.py`
- `backend/knowledge_base/vector_store.py`
- `requirements.txt`
