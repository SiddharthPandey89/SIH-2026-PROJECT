# Standard Operating Procedure
## MRPL Sovereign On-Premise Agentic AI Workbench

**Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)  
**Theme:** Smart Automation  
**Document type:** Workbench usage and governance SOP  
**Status:** Reference draft for review and approval

## 1. Purpose

This SOP defines a controlled method for using the MRPL Sovereign On-Premise Agentic AI Workbench to search internal knowledge, analyze documents, execute approved agentic tasks, and prepare draft outputs.

The procedure is designed for an offline-first environment in which sensitive organizational information is processed within MRPL-controlled infrastructure. It supports consistent use of the workbench while preserving human accountability for operational, safety, maintenance, production, and compliance decisions.

This document governs use of the AI workbench. It does not replace any MRPL process safety management procedure, permit-to-work requirement, emergency response plan, operating manual, maintenance instruction, cybersecurity policy, or statutory requirement.

## 2. Scope

This SOP applies to authorized users who use the workbench for:

- Searching approved manuals, SOPs, and past reports
- Asking questions about internal documents
- Summarizing or comparing uploaded documents
- Planning and completing approved multi-step knowledge tasks
- Preparing draft reports, notes, and other work outputs
- Reviewing the status of local services and knowledge-base operations

It applies to documents stored in approved workbench locations, including manuals, SOPs, past reports, and authorized uploads.

## 3. Roles and Responsibilities

### 3.1 Workbench User

The user shall:

- Use only an assigned or approved account.
- Submit information appropriate for the selected task and access level.
- Verify that uploaded documents are relevant, current, and approved for use.
- Review AI responses against authoritative source documents.
- Report incorrect, unsafe, incomplete, or suspicious outputs.
- Avoid treating an AI response as authorization to perform physical work.

### 3.2 Domain Reviewer

The domain reviewer shall:

- Check technical accuracy and source relevance.
- Confirm that important conclusions match current approved procedures.
- Review drafts before they are circulated or used operationally.
- Identify outdated, conflicting, or incomplete source documents.

### 3.3 Knowledge-Base Administrator

The administrator shall:

- Maintain approved document collections and folder permissions.
- Ingest, update, or remove documents according to document-control rules.
- Monitor ingestion failures and retrieval quality.
- Maintain traceability for changes to curated knowledge sources.

### 3.4 System Administrator

The system administrator shall:

- Maintain the local application, model runtime, and supporting services.
- Monitor system health, storage, access, and network state.
- Apply approved updates and security controls.
- Preserve logs and audit information according to MRPL policy.

## 4. Preconditions

Before using the workbench, confirm that:

1. You are authorized to access the workbench and the relevant document collection.
2. The local backend and required services are available.
3. The selected source documents are approved for the intended use.
4. The request does not require bypassing an established approval or control process.
5. The task is suitable for AI assistance and does not require unsupervised control of plant equipment or safety systems.

For safety-critical or time-critical situations, follow the applicable MRPL emergency or operational procedure first. Do not wait for an AI response.

## 5. Document Classification and Handling

Before uploading or adding a document, determine:

- Whether the document contains confidential, personal, security-sensitive, or proprietary information
- Whether the user is authorized to process that information
- Whether the document is current and approved
- Which collection is appropriate: manuals, SOPs, past reports, or uploads
- Whether the document contains instructions that require domain review

Do not place a document in a curated knowledge-base folder merely because it is available. Unapproved, obsolete, duplicate, corrupted, or unrelated documents should be excluded or quarantined according to local document-control rules.

## 6. Standard Operating Procedure

### 6.1 Start a Work Session

1. Open the MRPL AI workbench through the approved local interface.
2. Confirm that the application is connected to the intended local backend.
3. Review the system or network status when the task involves document ingestion or model execution.
4. Select the appropriate work area, such as Chat, Agent Tasks, Document Upload, Knowledge Base, Outputs, or System Monitor.
5. Define the task objective in one or two precise sentences.

A good task request identifies the desired output, relevant context, and important constraints. For example: "Summarize the maintenance recommendations in the approved compressor report and list the source sections used."

### 6.2 Ask a Knowledge Question

1. Identify the subject and the expected answer format.
2. Name the relevant document, equipment area, department, or reporting period when known.
3. Ask the workbench to use approved internal sources.
4. Review the response and its source references.
5. Compare important statements with the original document.
6. Record unresolved questions for a domain reviewer.

If the response has no reliable source support, mark it as unverified and do not use it as an operational instruction.

### 6.3 Upload a Document

1. Confirm the document is authorized for local processing.
2. Check the filename, format, version, and sensitivity classification.
3. Open the Document Upload area.
4. Select the correct document category.
5. Upload the file through the approved interface.
6. Wait for validation and storage to complete.
7. Check the resulting status and metadata.
8. Do not assume that a successful upload means the document is searchable.
9. Start or request ingestion when that action is available.
10. Confirm that ingestion completed successfully before relying on the document in a response.

Supported knowledge documents include Markdown, plain text, PDF, and DOCX files. Scanned or image-only PDFs may require a separate OCR process before their content can be retrieved.

### 6.4 Run an Agent Task

1. Describe the business objective and the expected final output.
2. Provide the agent with only the context required for the task.
3. Identify source collections or documents that should be used.
4. State actions that are allowed, such as searching, summarizing, comparing, or drafting.
5. State actions that are not allowed, such as sending messages, changing production data, or making operational decisions without approval.
6. Review the generated plan before execution when the interface provides a plan-review step.
7. Execute the task only within the user's authorization.
8. Review each material result and source reference.
9. Save or export the output only after checking its accuracy and sensitivity.

Agent tasks must remain advisory unless a separately approved integration authorizes a specific automated action.

### 6.5 Generate and Review an Output

1. Select the required output type and audience.
2. Confirm that the source material is sufficient and current.
3. Generate a draft using the approved workbench.
4. Check names, dates, units, calculations, document versions, and source references.
5. Check for unsupported conclusions, missing assumptions, and conflicting source statements.
6. Remove unnecessary sensitive information.
7. Submit the draft to the responsible domain reviewer.
8. Apply required approvals before distribution or operational use.
9. Store the approved version in the designated controlled location.

The generated output must be clearly distinguished from an approved MRPL record until the relevant review and approval process is complete.

## 7. Verification Checklist

Before accepting an AI-assisted answer or report, verify:

- The response answers the requested question.
- The sources are relevant to the task.
- The cited source content actually supports the response.
- The source documents are current and approved.
- Important numbers, dates, units, and technical terms are correct.
- The response does not invent facts, approvals, incidents, or measurements.
- Uncertainty and missing information are clearly stated.
- The output does not disclose information to an unauthorized audience.
- A qualified person has reviewed any safety-critical or operational content.

## 8. Handling Errors and Unexpected Results

### 8.1 No Answer or Irrelevant Answer

- Check the spelling and specificity of the request.
- Confirm that the expected document was uploaded and ingested.
- Identify whether the document is in the correct collection.
- Try a narrower question with the document name or section title.
- Escalate to the knowledge-base administrator if the document is missing or retrieval remains unreliable.

### 8.2 Unsupported or Uncertain Answer

- Treat the answer as unverified.
- Do not fill gaps using assumptions.
- Consult the current approved manual, SOP, or responsible expert.
- Record the question and the source gap for later knowledge-base improvement.

### 8.3 Upload or Ingestion Failure

- Confirm that the file type is supported.
- Confirm that the file is readable and not corrupted.
- Check whether the file is empty, encrypted, or image-only.
- Preserve the original file and error details for the administrator.
- Do not repeatedly upload sensitive files as a workaround without guidance.

### 8.4 Suspicious or Unsafe Output

- Stop using the output immediately.
- Do not execute the suggested action.
- Preserve the prompt, response, source references, and relevant timestamps.
- Notify the domain reviewer and system administrator through the approved channel.
- Use the authoritative MRPL procedure for the underlying task.

## 9. Security and Privacy Controls

- Keep prompts, documents, and outputs within approved local systems.
- Do not copy confidential content into unapproved external tools or services.
- Do not share credentials, access tokens, or private system information in prompts.
- Use the minimum necessary information for each task.
- Lock or sign out of the workbench when leaving the workstation.
- Report unauthorized access, unexpected network activity, or data leakage immediately.
- Follow MRPL retention, classification, access-control, and audit requirements.

## 10. Audit and Record Keeping

For significant tasks, retain the information required by local policy, such as:

- User or responsible team
- Task objective and date
- Source documents and versions used
- Agent or model used, where recorded by the system
- Generated output and review status
- Reviewer or approver
- Exceptions, corrections, and unresolved limitations

Do not retain duplicate copies of sensitive documents outside controlled storage unless required by an approved process.

## 11. Review and Maintenance of This SOP

This SOP should be reviewed when there is a material change to the workbench, model behavior, document-ingestion process, security requirements, or MRPL governance policy. The document owner should record the revision date, reviewer, approval status, and summary of changes in the controlled version of this document.

## 12. Important Limitation

The MRPL Sovereign On-Premise Agentic AI Workbench is an assistance and automation platform. It does not replace qualified personnel, approved operating procedures, engineering calculations, formal permits, emergency instructions, or management authorization. Human review remains mandatory wherever an AI-assisted result could affect safety, production, maintenance, compliance, security, or the environment.
