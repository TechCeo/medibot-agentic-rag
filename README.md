# MediBot

MediBot is an AI-powered symptom checker that combines structured medical data processing, local retrieval, specialized LangChain tools, a LangGraph ReAct orchestrator, conversational memory, safety fallbacks, and a Gradio web interface.

The application is designed as educational medical decision support. It is not a replacement for a licensed clinician.

## System Architecture Overview

MediBot is organized as a staged pipeline that turns structured symptom datasets into an interactive multi-agent medical assistant.

### 1. Data Ingestion and Analytics

The source CSV files live in `Medibot_dataset/`:

- `dataset.csv`: disease-to-symptom mapping in wide format.
- `Symptom-severity.csv`: symptom severity weights.
- `symptom_Description.csv`: disease descriptions.
- `symptom_precaution.csv`: disease-specific precautionary guidance.

`src/data_pipeline.py` loads these files, cleans inconsistent spacing and symptom formatting, normalizes disease/symptom names, converts the wide symptom table into disease-symptom records, joins severity scores, and creates disease profiles with symptoms, descriptions, severity statistics, and precautions.

It also generates statistical analytics in `analytics/`, including:

- symptom frequency distribution
- disease symptom counts
- severity score distribution
- symptom frequency/severity correlation
- disease symptom overlap matrix
- disease overlap heatmap
- JSON/CSV summaries for reviewer inspection

### 2. FAISS Vector Store

`src/vector_service.py` builds retrieval documents from the processed disease profiles. Each disease contributes:

- a disease profile document
- a symptom profile document
- individual symptom evidence records

The vector layer uses a local `TfidfVectorizer` embedding backend and stores normalized vectors in a FAISS `IndexFlatIP` index. The persisted retrieval artifacts live in `vector_store/`:

- `medibot.faiss`
- `tfidf_vectorizer.pkl`
- `documents.json`
- `manifest.json`

The retrieval utility `search_medical_records(query, top_k=5)` accepts raw natural language symptoms such as:

```text
throbbing headache and light sensitivity
```

and returns ranked medical records with similarity scores and metadata. Query expansion and symptom aliases improve recall for natural expressions such as `light sensitivity`, `shortness of breath`, and `sharp pain in my chest`.

### 3. Specialized Agent Tools

`src/agents.py` defines the four specialized medical agents and registers them as LangChain tools with `@tool`:

- `disease_diagnosis_agent`
  - Uses the FAISS retrieval service to rank probable diseases from symptoms.

- `symptom_severity_agent`
  - Uses processed severity weights to assign `Low`, `Medium`, `High`, or `Emergency` urgency.

- `disease_description_agent`
  - Retrieves patient-friendly disease explanations from the local knowledge base.

- `precaution_advisor_agent`
  - Retrieves dataset-backed precautionary and self-care guidance for a matched condition.

The same module also exposes plain Python functions for isolated testing:

- `diagnose_disease`
- `assess_symptom_severity`
- `describe_disease`
- `advise_precautions`

### 4. Conversational Memory

`src/memory.py` implements `MediBotMemoryManager`, which stores per-session chat history and extracted symptom state using LangChain `InMemoryChatMessageHistory`.

This allows follow-up turns such as:

```text
I also have a fever and nausea.
```

to be combined with previously mentioned symptoms, so retrieval and triage operate on the full conversation context rather than only the latest message.

### 5. LangGraph ReAct Orchestrator

`src/orchestrator.py` defines `MediBotReActOrchestrator`, which initializes a LangGraph prebuilt ReAct agent with:

- `create_react_agent(...)`
- the `REGISTERED_TOOLS` list from `src/agents.py`
- a medical safety system prompt
- Stage 3 conversational memory integration

The orchestrator lets the model reason over the tool docstrings and dynamically choose which tools to call. For example:

```text
I have a sharp pain in my chest, what should I do?
```

triggers:

1. `disease_diagnosis_agent`
2. `symptom_severity_agent`
3. `precaution_advisor_agent`

A follow-up such as:

```text
Can you explain what this might be and precautions I should take?
```

uses the prior likely disease and calls:

1. `disease_diagnosis_agent`
2. `symptom_severity_agent`
3. `disease_description_agent`
4. `precaution_advisor_agent`

The default model is a deterministic local tool-calling model for repeatable offline verification. The orchestrator is structured so a production tool-calling chat model can be injected later.

### 6. Safety Fallbacks and Disclaimer Guardrails

Stage 5 safety behavior is implemented in `src/orchestrator.py`.

Before the ReAct graph runs, MediBot checks whether the user query is medical, symptom-related, or supported by existing medical conversation context. Clearly non-medical or ambiguous out-of-scope prompts are rejected before any diagnostic tools run.

Example:

```text
How do I fix a broken car engine?
```

returns a safe fallback:

```text
I can only help with medical symptom-checking questions, condition explanations, severity triage, and dataset-backed precautions. I cannot help with that request.
```

Every final response, including valid medical answers and fallback responses, includes the mandatory clinical disclaimer:

```text
Disclaimer: MediBot is an AI assistant, not a licensed medical professional. Always consult a doctor for urgent health concerns.
```

### 7. Gradio UI

`src/app.py` provides the interactive Gradio interface. It includes:

- a conversational `gr.Chatbot`
- a text input for user symptoms/questions
- a visible Thought/Action Log showing ReAct tool routing
- per-browser-session state isolation using unique session IDs
- environment-variable configuration for deployment

The default local web port is `7860`.

## Project Directory Tree

```text
Medibot/
|-- Dockerfile
|-- README.md
|-- requirements.txt
|-- .dockerignore
|-- Medibot_dataset/
|   |-- dataset.csv
|   |-- Symptom-severity.csv
|   |-- symptom_Description.csv
|   `-- symptom_precaution.csv
|-- analytics/
|   |-- disease_overlap_matrix.csv
|   |-- disease_profiles.json
|   |-- disease_symptom_counts.png
|   |-- disease_symptom_overlap_heatmap.png
|   |-- eda_summary.json
|   |-- severity_frequency_correlation.png
|   |-- severity_score_distribution.png
|   |-- symptom_frequency_distribution.png
|   `-- symptom_frequency_severity.csv
|-- vector_store/
|   |-- documents.json
|   |-- manifest.json
|   |-- medibot.faiss
|   `-- tfidf_vectorizer.pkl
|-- scripts/
|   |-- build_index.py
|   |-- run_eda.py
|   |-- test_stage3_agents.py
|   |-- test_stage4_react.py
|   `-- test_stage5_safety.py
`-- src/
    |-- __init__.py
    |-- agents.py
    |-- app.py
    |-- config.py
    |-- data_pipeline.py
    |-- memory.py
    |-- orchestrator.py
    `-- vector_service.py
```

## Installation and Setup Manual

Run all commands from the MediBot project root:

```powershell
cd C:\Users\the_dell\OneDrive\Desktop\ML\Capstone\Medibot
```

### 1. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- faiss-cpu
- langchain
- langchain-core
- langgraph
- pydantic
- gradio

### 3. Run EDA Only

```bash
python scripts/run_eda.py
```

This regenerates analytics outputs in `analytics/`.

### 4. Build the FAISS Index and Analytics

```bash
python scripts/build_index.py
```

This command:

- loads and cleans the dataset
- regenerates analytics
- builds the FAISS vector store
- persists the index and metadata in `vector_store/`
- runs a smoke query for migraine-style symptoms

Expected smoke-query behavior:

```text
throbbing headache and light sensitivity -> Migraine
```

### 5. Run the Gradio Application Locally

```bash
python src/app.py
```

By default, the app launches on:

```text
http://0.0.0.0:7860
```

On a local workstation, open:

```text
http://127.0.0.1:7860
```

Environment variables:

```bash
MEDIBOT_HOST=0.0.0.0
MEDIBOT_PORT=7860
MEDIBOT_SHARE=false
MEDIBOT_MODEL_MODE=deterministic-local
```

PowerShell example using a custom port:

```powershell
$env:MEDIBOT_HOST="127.0.0.1"
$env:MEDIBOT_PORT="7865"
python .\src\app.py
```

## Docker Usage

The project includes a production-ready Dockerfile for Gradio deployment.

### 1. Build the Image

From `Capstone/Medibot`:

```bash
docker build -t medibot .
```

### 2. Run the Container

```bash
docker run --rm -p 7860:7860 medibot
```

Open:

```text
http://127.0.0.1:7860
```

### 3. Run with Environment Variables

```bash
docker run --rm -p 7860:7860 ^
  -e MEDIBOT_HOST=0.0.0.0 ^
  -e MEDIBOT_PORT=7860 ^
  -e MEDIBOT_MODEL_MODE=deterministic-local ^
  medibot
```

For macOS/Linux shells:

```bash
docker run --rm -p 7860:7860 \
  -e MEDIBOT_HOST=0.0.0.0 \
  -e MEDIBOT_PORT=7860 \
  -e MEDIBOT_MODEL_MODE=deterministic-local \
  medibot
```

## Verification and Testing Guide

All verification scripts are in `scripts/`.

### Stage 3: Isolated Agent and Memory Tests

```bash
python scripts/test_stage3_agents.py
```

This verifies:

- each specialized agent can be called as a plain Python function
- each LangChain tool is registered
- multi-turn memory enriches symptom context

Expected behavior includes:

```text
diagnosis: ['Migraine', 'Hypoglycemia']
severity: Medium
description: Migraine
turn 2 symptoms: ..., high fever, nausea
```

### Stage 4: ReAct Routing Tests

```bash
python scripts/test_stage4_react.py
```

This verifies:

- LangGraph ReAct initialization
- dynamic multi-tool routing
- Thought -> Action -> Observation trace generation
- sequential dependency handling for description and precautions
- memory carryover between turns

Expected routing for chest pain:

```text
Action: disease_diagnosis_agent
Action: symptom_severity_agent
Action: precaution_advisor_agent
```

Expected urgency:

```text
Emergency
```

### Stage 5: Safety and Disclaimer Tests

```bash
python scripts/test_stage5_safety.py
```

This verifies:

- non-medical queries are declined
- diagnostic tools are not triggered for out-of-scope requests
- valid medical requests still trigger appropriate tools
- every final response includes the mandatory clinical disclaimer

Expected fallback input:

```text
How do I fix a broken car engine?
```

Expected behavior:

```text
Fallback trace: [{'type': 'fallback', ...}]
Disclaimer: MediBot is an AI assistant, not a licensed medical professional. Always consult a doctor for urgent health concerns.
```

## Reviewer Walkthrough

For a clean end-to-end review, run:

```bash
python scripts/build_index.py
python scripts/test_stage3_agents.py
python scripts/test_stage4_react.py
python scripts/test_stage5_safety.py
python src/app.py
```

Then open:

```text
http://127.0.0.1:7860
```

Try these sample prompts:

```text
I have a throbbing headache and light sensitivity.
```

```text
I also have a fever and nausea.
```

```text
I have a sharp pain in my chest, what should I do?
```

```text
How do I fix a broken car engine?
```

The UI should show normal conversation in the chat window and tool routing details in the Thought/Action Log.
