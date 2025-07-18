# 🧠 Research Assistant

A research assistant that can answer general questions by retrieving information from the web (via SerpAPI) and perform Retrieval‑Augmented Generation (RAG) over user‑uploaded PDF documents. Responses are synthesized by OpenAI’s LLM, formatted with clear headings, lists, and citations, and can the answer can be be downloaded as a PDF report. Used APA format to define the structure of the report.

---

## Features
- **Orchestrator** : A controller to sequence tools (Web Retriever, Document Retreiver, Extractor(Web) and Summarizer(Web)
- **Web Retrieval**: Queries live web results using SerpAPI, extracts key insights, and generates a detailed answer with source URLs.
- **PDF Document Retrival(RAG based)**: Upload one or more PDF files to run a RAG pipeline over your own documents—indexing, chunking, similarity search, extraction, and summarization.
- **Downloadable PDF**: Export the final answer and source list to a well‑formatted PDF with proper headings, lists, and inline bold/italic styling.
- **Interactive UI**: Gradio‑based web interface with file upload, loading spinner, and download report (PDF).

---

## 📁 Project Structure
```text
Research_Assistant/
├── .env # your API keys
├── requirements.txt # required packages
├── main.py 
├── UI_interface.py 
├── orchestrator.py # Task Controller
└── tools/
├── init.py
├── web_retriever.py # SerpAPIWrapper 
├── doc_retriever.py # RAG -> Summarizer
├── extractor.py # To extract key insights from the web retriever to pass it to the summarizer
└── summarizer.py # Summarizes in the desired format
```
##Getting Started

Follow these steps to run the Research Assistant

```bash
git clone https://github.com/rvnisha/Research_Assistant.git
cd Research_Assistant
```

```bash
# Create & activate a virtual environment
python -m venv vename
vename\Scripts\activate
```

```bash
# Install required Python packages
pip install -r requirements.txt
```

```bash
#Run the app
python UI_interface.py
The app will start on http://localhost:7860
```



