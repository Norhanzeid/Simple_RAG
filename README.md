# Simple_RAG

A minimal Retrieval-Augmented Generation (RAG) example project.

<img width="1920" height="1080" alt="Screenshot (538)" src="https://github.com/user-attachments/assets/deea1cb9-91ec-4e73-b0a8-5afae36ae352" />

<img width="1920" height="1080" alt="Screenshot (539)" src="https://github.com/user-attachments/assets/50a43305-9184-41d1-b76a-a2d5ac606876" />

<img width="1920" height="1080" alt="Screenshot (540)" src="https://github.com/user-attachments/assets/7c7b508d-a181-4fe3-9e43-a1228b4770ed" />


## Overview

This project is designed to answer questions about Natural Language Processing (NLP) and Transformer models by retrieving relevant document context and using a generator to produce informed responses.
## Files

- `app.py` - Small app / demo entrypoint.
- `ingest.py` - Script to ingest documents into the vector store.
- `generator.py` - Generation logic for producing answers.
- `retriever.py` - Retrieval wrapper around the vector DB.
- `data/` - Source documents and vector DB directory (`vector_db/index.faiss`).

## Prerequisites

- Python 3.8+ recommended
- A virtual environment (recommended)

## Install

1. Create & activate virtualenv (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```
