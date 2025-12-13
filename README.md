# Simple_RAG

A minimal Retrieval-Augmented Generation (RAG) example project.

## Overview

This repository demonstrates a simple RAG pipeline using a local vector store and a retrieval layer to augment generation. It contains ingestion, generation, and a small app to serve or test the pipeline.

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

## Usage

- To ingest documents:

```powershell
python ingest.py
```

- To run the app (if it uses Streamlit or a Flask/FastAPI wrapper):

```powershell
python app.py
# or if streamlit is used:
streamlit run app.py
```

Adjust commands depending on the app's implementation.

## Push to GitHub

If you haven't initialized a git repo yet:

```bash
git init
git add .
git commit -m "Add README"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Replace `<your-username>` and `<your-repo>` with your repository values. If your repo already exists remotely, simply add the remote and push.

## Contributing

Feel free to open issues or PRs. Add unit tests for changes where practical.

## License

Add a license file if you wish (e.g., `LICENSE`).

---

If you want, I can also commit and push this README for you — provide GitHub remote URL or credentials, or I can show the exact commands to run locally.
