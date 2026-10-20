
# HR Policy Assistant

A document-based HR question-answering application built using
FastAPI, HTML, CSS, JavaScript, FAISS, Sentence Transformers,
CrossEncoder, and the Hugging Face API.

## Features

- Chat interface for HR policy questions
- PDF text extraction
- Semantic vector search using FAISS
- CrossEncoder reranking
- LLM-generated answers grounded in retrieved context
- Source document and page references
- FastAPI endpoints and interactive API documentation

## Project Structure

```text
hr-policy-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── rag.py
│   ├── schemas.py
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
├── documents/
│   └── .gitkeep
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- A Hugging Face API token with access to the selected model
- PDF documents containing extractable text

## Setup

Create and activate a virtual environment.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and add your
Hugging Face token.

Place your authorized PDF files in the `documents/` folder.

## Run the application

```bash
uvicorn app.main:app --reload
```

Open the chat UI:

http://127.0.0.1:8000

FastAPI Swagger documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/api/health

## API

POST `/api/ask`

Request:

```json
{
  "question": "What are the office timings?"
}
```

Response:

```json
{
  "answer": "The answer generated from your HR documents.",
  "sources": [
    {
      "document": "OfficeTime.pdf",
      "page": 1,
      "excerpt": "Relevant retrieved document passage."
    }
  ]
}
```

The response values above are examples, not actual policy answers.

## Security

- Never commit `.env` or API tokens.
- Do not upload confidential HR documents to a public repository.
- Configure authentication and access control before exposing
  the application to other users.
