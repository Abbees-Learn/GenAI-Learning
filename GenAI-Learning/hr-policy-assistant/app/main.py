
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.rag import HRPolicyRAG
from app.schemas import QuestionRequest, AnswerResponse


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine

    # Initialize PDF index and models once at startup.
    rag_engine = HRPolicyRAG()
    yield
    rag_engine = None


app = FastAPI(
    title="HR Policy Assistant API",
    description="Ask questions about company HR policy PDFs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "knowledge_base_ready": rag_engine is not None,
    }


@app.post("/api/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge base is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        result = rag_engine.answer(question)
        return AnswerResponse(**result)

    except Exception:
        # Log the detailed exception on the server, not to the browser.
        import logging
        logging.exception("Error while processing HR question")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check server logs.",
        )
