from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.app_service import AppConfig, RAGApplication


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web" / "static"
INDEX_FILE = BASE_DIR / "web" / "templates" / "index.html"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class DeleteSourceRequest(BaseModel):
    source_query: str = Field(min_length=1, max_length=1024)


def create_app() -> FastAPI:
    config = AppConfig.from_env()
    rag_app = RAGApplication(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        rag_app.validate_dependencies()
        rag_app.index_documents()
        yield

    app = FastAPI(
        title="Ollama RAG Chatbot",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.rag_app = rag_app
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    data_dir = Path(config.data_dir).resolve()
    if data_dir.is_dir():
        app.mount("/docs", StaticFiles(directory=data_dir), name="docs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_FILE.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, **app.state.rag_app.get_status()}

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> dict:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Frage darf nicht leer sein.")

        try:
            return app.state.rag_app.answer_question(question)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/reindex")
    def reindex() -> dict:
        try:
            return app.state.rag_app.index_documents()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/delete-source")
    def delete_source(payload: DeleteSourceRequest) -> dict:
        source_query = payload.source_query.strip()
        if not source_query:
            raise HTTPException(status_code=400, detail="source_query darf nicht leer sein.")

        try:
            return app.state.rag_app.delete_from_index(source_query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/clear-index")
    def clear_index() -> dict:
        try:
            return app.state.rag_app.clear_index()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()