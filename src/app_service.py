import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List

from src.document_loader import DocumentLoader
from src.embeddings import OllamaEmbeddings
from src.llm_client import OllamaLLMClient
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


@dataclass
class AppConfig:
    ollama_base_url: str
    llm_model: str
    embed_model: str
    chroma_dir: str
    data_dir: str
    top_k: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            llm_model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            chroma_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            data_dir=os.getenv("DATA_DIR", "./data"),
            top_k=int(os.getenv("TOP_K_RESULTS", "5")),
        )


class RAGApplication:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm_client = OllamaLLMClient(
            model=config.llm_model,
            base_url=config.ollama_base_url,
        )
        self.embeddings = OllamaEmbeddings(
            model=config.embed_model,
            base_url=config.ollama_base_url,
        )
        self.vector_store = VectorStore(
            persist_dir=config.chroma_dir,
            embeddings=self.embeddings,
        )
        self.document_loader = DocumentLoader(data_dir=config.data_dir)
        self.rag_pipeline = RAGPipeline(
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            top_k=config.top_k,
        )

    def validate_dependencies(self) -> None:
        if not self.llm_client.is_available():
            raise RuntimeError(
                f"Ollama ist nicht erreichbar unter {self.config.ollama_base_url}. "
                "Starte Ollama mit: ollama serve"
            )

        missing_models = []
        if not self.llm_client.model_exists(self.config.llm_model):
            missing_models.append(self.config.llm_model)
        if not self.llm_client.model_exists(self.config.embed_model):
            missing_models.append(self.config.embed_model)

        if missing_models:
            model_list = ", ".join(missing_models)
            raise RuntimeError(
                f"Folgende Ollama-Modelle fehlen: {model_list}. "
                f"Lade sie herunter mit: ollama pull {missing_models[0]}"
            )

    def index_documents(self) -> Dict[str, int]:
        documents = self.document_loader.load_all()
        added_chunks = self.vector_store.add_documents(documents)
        return {
            "documents_loaded": len(documents),
            "chunks_added": added_chunks,
            "chunks_total": self.vector_store.get_document_count(),
        }

    def index_documents_stream(self) -> Generator[str, None, None]:
        """Generator der SSE-Fortschritts-Events als data:-Strings liefert."""
        documents = self.document_loader.load_all()
        yield f"data: {json.dumps({'type': 'start', 'documents': len(documents)})}\n\n"

        q: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        def on_progress(filename: str, chunk_done: int, chunk_total: int,
                        total_done: int, total_all: int) -> None:
            q.put({
                "type": "progress",
                "filename": filename,
                "chunk_done": chunk_done,
                "chunk_total": chunk_total,
                "total_done": total_done,
                "total_all": total_all,
            })

        def run() -> None:
            try:
                added = self.vector_store.add_documents(documents, on_progress=on_progress)
                q.put({
                    "type": "done",
                    "chunks_added": added,
                    "chunks_total": self.vector_store.get_document_count(),
                    "documents_loaded": len(documents),
                })
            except Exception as exc:
                q.put({"type": "error", "message": str(exc)})

        threading.Thread(target=run, daemon=True).start()

        while True:
            event = q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break

    def clear_index(self) -> Dict[str, int]:
        before = self.vector_store.get_document_count()
        self.vector_store.clear()
        return {
            "chunks_deleted": before,
            "chunks_total": self.vector_store.get_document_count(),
        }

    def delete_from_index(self, source_query: str) -> Dict[str, int]:
        deleted = self.vector_store.delete_by_source(source_query)
        return {
            "chunks_deleted": deleted,
            "chunks_total": self.vector_store.get_document_count(),
        }

    def list_indexed_sources(self) -> List[Dict]:
        """Gibt alle indizierten Quellen mit exists_on_disk-Flag zurück."""
        stats = self.vector_store.get_source_stats()
        return [
            {**entry, "exists_on_disk": self._resolve_source_path(entry["source"]).is_file()}
            for entry in stats
        ]

    def cleanup_stale_sources(self) -> Dict[str, Any]:
        """Löscht Indexeinträge für Quellen, die nicht mehr auf der Disk existieren."""
        sources_info = self.list_indexed_sources()
        stale = [s["source"] for s in sources_info if not s["exists_on_disk"]]
        total = self.vector_store.get_document_count()
        if not stale:
            return {"removed_sources": 0, "chunks_deleted": 0, "chunks_total": total}
        deleted = self.vector_store.delete_by_sources(stale)
        return {
            "removed_sources": len(stale),
            "chunks_deleted": deleted,
            "chunks_total": self.vector_store.get_document_count(),
        }

    def _resolve_source_path(self, source: str) -> Path:
        p = Path(source)
        if p.is_absolute():
            return p
        return Path(self.config.data_dir).resolve() / p

    def answer_question(self, question: str) -> Dict[str, Any]:
        answer = self.rag_pipeline.answer(question)
        sources = self.rag_pipeline.get_sources(question)
        return {
            "answer": answer,
            "sources": [self._serialize_source(source) for source in sources],
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "ollama_base_url": self.config.ollama_base_url,
            "llm_model": self.config.llm_model,
            "embed_model": self.config.embed_model,
            "data_dir": self.config.data_dir,
            "document_chunks": self.vector_store.get_document_count(),
        }

    def _serialize_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(source["metadata"])
        distance = float(source["distance"])
        metadata.setdefault("page", None)
        return {
            "content": source["content"],
            "metadata": metadata,
            "distance": distance,
            "relevance": max(0.0, min(1.0, 1.0 - distance)),
        }