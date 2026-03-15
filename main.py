import os
import sys
import argparse

from dotenv import load_dotenv

load_dotenv()

from src.app_service import AppConfig, RAGApplication
from src.chatbot import Chatbot
from src.web_app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lokaler Ollama RAG Chatbot")
    parser.add_argument(
        "mode",
        nargs="?",
        default="cli",
        choices=("cli", "web"),
        help="Startmodus: cli oder web",
    )
    parser.add_argument("--host", default=os.getenv("WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEB_PORT", "8000")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env()
    rag_app = RAGApplication(config)

    try:
        rag_app.validate_dependencies()
    except RuntimeError as exc:
        print(f"[Fehler] {exc}")
        sys.exit(1)

    if args.mode == "web":
        try:
            import uvicorn
        except ImportError as exc:
            print("[Fehler] uvicorn ist nicht installiert. Führe aus: pip install -r requirements.txt")
            raise SystemExit(1) from exc
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return

    chatbot = Chatbot(
        rag_pipeline=rag_app.rag_pipeline,
        vector_store=rag_app.vector_store,
        document_loader=rag_app.document_loader,
    )
    chatbot.run()


if __name__ == "__main__":
    main()
