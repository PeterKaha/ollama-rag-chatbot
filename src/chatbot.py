import sys

from colorama import Fore, Style, init

from src.document_loader import DocumentLoader
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

init(autoreset=True)

BANNER = f"""{Fore.CYAN}
╔══════════════════════════════════════════╗
║        Ollama RAG Chatbot  v1.0          ║
║   Lokaler Chatbot mit Dokumenten-RAG     ║
╚══════════════════════════════════════════╝
{Style.RESET_ALL}"""

HELP_TEXT = f"""{Fore.YELLOW}
Befehle:
  exit / quit  — Chatbot beenden
  reindex      — Neue Dokumente aus ./data einlesen
  sources      — Quellen der letzten Antwort anzeigen
  help         — Diese Hilfe anzeigen
{Style.RESET_ALL}"""


class Chatbot:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        vector_store: VectorStore,
        document_loader: DocumentLoader,
    ):
        self.rag_pipeline = rag_pipeline
        self.vector_store = vector_store
        self.document_loader = document_loader
        self._last_question: str = ""

    def run(self) -> None:
        print(BANNER)
        self._index_documents()
        print(HELP_TEXT)
        self._chat_loop()

    # ------------------------------------------------------------------
    # Indexierung
    # ------------------------------------------------------------------

    def _index_documents(self) -> None:
        print(Fore.YELLOW + "Scanne ./data nach Dokumenten...")
        documents = self.document_loader.load_all()

        if not documents:
            print(
                Fore.RED
                + "Keine Dokumente gefunden. "
                + "Lege .txt / .pdf / .md Dateien in ./data ab."
            )
        else:
            print(Fore.GREEN + f"{len(documents)} Dokument(e) geladen.")
            added = self.vector_store.add_documents(documents)
            if added:
                print(Fore.GREEN + f"{added} neue Chunks indexiert.")
            else:
                print(Fore.YELLOW + "Alle Dokumente bereits indexiert.")

        total = self.vector_store.get_document_count()
        print(Fore.CYAN + f"Vector Store: {total} Chunks gespeichert.\n")

    # ------------------------------------------------------------------
    # Chat-Loop
    # ------------------------------------------------------------------

    def _chat_loop(self) -> None:
        print(Fore.CYAN + "Stelle deine Frage (oder 'help' für alle Befehle):")
        print(Fore.CYAN + "-" * 50)

        while True:
            try:
                user_input = input(Fore.GREEN + "\nDu: " + Style.RESET_ALL).strip()

                if not user_input:
                    continue

                command = user_input.lower()

                if command in ("exit", "quit", "bye"):
                    print(Fore.CYAN + "Auf Wiedersehen!")
                    break

                if command == "help":
                    print(HELP_TEXT)
                    continue

                if command == "reindex":
                    self._index_documents()
                    continue

                if command == "sources":
                    self._show_sources()
                    continue

                self._answer(user_input)

            except KeyboardInterrupt:
                print(Fore.CYAN + "\n\nAuf Wiedersehen!")
                sys.exit(0)
            except Exception as e:
                print(Fore.RED + f"\n[Fehler] {e}")

    def _answer(self, question: str) -> None:
        self._last_question = question
        print(Fore.BLUE + "\nAssistent: " + Style.RESET_ALL, end="", flush=True)
        for chunk in self.rag_pipeline.query(question):
            print(chunk, end="", flush=True)
        print()

    def _show_sources(self) -> None:
        if not self._last_question:
            print(Fore.YELLOW + "Noch keine Frage gestellt.")
            return

        sources = self.rag_pipeline.get_sources(self._last_question)
        print(Fore.YELLOW + f"\nQuellen für: '{self._last_question}'")
        for i, doc in enumerate(sources, 1):
            filename = doc["metadata"].get("filename", "?")
            page = doc["metadata"].get("page", "")
            page_info = f"  Seite {page}" if page else ""
            score = 1 - doc["distance"]
            print(f"  [{i}] {filename}{page_info}  (Relevanz: {score:.2f})")
