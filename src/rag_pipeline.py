import re
from typing import Any, Dict, Iterator, List

from src.llm_client import OllamaLLMClient
from src.vector_store import VectorStore

PROMPT_TEMPLATE = """\
Du bist ein hilfreicher Assistent. Beantworte die Frage des Nutzers \
ausschließlich auf Basis des folgenden Kontexts aus den bereitgestellten Dokumenten.
Wenn der Kontext keine ausreichenden Informationen enthält, teile das dem Nutzer \
ehrlich mit und spekuliere nicht.
Wenn im Kontext eine Frist, Definition oder direkte Anweisung explizit genannt wird, \
uebernimm sie praezise in die Antwort.
Wenn mehrere Zeitangaben im Kontext vorkommen, erklaere sie getrennt und ohne Widerspruch.
Antworte knapp, konkret und auf Deutsch.

Kontext:
{context}

Frage: {question}

Antwort:"""


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: OllamaLLMClient,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def query(self, question: str, stream: bool = True) -> Iterator[str]:
        """Führt einen RAG-Query durch: Retrieval + Generation."""
        relevant_docs = self.vector_store.query(question, n_results=self.top_k)

        if not relevant_docs:
            yield (
                "Keine Dokumente im Vector Store gefunden. "
                "Bitte füge Dateien in das ./data Verzeichnis ein und starte neu."
            )
            return

        direct_answer = self._extract_direct_answer(question, relevant_docs)
        if direct_answer:
            yield direct_answer
            return

        context = self._build_context(question, relevant_docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        yield from self.llm_client.generate(prompt, stream=stream)

    def answer(self, question: str) -> str:
        return "".join(self.query(question, stream=False))

    def get_sources(self, question: str) -> List[Dict[str, Any]]:
        """Gibt die relevantesten Dokument-Chunks für eine Frage zurück."""
        return self.vector_store.query(question, n_results=self.top_k)

    def _build_context(self, question: str, documents: List[Dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(documents, 1):
            filename = doc["metadata"].get("filename", "Unbekannt")
            page = doc["metadata"].get("page")
            page_info = f" (Seite {page})" if page else ""
            excerpt = self._extract_relevant_excerpt(question, doc["content"])
            parts.append(f"[{i}] Quelle: {filename}{page_info}\n{excerpt}")

        return "\n\n---\n\n".join(parts)

    def _extract_relevant_excerpt(self, question: str, content: str, max_sentences: int = 4) -> str:
        sentences = self._split_sentences(content)
        if len(sentences) <= max_sentences:
            return self._clean_ocr(content)

        ranked_sentences = self._rank_sentences(question, sentences)
        selected = sorted(ranked_sentences[:max_sentences], key=lambda item: item[1])
        return " ".join(sentence for _, index, sentence in selected)

    def _extract_direct_answer(self, question: str, documents: List[Dict[str, Any]]) -> str | None:
        if not self._is_timing_question(question):
            return None

        sentence_map: Dict[str, tuple[int, int, str]] = {}
        for doc in documents:
            ranked = self._rank_sentences(question, self._split_sentences(doc["content"]))
            for score, index, sentence in ranked[:3]:
                cleaned_sentence = self._clean_ocr(sentence)
                existing = sentence_map.get(cleaned_sentence)
                if existing is None or score > existing[0]:
                    sentence_map[cleaned_sentence] = (score, index, cleaned_sentence)

        ranked_sentences = sorted(sentence_map.values(), key=lambda item: (-item[0], item[1]))
        preferred_sentences = [sentence for score, _, sentence in ranked_sentences if score >= 4 and self._has_time_marker(sentence)]
        fallback_sentences = [sentence for score, _, sentence in ranked_sentences if score >= 4 and sentence not in preferred_sentences]
        strong_sentences = (preferred_sentences + fallback_sentences)[:2]
        if not strong_sentences:
            return None

        return self._clean_ocr(" ".join(strong_sentences))

    def _rank_sentences(self, question: str, sentences: List[str]) -> List[tuple[int, int, str]]:
        question_tokens = set(self._tokenize(question))
        ranked_sentences = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = set(self._tokenize(sentence))
            overlap = len(question_tokens & sentence_tokens)
            bonus = 0
            sentence_normalized = sentence.casefold()
            if "rechtzeitig" in sentence_normalized:
                bonus += 3
            if "frist" in sentence_normalized or "woche" in sentence_normalized or "tag" in sentence_normalized:
                bonus += 2
            if "vor " in sentence_normalized or "nach " in sentence_normalized or "bis " in sentence_normalized:
                bonus += 2
            if "umzug" in sentence_normalized or "umziehen" in sentence_normalized:
                bonus += 2
            if "agentur" in sentence_normalized and "arbeit" in sentence_normalized:
                bonus += 1
            if "mitteil" in sentence_normalized:
                bonus += 2
            ranked_sentences.append((overlap + bonus, index, sentence))

        ranked_sentences.sort(key=lambda item: (-item[0], item[1]))
        return ranked_sentences

    def _is_timing_question(self, question: str) -> bool:
        normalized = question.casefold()
        cues = ("wann", "frist", "bis wann", "rechtzeitig", "wie lange", "innerhalb")
        return any(cue in normalized for cue in cues)

    def _has_time_marker(self, sentence: str) -> bool:
        normalized = sentence.casefold()
        markers = (
            "vor dem",
            "nach dem",
            "innerhalb",
            "rechtzeitig bedeutet",
            "woche",
            "wochen",
            "tag",
            "tage",
            "monat",
            "monate",
            "frist",
        )
        return any(marker in normalized for marker in markers)

    def _clean_ocr(self, text: str) -> str:
        """Bereinigt typische OCR-Artefakte aus PDF-extrahiertem Text."""
        # Silbentrennung: "inner-\nhalb" → "innerhalb", "Um-\nzug" → "Umzug"
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        # Silbentrennung ohne Zeilenumbruch: "inner- halb" → "innerhalb"
        text = re.sub(r"(\w+)-\s{1,3}([a-zäöüß]+)", r"\1\2", text)
        # Mehrzeilige Leerzeichen normalisieren
        text = re.sub(r"\s+", " ", text)
        # Leerzeichen vor Satzzeichen entfernen
        text = re.sub(r" ([.,;:!?])", r"\1", text)
        return text.strip()

    def _split_sentences(self, content: str) -> List[str]:
        cleaned = self._clean_ocr(content)
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return [part.strip() for part in parts if part.strip()]

    def _tokenize(self, text: str) -> List[str]:
        normalized = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß]+", " ", text.casefold())
        return [token for token in normalized.split() if len(token) > 2]
