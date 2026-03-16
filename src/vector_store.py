import uuid
import re
from typing import Any, Dict, List, Set

import chromadb

from src.document_loader import Document
from src.embeddings import OllamaEmbeddings


class VectorStore:
    COLLECTION_NAME = "rag_documents"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    STOPWORDS = {
        "aber",
        "alle",
        "aller",
        "alles",
        "als",
        "also",
        "am",
        "an",
        "auch",
        "auf",
        "aus",
        "bei",
        "bin",
        "bis",
        "bist",
        "da",
        "dann",
        "darauf",
        "das",
        "dass",
        "dein",
        "deine",
        "dem",
        "den",
        "der",
        "des",
        "dessen",
        "die",
        "dies",
        "diese",
        "dieser",
        "doch",
        "dort",
        "du",
        "ein",
        "eine",
        "einem",
        "einen",
        "einer",
        "es",
        "etwas",
        "euch",
        "euer",
        "fuer",
        "hat",
        "hier",
        "hinter",
        "ich",
        "ihr",
        "ihre",
        "im",
        "in",
        "ist",
        "ja",
        "kann",
        "man",
        "mein",
        "mit",
        "muss",
        "nach",
        "nicht",
        "noch",
        "nun",
        "oder",
        "seid",
        "sein",
        "seine",
        "sich",
        "sie",
        "sind",
        "soll",
        "sollte",
        "sonst",
        "ueber",
        "um",
        "und",
        "uns",
        "unter",
        "von",
        "vor",
        "wann",
        "warum",
        "was",
        "weil",
        "weiter",
        "welche",
        "wenn",
        "wer",
        "werden",
        "wie",
        "wir",
        "wird",
        "wo",
        "zu",
        "zum",
        "zur",
    }

    def __init__(self, persist_dir: str, embeddings: OllamaEmbeddings):
        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document], on_progress=None) -> int:
        """Chunked documents in den Vector Store einfügen. Bereits indexierte Quellen werden übersprungen."""
        all_chunks: List[Document] = []
        for doc in documents:
            for i, chunk_text in enumerate(self._split_text(doc.content)):
                all_chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )

        if not all_chunks:
            return 0

        existing_sources = self._get_existing_sources()
        new_chunks = [c for c in all_chunks if c.metadata.get("source") not in existing_sources]

        if not new_chunks:
            return 0

        # Quellen einzeln verarbeiten für Fortschrittsanzeige
        chunks_by_source: Dict[str, List[Document]] = {}
        for c in new_chunks:
            src = c.metadata.get("source", "")
            chunks_by_source.setdefault(src, []).append(c)

        total_all = len(new_chunks)
        total_done = 0

        for source, chunks in chunks_by_source.items():
            filename = chunks[0].metadata.get("filename") or source.split("/")[-1]
            n = len(chunks)

            def _make_chunk_cb(fn: str, chunk_count: int, done_before: int):
                def on_chunk(chunk_idx: int, _total: int) -> None:
                    if on_progress and (chunk_idx % 25 == 0 or chunk_idx == chunk_count):
                        on_progress(fn, chunk_idx, chunk_count, done_before + chunk_idx, total_all)
                return on_chunk

            chunk_cb = _make_chunk_cb(filename, n, total_done) if on_progress else None

            print(f"Erstelle Embeddings für {n} Chunks ({filename})...")
            embeddings = self.embeddings.embed_many([c.content for c in chunks], on_chunk=chunk_cb)

            self.collection.add(
                ids=[str(uuid.uuid4()) for _ in chunks],
                embeddings=embeddings,
                documents=[c.content for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
            total_done += n

        return total_done

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Ähnlichkeitssuche im Vector Store."""
        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self.embeddings.embed(query_text)
        candidate_count = min(max(n_results * 5, 20), count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
        )

        semantic_candidates = [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["documents"][0]))
        ]

        lexical_candidates = self._keyword_candidates(query_text)
        merged_candidates = self._merge_candidates(semantic_candidates, lexical_candidates)
        reranked = self._rerank_candidates(query_text, merged_candidates)
        return reranked[:n_results]

    def get_document_count(self) -> int:
        return self.collection.count()

    def get_source_stats(self) -> List[Dict[str, Any]]:
        """Gibt pro Quelle: source, filename, type, chunks, page_count zurück."""
        if self.collection.count() == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        stats: Dict[str, Dict[str, Any]] = {}
        for metadata in data["metadatas"]:
            if not metadata:
                continue
            source = metadata.get("source", "")
            if not source:
                continue
            if source not in stats:
                stats[source] = {
                    "source": source,
                    "filename": metadata.get("filename", source.split("/")[-1]),
                    "type": metadata.get("type", ""),
                    "chunks": 0,
                    "page_count": 0,
                }
            stats[source]["chunks"] += 1
            page = metadata.get("page")
            if page is not None:
                stats[source]["page_count"] = max(stats[source]["page_count"], int(page))

        return sorted(stats.values(), key=lambda s: s["filename"].lower())

    def clear(self) -> None:
        """Löscht die komplette Collection und legt sie neu an."""
        self.client.delete_collection(name=self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_by_source(self, source_query: str) -> int:
        """Löscht alle Chunks, deren Quelle den Query-String enthält."""
        query = source_query.strip().casefold()
        if not query or self.collection.count() == 0:
            return 0

        data = self.collection.get(include=["metadatas"])
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])

        ids_to_delete = []
        for doc_id, metadata in zip(ids, metadatas):
            source = str((metadata or {}).get("source", "")).casefold()
            filename = str((metadata or {}).get("filename", "")).casefold()
            if query in source or query in filename:
                ids_to_delete.append(doc_id)

        if not ids_to_delete:
            return 0

        self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def delete_by_sources(self, sources: List[str]) -> int:
        """Löscht alle Chunks, deren source exakt in der übergebenen Liste enthalten ist."""
        if not sources or self.collection.count() == 0:
            return 0

        source_set = {s.strip() for s in sources}
        data = self.collection.get(include=["metadatas"])
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])

        ids_to_delete = [
            doc_id
            for doc_id, metadata in zip(ids, metadatas)
            if str((metadata or {}).get("source", "")) in source_set
        ]

        if not ids_to_delete:
            return 0

        self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def _get_existing_sources(self) -> Set[str]:
        if self.collection.count() == 0:
            return set()
        result = self.collection.get(include=["metadatas"])
        return {m.get("source") for m in result["metadatas"] if m.get("source")}

    def _keyword_candidates(self, query_text: str) -> List[Dict[str, Any]]:
        data = self.collection.get(include=["documents", "metadatas"])
        candidates = []
        for document, metadata in zip(data["documents"], data["metadatas"]):
            score = self._keyword_score(query_text, document, metadata)
            if score <= 0:
                continue
            candidates.append(
                {
                    "content": document,
                    "metadata": metadata,
                    "distance": 1.0,
                    "keyword_score": score,
                }
            )

        candidates.sort(key=lambda item: item["keyword_score"], reverse=True)
        return candidates

    def _merge_candidates(
        self,
        semantic_candidates: List[Dict[str, Any]],
        lexical_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[tuple, Dict[str, Any]] = {}

        for candidate in semantic_candidates + lexical_candidates:
            normalized_content = self._normalize_text(candidate["content"])
            key = (
                candidate["metadata"].get("source"),
                normalized_content,
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                continue

            existing["distance"] = min(existing.get("distance", 1.0), candidate.get("distance", 1.0))
            existing["keyword_score"] = max(
                existing.get("keyword_score", 0.0),
                candidate.get("keyword_score", 0.0),
            )

        return list(merged.values())

    def _rerank_candidates(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        reranked = []
        for candidate in candidates:
            semantic_score = 1.0 - float(candidate.get("distance", 1.0))
            keyword_score = float(
                candidate.get("keyword_score", self._keyword_score(query_text, candidate["content"], candidate["metadata"]))
            )
            final_score = max(
                (semantic_score * 0.25) + (keyword_score * 0.75),
                keyword_score * 0.95,
            )
            reranked.append(
                {
                    "content": candidate["content"],
                    "metadata": candidate["metadata"],
                    "distance": 1.0 - final_score,
                }
            )

        reranked.sort(key=lambda item: item["distance"])
        return reranked

    def _keyword_score(self, query_text: str, document_text: str, metadata: Dict[str, Any]) -> float:
        query_normalized = self._normalize_text(query_text)
        document_normalized = self._normalize_text(document_text)
        filename_normalized = self._normalize_text(str(metadata.get("filename", "")))

        query_tokens = self._tokenize(query_normalized)
        if not query_tokens:
            return 0.0

        document_tokens = set(self._tokenize(document_normalized))
        filename_tokens = set(self._tokenize(filename_normalized))

        overlap = sum(1 for token in query_tokens if self._matches_token(token, document_tokens))
        filename_overlap = sum(1 for token in query_tokens if self._matches_token(token, filename_tokens))

        phrase_bonus = 0.0
        if query_normalized in document_normalized:
            phrase_bonus += 0.45

        consecutive_bonus = 0.0
        compact_query = query_normalized.replace(" ", "")
        compact_document = document_normalized.replace(" ", "")
        if compact_query and compact_query in compact_document:
            consecutive_bonus += 0.25

        cooccurrence_bonus = 0.0
        if "mitteil" in document_normalized and ("umzug" in document_normalized or "umzieh" in document_normalized):
            cooccurrence_bonus += 0.5
        if "rechtzeitig" in document_normalized and "mitteil" in document_normalized:
            cooccurrence_bonus += 0.2

        score = overlap / len(query_tokens)
        score += (filename_overlap / max(1, len(query_tokens))) * 0.35
        score += phrase_bonus + consecutive_bonus + cooccurrence_bonus
        return min(score, 1.8)

    def _normalize_text(self, text: str) -> str:
        normalized = text.casefold()
        normalized = normalized.replace("ä", "ae")
        normalized = normalized.replace("ö", "oe")
        normalized = normalized.replace("ü", "ue")
        normalized = normalized.replace("ß", "ss")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _tokenize(self, text: str) -> List[str]:
        return [
            token
            for token in text.split()
            if len(token) > 2 and token not in self.STOPWORDS
        ]

    def _matches_token(self, query_token: str, candidate_tokens: Set[str]) -> bool:
        if query_token in candidate_tokens:
            return True

        if len(query_token) < 5:
            return False

        for candidate in candidate_tokens:
            if query_token.startswith(candidate[:5]) or candidate.startswith(query_token[:5]):
                return True

        return False

    def _split_text(self, text: str) -> List[str]:
        """Text in überlappende Chunks aufteilen."""
        if len(text) <= self.CHUNK_SIZE:
            return [text.strip()]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            if end < len(text):
                # Satzgrenze bevorzugen
                for sep in ("\n\n", "\n", ". ", " "):
                    pos = text.rfind(sep, start, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.CHUNK_OVERLAP

        return chunks
