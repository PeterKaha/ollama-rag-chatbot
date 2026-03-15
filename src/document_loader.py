import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import pypdf


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_all(self) -> List[Document]:
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
            return []

        documents = []
        for file_path in sorted(self.data_dir.rglob("*")):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                docs = self._load_file(file_path)
                documents.extend(docs)

        return documents

    def _load_file(self, file_path: Path) -> List[Document]:
        try:
            if file_path.suffix.lower() == ".pdf":
                return self._load_pdf(file_path)
            return self._load_text(file_path)
        except Exception as e:
            print(f"[Warnung] Konnte {file_path.name} nicht laden: {e}")
            return []

    def _load_text(self, file_path: Path) -> List[Document]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return []

        return [
            Document(
                content=content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "type": file_path.suffix.lower(),
                },
            )
        ]

    def _load_pdf(self, file_path: Path) -> List[Document]:
        documents = []
        reader = pypdf.PdfReader(str(file_path))

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "type": ".pdf",
                            "page": i + 1,
                        },
                    )
                )

        return documents
