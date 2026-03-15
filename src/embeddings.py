from typing import List

import ollama


class OllamaEmbeddings:
    """Erzeugt Embeddings via lokaler Ollama-Instanz."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self.embed(text))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(texts)} Embeddings erstellt...")
        return embeddings
