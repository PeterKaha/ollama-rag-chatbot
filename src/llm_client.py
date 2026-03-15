from typing import Iterator

import ollama


class OllamaLLMClient:
    """Client für Text-Generierung via lokaler Ollama-Instanz."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def generate(self, prompt: str, stream: bool = True) -> Iterator[str]:
        if stream:
            for chunk in self.client.generate(model=self.model, prompt=prompt, stream=True):
                yield chunk["response"]
        else:
            response = self.client.generate(model=self.model, prompt=prompt, stream=False)
            yield response["response"]

    def generate_text(self, prompt: str) -> str:
        return "".join(self.generate(prompt, stream=False))

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def model_exists(self, model_name: str | None = None) -> bool:
        target_model = model_name or self.model
        try:
            models = self.client.list()
            names = [m["model"] for m in models.get("models", [])]
            return any(target_model in name for name in names)
        except Exception:
            return False
