"""Ollama backend for the NL→Prolog translator.

Default model is `llama3.2:3b` (small enough to run snappily on an M-series
Mac; chosen in the plan after benchmarking against qwen and mistral). `format="json"`
forces structured JSON output, eliminating the markdown-fence problem on the
happy path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from nl2prolog.translator import TranslatorBackend


@dataclass
class OllamaBackend(TranslatorBackend):
    model: str = "llama3.2:3b"
    temperature: float = 0.2
    host: str = ""
    name: str = "ollama"

    def __post_init__(self) -> None:
        if not self.host:
            self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def call(self, system_prompt: str, user_prompt: str) -> str:
        # Lazy import so the dependency only matters when the backend runs.
        from ollama import Client

        client = Client(host=self.host)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]
