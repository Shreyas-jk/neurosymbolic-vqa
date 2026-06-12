"""Ollama backend for the NL→Prolog translator.

Default model is `qwen2.5-coder:7b` — chosen after Phase 4 benchmarking on the
30-triple golden dataset. The plan's first pick (`llama3.2:3b`) scored 53% /
30, well below the 83% target; qwen2.5-coder:7b scored 100% / 30 — the plan's
documented Plan B kicks in. Trade-off: 4.7GB on disk + ~4.5s/query vs 2.0GB +
~2.7s/query for llama3.2:3b. The accuracy delta is decisive for this task.
`format="json"` forces structured JSON output, eliminating the markdown-fence
problem on the happy path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from nl2prolog.translator import TranslatorBackend


@dataclass
class OllamaBackend(TranslatorBackend):
    model: str = "qwen2.5-coder:7b"
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
