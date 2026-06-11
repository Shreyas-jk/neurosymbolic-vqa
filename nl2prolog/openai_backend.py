"""OpenAI backend for the NL→Prolog translator.

Uses gpt-4o-mini at temperature 0.2 with `response_format={"type": "json_object"}`
so the model returns raw JSON without markdown wrappers. The system prompt
mentions "JSON" (required by OpenAI's structured-output mode).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from nl2prolog.translator import TranslatorBackend


@dataclass
class OpenAIBackend(TranslatorBackend):
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    api_key: Optional[str] = None
    name: str = "openai"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Either export it or use the local backend."
            )

    def call(self, system_prompt: str, user_prompt: str) -> str:
        # Lazy import so importing the module doesn't require openai to be
        # installed (matches the same pattern in local_backend).
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content
        return content or ""
