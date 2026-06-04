"""MiniMax LLM client integration."""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class ChatResult:
    """Text response plus token usage reported by the provider."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class MiniMaxClient:
    """MiniMax chat client using the OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or config.MINIMAX_API_KEY
        if not self.api_key:
            raise RuntimeError("MINIMAX_API_KEY is not configured. Add it to .env.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install openai to call MiniMax.") from exc

        self.model = model or config.MINIMAX_MODEL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or config.MINIMAX_BASE_URL,
        )

    def chat_with_usage(self, prompt: str) -> ChatResult:
        """Send a single-turn chat request and return token usage if provided."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个严谨、可引用资料的研究助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_LLM_TOKENS,
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        response_model = getattr(response, "model", None) or self.model
        return ChatResult(
            content=response.choices[0].message.content or "",
            model=response_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def chat(self, prompt: str) -> str:
        """Send a single-turn chat request and return only the text content."""
        return self.chat_with_usage(prompt).content
