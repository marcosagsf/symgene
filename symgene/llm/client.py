from __future__ import annotations

from typing import Any


class LLMClient:
    """Thin wrapper around LLM provider APIs.

    Provides a single ``complete(prompt, system)`` method that works with
    either the Anthropic or OpenAI SDK, keeping the rest of the LLM
    subsystem provider-agnostic.

    Parameters
    ----------
    provider : {"anthropic", "openai", "gemini"}
        Which SDK to use.
    model : str
        Model identifier (e.g. "claude-sonnet-4-6", "gpt-4o-mini", "gemini-2.5-flash").
    api_key : str or None
        API key. If None, reads from the environment variable expected by
        the chosen SDK (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY).
    max_tokens : int
        Maximum tokens in the completion.
    temperature : float
        Sampling temperature.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        if provider not in {"anthropic", "openai", "gemini"}:
            raise ValueError(
                f"provider must be 'anthropic', 'openai', or 'gemini', got {provider!r}"
            )

        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = self._build_client(provider, api_key)

    def _build_client(self, provider: str, api_key: str | None) -> Any:
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "Anthropic SDK not found. Install with: pip install symgene[llm]"
                ) from e
            return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        elif provider == "openai":
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "OpenAI SDK not found. Install with: pip install symgene[llm]"
                ) from e
            return openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        else:  # gemini
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "Google GenAI SDK not found. Install with: pip install symgene[llm]"
                ) from e
            import os as _os
            resolved_key = api_key or _os.environ.get("GEMINI_API_KEY") or _os.environ.get("GOOGLE_API_KEY")
            return genai.Client(api_key=resolved_key) if resolved_key else genai.Client()

    def complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt and return the text response.

        Parameters
        ----------
        prompt : str
            User message.
        system : str
            System instructions (optional).

        Returns
        -------
        str
            Text content of the first response choice.
        """
        if self.provider == "anthropic":
            return self._complete_anthropic(prompt, system)
        if self.provider == "openai":
            return self._complete_openai(prompt, system)
        return self._complete_gemini(prompt, system)

    def _complete_anthropic(self, prompt: str, system: str) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _complete_openai(self, prompt: str, system: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        return response.choices[0].message.content

    def _complete_gemini(self, prompt: str, system: str) -> str:
        from google.genai import types
        # Cache thinking config — computed once on first call, never changes per instance.
        # Disable thinking for models that support it (e.g. gemini-2.5-flash): thinking
        # tokens count against max_output_tokens, starving the actual response.
        if not hasattr(self, "_gemini_thinking_cfg"):
            self._gemini_thinking_cfg = None
            if "2.5" in self.model or "thinking" in self.model.lower():
                try:
                    self._gemini_thinking_cfg = types.ThinkingConfig(thinking_budget=0)
                except Exception:
                    pass
        config_kwargs: dict = dict(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if system:
            config_kwargs["system_instruction"] = system
        if self._gemini_thinking_cfg is not None:
            config_kwargs["thinking_config"] = self._gemini_thinking_cfg
        config = types.GenerateContentConfig(**config_kwargs)
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        try:
            return response.text
        except ValueError as e:
            finish = (
                response.candidates[0].finish_reason
                if response.candidates
                else "unknown"
            )
            raise RuntimeError(
                f"Gemini returned no text (finish_reason={finish}). "
                "Response may have been blocked by safety filters."
            ) from e
