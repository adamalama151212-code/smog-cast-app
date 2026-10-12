"""Language model behind any OpenAI-compatible endpoint (LM Studio locally, Databricks Model Serving later).

Only the endpoint changes between environments (conf/app.yaml + SMOGCAST_LLM_* variables), so a
locally distilled model, a model hosted in the cloud and a Databricks foundation model are
interchangeable without code changes.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Reasoning models (and their distillates) put their chain of thought in <think>…</think> before the
# answer. The user gets the answer only; an unfinished block (cut by max tokens) is dropped as well.
# Some models write <thinking>…</thinking> instead.
_THINK = re.compile(r"<(think|thinking)>.*?(</\1>|$)", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text or "").strip()


class LLMUnavailable(RuntimeError):
    """The endpoint cannot be reached – callers fall back to answers without a language model."""


class LLM:
    def __init__(self, cfg: dict, client=None):
        self.cfg = cfg
        self._client = client
        self._model = None if cfg["model"] == "auto" else cfg["model"]

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily: tests and the pipeline do not need it

            self._client = OpenAI(base_url=self.cfg["base_url"], api_key=self.cfg["api_key"],
                                  timeout=self.cfg["timeout_s"], max_retries=0)
        return self._client

    @property
    def model(self) -> str:
        """``auto``: the first model the server lists – in LM Studio, the one currently loaded."""
        if self._model is None:
            try:
                models = [m.id for m in self.client.models.list().data]
            except Exception as exc:  # connection refused, timeout, wrong URL …
                raise LLMUnavailable(f"No language model at {self.cfg['base_url']}: {exc}") from exc
            chat = [m for m in models if "embed" not in m.lower()]  # LM Studio also lists embedding models
            if not chat:
                raise LLMUnavailable(f"The server at {self.cfg['base_url']} has no model loaded")
            self._model = chat[0]
        return self._model

    def available(self) -> bool:
        try:
            return bool(self.model)
        except LLMUnavailable:
            return False

    def chat(self, system: str, user: str, history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
        try:
            r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                    temperature=self.cfg["temperature"])
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"Language model call failed: {exc}") from exc
        return strip_reasoning(r.choices[0].message.content)
