from __future__ import annotations

import json
import time

import requests
from pydantic import BaseModel, ValidationError

from . import config
from .document import OcrLine
from .prompts import classify_messages, extraction_messages
from .schemas import SCHEMAS, TipusValasz, inline_schema


class LlmError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str = config.LLM_MODEL, url: str = config.OLLAMA_URL, timeout: int = config.LLM_TIMEOUT):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def available(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException:
            return False, f"Az Ollama szerver nem érhető el ({self.url})."
        names = {m["name"] for m in r.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, f"A(z) {self.model} modell nincs letöltve (ollama pull {self.model})."
        return True, "ok"

    def chat_json(self, messages: list[dict], schema: dict, temperature: float = 0.0, num_ctx: int = 8192) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx, "seed": 0, "num_predict": 2500},
        }
        try:
            r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LlmError(f"LLM-hívás sikertelen: {e}") from e
        content = r.json().get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LlmError(f"Az LLM válasza nem érvényes JSON: {content[:200]}") from e


class Extractor:
    def __init__(self, client: OllamaClient | None = None, retries: int = config.LLM_RETRIES):
        self.client = client or OllamaClient()
        self.retries = retries
        self.last_stats: dict = {}

    def _call(self, messages: list[dict], model: type[BaseModel]) -> BaseModel:
        schema = inline_schema(model)
        error = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                data = self.client.chat_json(messages, schema, temperature=0.0 if attempt == 0 else 0.2)
                result = model.model_validate(data)
                self.last_stats = {"probalkozasok": attempt + 1, "ido": time.perf_counter() - start}
                return result
            except (LlmError, ValidationError) as e:
                error = e
        raise LlmError(f"Az LLM {self.retries + 1} próbálkozás után sem adott érvényes választ: {error}")

    def extract(self, doc_type: str, lines: list[OcrLine]) -> dict:
        if doc_type not in SCHEMAS:
            raise ValueError(f"Ismeretlen dokumentumtípus: {doc_type}")
        result = self._call(extraction_messages(doc_type, lines), SCHEMAS[doc_type])
        return result.model_dump()

    def classify(self, lines: list[OcrLine]) -> tuple[str, str]:
        result = self._call(classify_messages(lines), TipusValasz)
        return result.tipus, result.indoklas
