"""Local CPU LLM (llama.cpp, GGUF) with the same interface and disk cache as :class:`LLMClient`.

Used when no hosted endpoint is configured. Default model: Qwen2.5-1.5B-Instruct
(Q4_K_M, ~1.1 GB), which runs on a laptop CPU at ~10-20 tokens/s.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..config import PROJECT_ROOT

DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


class LocalLLM:
    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILE,
        n_ctx: int = 4096,
        n_threads: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 512,
        cache_dir: str | Path = PROJECT_ROOT / ".cache" / "llm",
    ):
        self.model = f"{repo}/{filename}"
        self.repo, self.filename = repo, filename
        self.n_ctx, self.n_threads = n_ctx, n_threads
        self.temperature, self.max_tokens = temperature, max_tokens
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            path = hf_hub_download(self.repo, self.filename)
            self._llm = Llama(
                model_path=path, n_ctx=self.n_ctx, n_threads=self.n_threads, verbose=False
            )
        return self._llm

    def _key(self, messages: list[dict], **kw) -> Path:
        blob = json.dumps({"m": self.model, "msg": messages, **kw}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    def chat(self, messages: list[dict], json_mode: bool = False, **kw) -> str:
        params = {
            "temperature": kw.pop("temperature", self.temperature),
            "max_tokens": kw.pop("max_tokens", self.max_tokens),
        }
        path = self._key(messages, json_mode=json_mode, **params)
        if path.exists():
            return json.loads(path.read_text())["content"]
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self.llm.create_chat_completion(messages=messages, **params, **extra)
        content = resp["choices"][0]["message"]["content"] or ""
        path.write_text(json.dumps({"content": content}))
        return content

    def chat_json(self, messages: list[dict], **kw) -> dict:
        text = self.chat(messages, json_mode=True, **kw)
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            try:
                out = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                out = {}
        return out if isinstance(out, dict) else {}
