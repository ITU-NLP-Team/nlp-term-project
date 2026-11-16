"""Batched offline LLM inference with vLLM (GPU), with the same disk cache as the other clients.

``chat_many`` takes a list of conversations and returns one completion per
conversation; cached conversations are not recomputed. Qwen3 models are run in
non-thinking mode (``enable_thinking=False``) so that outputs are directly the
requested JSON / ranking.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from ..config import PROJECT_ROOT


class VLLMChat:
    def __init__(
        self,
        model: str | None = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.55,
        cache_dir: str | Path = PROJECT_ROOT / ".cache" / "llm",
    ):
        self.model = model or os.getenv("CF_LLM", "Qwen/Qwen3-8B")
        self.max_model_len, self.gpu_mem = max_model_len, gpu_memory_utilization
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                self.model,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_mem,
                dtype="bfloat16",
                enable_prefix_caching=True,
            )
        return self._llm

    def _key(self, messages: list[dict], max_tokens: int, schema: dict | None = None) -> Path:
        d = {"m": self.model, "msg": messages, "mt": max_tokens}
        if schema is not None:
            d["schema"] = schema
        blob = json.dumps(d, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    def chat_many(
        self, convs: list[list[dict]], max_tokens: int = 512, schema: dict | None = None
    ) -> list[str]:
        """Greedy completions; ``schema`` constrains the output to that JSON schema."""
        out: list[str | None] = []
        todo: list[int] = []
        for i, c in enumerate(convs):
            p = self._key(c, max_tokens, schema)
            if p.exists():
                out.append(json.loads(p.read_text())["content"])
            else:
                out.append(None)
                todo.append(i)
        if todo:
            from vllm import SamplingParams

            gd = None
            if schema is not None:
                from vllm.sampling_params import GuidedDecodingParams

                gd = GuidedDecodingParams(json=schema)
            sp = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                guided_decoding=gd,
                repetition_penalty=1.05 if schema is not None else 1.0,
            )
            tok = self.llm.get_tokenizer()
            kw = {"enable_thinking": False} if "Qwen3" in self.model else {}
            prompts = [
                tok.apply_chat_template(convs[i], tokenize=False, add_generation_prompt=True, **kw)
                for i in todo
            ]
            res = self.llm.generate(prompts, sp, use_tqdm=True)
            for i, r in zip(todo, res):
                text = r.outputs[0].text
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                out[i] = text
                self._key(convs[i], max_tokens, schema).write_text(json.dumps({"content": text}))
        return [o or "" for o in out]

    def chat(self, messages: list[dict], max_tokens: int = 512, **_) -> str:
        return self.chat_many([messages], max_tokens)[0]

    @staticmethod
    def parse_json(text: str) -> dict:
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            try:
                out = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                out = {}
        return out if isinstance(out, dict) else {}

    def chat_json(self, messages: list[dict], max_tokens: int = 512, **_) -> dict:
        return self.parse_json(self.chat(messages, max_tokens))
