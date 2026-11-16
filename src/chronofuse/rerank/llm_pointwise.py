"""Pointwise LLM reranking ("cross-decoder"): Qwen3-Reranker.

A decoder-only LLM reads ``<Instruct> <Query> <Document>`` and the relevance
score is P("yes") at the answer position, i.e. softmax over the logits of the
"yes"/"no" tokens (Zhang et al., 2025, Qwen3 Embedding/Reranker). Unlike a
BERT cross-encoder, the instruction lets us state RETECO's notion of relevance
(right topic *and* right period). Runs on CPU in fp32; meant for the top 30-100
candidates.
"""

from __future__ import annotations

import torch

DEFAULT_INSTRUCTION = (
    "Given a time-sensitive question from a StackExchange forum, judge whether the document "
    "contains evidence that helps answer it, including information about the relevant events "
    "and time period"
)

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LLMPointwiseReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str = DEFAULT_INSTRUCTION,
        max_length: int = 512,
        batch_size: int = 8,
        threads: int = 4,
        device: str = "cpu",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cpu":
            torch.set_num_threads(threads)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        )
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.instruction, self.max_length, self.batch_size = instruction, max_length, batch_size
        self.pre = self.tok.encode(_PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(_SUFFIX, add_special_tokens=False)

    def _encode(self, query: str, doc: str) -> list[int]:
        q = self.tok.encode(
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n", add_special_tokens=False
        )[: self.max_length // 3]
        budget = self.max_length - len(self.pre) - len(self.suf) - len(q)
        d = self.tok.encode(f"<Document>: {doc}", add_special_tokens=False)[: max(16, budget)]
        return self.pre + q + d + self.suf

    @torch.inference_mode()
    def score(self, query: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(docs), self.batch_size):
            ids = [self._encode(query, d) for d in docs[i : i + self.batch_size]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :].float()
            two = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            out.extend(torch.log_softmax(two, dim=1)[:, 1].exp().tolist())
        return out
