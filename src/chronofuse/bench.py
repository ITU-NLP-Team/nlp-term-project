"""Staged benchmark runner with per-stage caching (CPU or GPU).

Every retrieval *stage* (a first-stage ranker, an LLM plan, a reranker, ...)
is computed once per (domain, split, sub-track) and cached as JSON under
``results/cache/<domain>/<split>_<sub>/<stage>.json``. *Systems* are cheap
recipes over cached stages (fusion, priors) evaluated with the official metric
(:mod:`chronofuse.systems`). This lets dozens of configurations be compared on
the full RETECO Track 1 data.

    python -m chronofuse.bench stage --stage bm25 dense plan_fb --split dev --subtrack 1a 1b
    python -m chronofuse.bench stage --stage pool q3rr4 --split train dev  # rerank BM25 ⊕ dense
    python -m chronofuse.bench select     # train: pool channels + best reranker
    python -m chronofuse.bench stage --stage pool_sel q3rr4@sel listwise --split train dev
    python -m chronofuse.bench report --split dev

Models (override with environment variables):
    CF_EMBED   dense encoder          Qwen/Qwen3-Embedding-0.6B (GPU) | BAAI/bge-small-en-v1.5 (CPU)
    CF_LLM     generator (vLLM, GPU)  Qwen/Qwen3-8B                   | local Qwen2.5-1.5B GGUF (CPU)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import orjson

from .config import resolve_path
from .data.reteco import TRACK1_DOMAINS, Query, load_corpus, load_guidance, load_queries, load_steps
from .indexing.base import Ranking, maxp
from .indexing.bm25_index import BM25Index
from .metadata.chunking import chunk_document
from .metadata.profile import DocProfile, extract_entities, read_profiles
from .metadata.temporal import extract_temporal
from .retrieval.fusion import rrf
from .store import DocStore

log = logging.getLogger("chronofuse.bench")

DATA = "data/raw/reteco"
CHUNK = (200, 40, 64)
DEPTH = 100


def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================== topics ====
@dataclass
class Topic:
    qid: str
    text: str  # retrieval text
    query: Query  # parent query (1b) or the query itself (1a)
    instruction: str = ""
    years: list[int] = field(default_factory=list)


def load_topics(domain: str, split: str, sub: str) -> list[Topic]:
    root = resolve_path(DATA)
    out: list[Topic] = []
    if sub == "1a":
        for q in load_queries(root, domain, split):
            out.append(Topic(q.id, q.text, q, years=extract_temporal(q.text).years))
    else:
        for st in load_steps(root, domain, split):
            out.append(
                Topic(
                    st.id,
                    f"{st.query.title} {st.instruction}",
                    st.query,
                    st.instruction,
                    extract_temporal(st.query.text).years,
                )
            )
    return out


def rerank_query(t: Topic, max_words: int = 250) -> str:
    """What rerankers / encoders see: title + body start (1a) or title + step (1b)."""
    if t.instruction:
        return f"{t.query.title}\nStep: {t.instruction}"
    return " ".join(t.text.split()[:max_words])


# ============================================================= context ====
class DomainContext:
    """Lazily loaded per-domain resources."""

    def __init__(self, domain: str):
        self.domain = domain
        self.dir = resolve_path("data/indexes") / domain
        self._bm25 = self._prof = self._store = self._yi = self._prior = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load(self.dir / "bm25")
            pdoc, docs = self._bm25.passage_doc_index()
            self.doc_index = {d: i for i, d in enumerate(docs)}
            starts = np.flatnonzero(np.r_[True, np.diff(pdoc) != 0])
            self.p_start, self.p_end = starts, np.r_[starts[1:], len(pdoc)]
        return self._bm25

    @property
    def profiles(self) -> dict[str, DocProfile]:
        if self._prof is None:
            self._prof = read_profiles(self.dir / "profiles.jsonl")
        return self._prof

    @property
    def store(self) -> DocStore:
        if self._store is None:
            self._store = DocStore(self.dir / "docs.sqlite")
        return self._store

    @property
    def year_index(self):
        if self._yi is None:
            from .metadata.year_index import YearIndex

            pdoc, docs = self.bm25.passage_doc_index()
            self._yi = YearIndex(pdoc, docs, self.profiles)
        return self._yi

    @property
    def year_prior(self) -> dict[int, float]:
        if self._prior is None:
            from .retrieval.prf import corpus_year_prior

            self._prior = corpus_year_prior(self.profiles)
        return self._prior

    def best_passages(self, weights: dict[str, float], docs: list[str]) -> dict[str, str]:
        """Text of each document's highest-BM25 passage for the weighted query."""
        scores = self.bm25.score_terms(weights)
        texts = self.store.get_many(docs)
        out: dict[str, str] = {}
        for d in docs:
            i = self.doc_index.get(d)
            ps = list(chunk_document(d, texts.get(d, ""), *CHUNK))
            if i is None or not ps:
                out[d] = ps[0].text if ps else ""
                continue
            k = int(np.argmax(scores[self.p_start[i] : self.p_end[i]]))
            out[d] = ps[min(k, len(ps) - 1)].text
        return out

    def render(self, d: str, passage: str, max_words: int = 200) -> str:
        p = self.profiles.get(d)
        head = ""
        if p is not None:
            yrs = f"[{p.years[0]}-{p.years[-1]}] " if p.years else ""
            head = f"{yrs}{p.title[:150]} | "
        return head + " ".join(passage.split()[:max_words])

    def candidate_texts(self, t: Topic, docs: list[str], max_words: int = 200) -> list[str]:
        ps = self.best_passages(self.bm25.query_weights(t.text), docs)
        return [self.render(d, ps[d], max_words) for d in docs]


# =============================================================== cache ====
def cache_path(domain: str, split: str, sub: str, stage: str) -> Path:
    root = resolve_path(os.getenv("CF_CACHE", "results/cache"))
    return root / domain / f"{split}_{sub}" / f"{stage}.json"


def load_stage(domain: str, split: str, sub: str, stage: str) -> dict | None:
    p = cache_path(domain, split, sub, stage)
    return orjson.loads(p.read_bytes()) if p.exists() else None


def save_stage(domain: str, split: str, sub: str, stage: str, obj: dict) -> None:
    p = cache_path(domain, split, sub, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(orjson.dumps(obj))


# ============================================================== models ====
_MODELS: dict[str, object] = {}

EMBED_TASK = (
    "Given a time-sensitive question from a StackExchange forum, retrieve passages that help "
    "answer it"
)
RERANKERS = {
    "ce_minilm": ("ce", "cross-encoder/ms-marco-MiniLM-L12-v2", 512),
    "ce_bge": ("ce", "BAAI/bge-reranker-v2-m3", 1024),
    "q3rr06": ("dec", "Qwen/Qwen3-Reranker-0.6B", 1024),
    "q3rr4": ("dec", "Qwen/Qwen3-Reranker-4B", 1024),
}


class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer

        self.gpu = has_gpu()
        self.name = os.getenv(
            "CF_EMBED", "Qwen/Qwen3-Embedding-0.6B" if self.gpu else "BAAI/bge-small-en-v1.5"
        )
        kw = {}
        if self.gpu:
            kw = {"model_kwargs": {"torch_dtype": torch.float16}}
        if "Qwen3" in self.name:
            kw["tokenizer_kwargs"] = {"padding_side": "left"}
        self.m = SentenceTransformer(self.name, device="cuda" if self.gpu else "cpu", **kw)
        self.m.max_seq_length = 384 if self.gpu else 256
        self.bs = 256 if self.gpu else 32

    def queries(self, texts: list[str]) -> np.ndarray:
        if "Qwen3" in self.name:
            p = f"Instruct: {EMBED_TASK}\nQuery:"
            return self.m.encode(texts, prompt=p, batch_size=64, normalize_embeddings=True)
        if "bge" in self.name:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return self.m.encode(texts, batch_size=64, normalize_embeddings=True)

    def docs(self, texts: list[str]) -> np.ndarray:
        return self.m.encode(texts, batch_size=self.bs, normalize_embeddings=True)


def model(name: str):
    if name not in _MODELS:
        import torch

        if not has_gpu():
            torch.set_num_threads(4)
        dev = "cuda" if has_gpu() else "cpu"
        if name == "embed":
            m = Embedder()
        elif name in RERANKERS:
            kind, hf, maxlen = RERANKERS[name]
            if kind == "ce":
                from sentence_transformers import CrossEncoder

                kw = {"model_kwargs": {"torch_dtype": torch.float16}} if dev == "cuda" else {}
                m = CrossEncoder(hf, max_length=maxlen, device=dev, **kw)
            else:
                from .rerank.llm_pointwise import LLMPointwiseReranker

                m = LLMPointwiseReranker(
                    hf, max_length=maxlen, device=dev, batch_size=16 if dev == "cuda" else 8
                )
        elif name == "llm":
            if has_gpu():
                from .llm.vllm_client import VLLMChat

                m = VLLMChat()
            else:
                from .llm.local import LocalLLM

                m = LocalLLM()
        elif name == "rir":
            from .retrieval.reasonir import RIREncoder

            m = RIREncoder()
        elif name == "small_llm":
            from .metadata.llm_meta import small_llm

            m = small_llm()
        else:
            raise KeyError(name)
        _MODELS[name] = m
    return _MODELS[name]


def free_model(name: str) -> None:
    if name in _MODELS:
        del _MODELS[name]
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


# ============================================================== stages ====
StageFn = Callable[[DomainContext, list[Topic], str, str], dict]
STAGES: dict[str, StageFn] = {}


def stage(name: str):
    def deco(fn: StageFn) -> StageFn:
        STAGES[name] = fn
        return fn

    return deco


def _rank(ctx: DomainContext, w: dict[str, float], k: int = DEPTH, mask=None) -> list:
    return [[d, round(float(s), 5)] for d, s in ctx.bm25.search_weighted(w, k, mask)]


# ------------------------------------------------------------ lexical ----
@stage("bm25")
def st_bm25(ctx, topics, split, sub):
    return {t.qid: _rank(ctx, ctx.bm25.query_weights(t.text)) for t in topics}


@stage("bm25_full")  # 1b: full query text + step instruction (the official 1b input)
def st_bm25_full(ctx, topics, split, sub):
    return {
        t.qid: _rank(ctx, ctx.bm25.query_weights(f"{t.query.text} {t.instruction}")) for t in topics
    }


def _year_stage(gamma: float) -> StageFn:
    def fn(ctx, topics, split, sub):
        from .retrieval.prf import downweight_years

        return {
            t.qid: _rank(ctx, downweight_years(ctx.bm25.query_weights(t.text), gamma))
            for t in topics
        }

    return fn


STAGES["bm25_y0"] = _year_stage(0.0)
STAGES["bm25_y05"] = _year_stage(0.5)


def _doc_tokens(ctx: DomainContext, docs: list[str], max_words: int = 600) -> dict:
    texts = ctx.store.get_many(docs)
    return {d: ctx.bm25.analyze(" ".join(texts.get(d, "").split()[:max_words])) for d in docs}


def _rm3_stage(alpha: float, fb_terms: int = 20) -> StageFn:
    def fn(ctx, topics, split, sub):
        from .retrieval.prf import rm3_weights

        base = load_stage(ctx.domain, split, sub, "bm25")
        out = {}
        for t in topics:
            first = [(d, s) for d, s in base.get(t.qid, [])]
            toks = _doc_tokens(ctx, [d for d, _ in first[:10]])
            w = rm3_weights(ctx.bm25.query_weights(t.text), first, toks, 10, fb_terms, alpha)
            out[t.qid] = _rank(ctx, w)
        return out

    return fn


STAGES["rm3_a03"] = _rm3_stage(0.3)
STAGES["rm3_a05"] = _rm3_stage(0.5)
STAGES["rm3_a07"] = _rm3_stage(0.7)


# ----------------------------------------------------------- temporal ----
@stage("tprf")  # inferred year windows (temporal PRF), not a ranking
def st_tprf(ctx, topics, split, sub):
    from .retrieval.prf import temporal_feedback

    base = load_stage(ctx.domain, split, sub, "bm25")
    return {
        t.qid: temporal_feedback(
            [(d, s) for d, s in base.get(t.qid, [])], ctx.profiles, ctx.year_prior
        )
        for t in topics
    }


@stage("timefilter")  # BM25 restricted by the year index to the query's explicit window
def st_timefilter(ctx, topics, split, sub):
    out = {}
    for t in topics:
        if not t.years or t.years[-1] - t.years[0] > 60:
            out[t.qid] = []
            continue
        mask = ctx.year_index.passage_mask((t.years[0], t.years[-1]), 1)
        out[t.qid] = _rank(ctx, ctx.bm25.query_weights(t.text), mask=mask)
    return out


@stage("kg")
def st_kg(ctx, topics, split, sub):
    from .kg.temporal_kg import TemporalKG

    path = ctx.dir / "kg.pkl"
    if path.exists():
        kg = TemporalKG.load(path)
    else:
        kg = TemporalKG(max_entity_df=0.05).build(ctx.profiles.values())
        kg.save(path)
    out = {}
    for t in topics:
        ents = extract_entities(t.query.text, 10)
        ys = t.years if len(t.years) <= 10 else []
        out[t.qid] = [[d, round(s, 8)] for d, s in kg.search(ents, ys, DEPTH)]
    del kg
    gc.collect()
    return out


# -------------------------------------------------------------- dense ----
def dense_dir(ctx: DomainContext) -> Path:
    name = model("embed").name.split("/")[-1]
    return ctx.dir / f"dense_{name}"


def iter_passages(ctx: DomainContext):
    """(pid, text) of every passage of the domain, in index order; passages after the first
    carry the document title as context (the encoder input of every dense channel)."""
    for doc in load_corpus(resolve_path(DATA), ctx.domain):
        p = ctx.profiles.get(doc.id)
        head = f"{p.title[:120]}\n" if p is not None else ""
        for ps in chunk_document(doc.id, doc.content, *CHUNK):
            yield ps.pid, (head + ps.text if ps.index else ps.text)


def passage_embeddings(ctx: DomainContext) -> tuple[list[str], np.ndarray]:
    """Embed every passage of the domain (same chunking as BM25); cached as fp16."""
    path = dense_dir(ctx)
    if (path / "emb.npy").exists():
        return orjson.loads((path / "pids.json").read_bytes()), np.load(path / "emb.npy")
    enc = model("embed")
    pids, embs, buf = [], [], []
    t0 = time.time()

    def flush():
        if buf:
            embs.append(enc.docs([b[1] for b in buf]).astype(np.float16))
            pids.extend(b[0] for b in buf)
            buf.clear()
            log.info("[%s] dense %d passages (%.0fs)", ctx.domain, len(pids), time.time() - t0)

    for pid, text in iter_passages(ctx):
        buf.append((pid, text))
        if len(buf) >= 20000:
            flush()
    flush()
    emb = np.vstack(embs)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "emb.npy", emb)
    (path / "pids.json").write_bytes(orjson.dumps(pids))
    return pids, emb


class DenseSearcher:
    def __init__(self, pids: list[str], emb: np.ndarray):
        self.pids = pids
        if has_gpu():
            import torch

            self.emb = torch.from_numpy(emb).cuda()
        else:
            self.emb = emb.astype(np.float32)

    def search(self, qv: np.ndarray, k: int = DEPTH, mask: np.ndarray | None = None) -> Ranking:
        """Top-k documents (MaxP); ``mask`` (bool per passage) restricts the candidates."""
        n = min(k * 5, len(self.pids))
        if has_gpu():
            import torch

            q = torch.from_numpy(qv.astype(np.float16)).cuda()
            sc = self.emb @ q
            if mask is not None:
                sc = sc.masked_fill(~torch.from_numpy(mask).cuda(), float("-inf"))
            s, i = torch.topk(sc, n)
            s, i = s.float().cpu().numpy(), i.cpu().numpy()
        else:
            sc = self.emb @ qv
            if mask is not None:
                sc = np.where(mask, sc, -np.inf)
            i = np.argpartition(-sc, n - 1)[:n]
            s = sc[i]
        keep = np.isfinite(s)
        i, s = i[keep], s[keep]
        return maxp([self.pids[j] for j in i], s.tolist(), k)


@stage("dense")  # full-corpus passage-level dense retrieval + MaxP
def st_dense(ctx, topics, split, sub):
    pids, emb = passage_embeddings(ctx)
    ds = DenseSearcher(pids, emb)
    q = model("embed").queries([rerank_query(t) for t in topics])
    out = {t.qid: [[d, round(s, 5)] for d, s in ds.search(qv)] for t, qv in zip(topics, q)}
    del ds
    return out


_MEMO: dict = {}


def doc_embeddings(ctx: DomainContext) -> tuple[list[str], np.ndarray]:
    """Document vectors = normalised mean of their passage vectors (memoised per domain)."""
    key = ("docemb", ctx.domain)
    if key in _MEMO:
        return _MEMO[key]
    pids, emb = passage_embeddings(ctx)
    ids: dict[str, int] = {}
    di = np.fromiter((ids.setdefault(p.rsplit("#", 1)[0], len(ids)) for p in pids), np.int64)
    out = np.zeros((len(ids), emb.shape[1]), dtype=np.float32)
    np.add.at(out, di, emb.astype(np.float32))
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    _MEMO[key] = (list(ids), out)
    return _MEMO[key]


@stage("dense_doc")
def st_dense_doc(ctx, topics, split, sub):
    ids, emb = doc_embeddings(ctx)
    q = model("embed").queries([rerank_query(t) for t in topics])
    out = {}
    for t, qv in zip(topics, q):
        s = emb @ qv.astype(np.float32)
        top = np.argpartition(-s, DEPTH)[:DEPTH]
        top = top[np.argsort(-s[top])]
        out[t.qid] = [[ids[i], round(float(s[i]), 5)] for i in top]
    return out


def _raptor_stage(beta: float, use_window: bool) -> StageFn:
    def fn(ctx, topics, split, sub):
        from .retrieval.raptor import TemporalRaptor

        key = ("raptor", ctx.domain, beta)
        if key not in _MEMO:
            ids, emb = doc_embeddings(ctx)
            _MEMO[key] = TemporalRaptor(beta=beta).build(ids, emb, ctx.profiles)
        tree = _MEMO[key]
        q = model("embed").queries([rerank_query(t) for t in topics])
        tprf = load_stage(ctx.domain, split, sub, "tprf") or {}
        out = {}
        for t, qv in zip(topics, q):
            win = None
            if use_window:
                win = (t.years[0], t.years[-1]) if t.years else tprf.get(t.qid)
            out[t.qid] = [[d, round(s, 5)] for d, s in tree.search(qv, DEPTH, window=win)]
        return out

    return fn


STAGES["raptor"] = _raptor_stage(0.0, False)  # RAPTOR-style tree, no time
STAGES["raptor_t"] = _raptor_stage(0.5, True)  # temporal clustering + period bonus


# ---------------------------------------------------------------- LLM ----
def _planner(ctx):
    """Planner whose single demonstration is a train query of this domain with its guidance
    and the text of its first gold document (train labels only)."""
    from .data.reteco import load_qrels
    from .query.planner import QueryPlanner

    root = resolve_path(DATA)
    g = load_guidance(root, ctx.domain, "train")
    qrels = load_qrels(root, ctx.domain, "train")
    tr = sorted(
        (q for q in load_queries(root, ctx.domain, "train") if q.id in g and qrels.get(q.id)),
        key=lambda q: len(q.text),
    )
    if not tr:
        return QueryPlanner(model("llm"))
    q = tr[len(tr) // 2]
    gold = [d for d, r in qrels[q.id].items() if r > 0]
    text = ctx.store.get_many(gold[:1]).get(gold[0], "") if gold else ""
    return QueryPlanner(model("llm"), demo=(q, g[q.id], text))


def _plan_stage(feedback: bool) -> StageFn:
    def fn(ctx, topics, split, sub):
        from .query.planner import normalise_plan

        pl = _planner(ctx)
        base = load_stage(ctx.domain, split, "1a", "bm25") if feedback else {}
        if base is None:
            raise RuntimeError(f"plan_fb needs the 1a bm25 stage of {ctx.domain}/{split}")
        parents: dict[str, Query] = {}
        for t in topics:
            parents.setdefault(t.query.id, t.query)
        convs = []
        for qid, q in parents.items():
            fb = None
            if feedback:
                docs = [d for d, _ in base.get(qid, [])[:5]]
                texts = ctx.store.get_many(docs)
                fb = [" ".join(texts.get(d, "").split()[:60]) for d in docs]
            convs.append(pl.messages(q, fb))
        llm = pl.llm
        from .query.planner import PLAN_SCHEMA, parse_plan_text

        if hasattr(llm, "chat_many"):
            try:
                texts = llm.chat_many(convs, max_tokens=900, schema=PLAN_SCHEMA)
            except Exception:  # noqa: BLE001 - guided decoding unavailable: plain decoding
                log.exception("guided decoding failed; falling back to free decoding")
                texts = llm.chat_many(convs, max_tokens=900)
            raw = [parse_plan_text(x) for x in texts]
        else:
            raw = [parse_plan_text(llm.chat(c, max_tokens=600)) for c in convs]
        plans = {qid: normalise_plan(r) for qid, r in zip(parents, raw)}
        return {t.qid: plans[t.query.id] for t in topics}

    return fn


STAGES["plan"] = _plan_stage(False)
STAGES["plan_fb"] = _plan_stage(True)  # LLM pseudo-relevance feedback (BM25 snippets)


def _plan_channels(plan_name: str) -> dict[str, StageFn]:
    def get(ctx, split, sub):
        return load_stage(ctx.domain, split, sub, plan_name) or {}

    def kw(ctx, topics, split, sub):
        plans, out = get(ctx, split, sub), {}
        for t in topics:
            w = ctx.bm25.query_weights(t.text)
            for term, c in ctx.bm25.query_weights(
                " ".join(plans.get(t.qid, {}).get("keywords", []))
            ).items():
                w[term] = w.get(term, 0.0) + c
            out[t.qid] = _rank(ctx, w)
        return out

    def q2d(ctx, topics, split, sub):  # query2doc: query + pseudo answer + keywords
        plans, out = get(ctx, split, sub), {}
        for t in topics:
            p = plans.get(t.qid, {})
            w = ctx.bm25.query_weights(t.text)
            extra = " ".join([p.get("answer_passage", "")] + p.get("keywords", []))
            for term, c in ctx.bm25.query_weights(extra).items():
                w[term] = w.get(term, 0.0) + c
            out[t.qid] = _rank(ctx, w)
        return out

    def multi(ctx, topics, split, sub):  # multi-query + event chain, fused with RRF
        plans, out = get(ctx, split, sub), {}
        for t in topics:
            p = plans.get(t.qid, {})
            qs = [t.text] + p.get("sub_queries", [])
            qs += [f"{e['event']} {e['year'] or ''}".strip() for e in p.get("events", [])]
            runs = {
                str(i): ctx.bm25.search_weighted(ctx.bm25.query_weights(q), DEPTH)
                for i, q in enumerate(qs)
                if q.strip()
            }
            out[t.qid] = [[d, round(s, 6)] for d, s in rrf(runs, {"0": 2.0}, top=DEPTH)]
        return out

    sfx = "" if plan_name == "plan" else "_fb"
    return {f"llm_kw{sfx}": kw, f"llm_q2d{sfx}": q2d, f"llm_multi{sfx}": multi}


STAGES.update(_plan_channels("plan"))
STAGES.update(_plan_channels("plan_fb"))


# ------------------------------------- LLM metadata + neighbouring events ----
# A small LLM extracts {period, events, entities, keywords, topic} from every document
# (chronofuse.metadata.llm_meta, offline) and from every query (stage ``qmeta``); the planner
# LLM expands queries with neighbouring events and period-anchored sub-queries (``nbr``).
def need(ctx: DomainContext, split: str, sub: str, name: str) -> dict:
    """A cached stage this stage depends on; missing -> fail (instead of a silent fallback)."""
    out = load_stage(ctx.domain, split, sub, name)
    if out is None:
        raise RuntimeError(f"needs stage {name} for {ctx.domain}/{split}_{sub}")
    return out


def _parent_queries(topics: list[Topic]) -> dict[str, Query]:
    out: dict[str, Query] = {}
    for t in topics:
        out.setdefault(t.query.id, t.query)
    return out


@stage("qmeta")  # small LLM, same schema as the documents
def st_qmeta(ctx, topics, split, sub):
    from .metadata.llm_meta import META_SCHEMA, generate, meta_messages, normalise_meta

    parents = _parent_queries(topics)
    outs = generate(
        model("small_llm"),
        [meta_messages(q.text, 300) for q in parents.values()],
        META_SCHEMA,
        200,
        cache=True,
    )
    m = {qid: normalise_meta(o) for qid, o in zip(parents, outs)}
    return {t.qid: m[t.query.id] for t in topics}


@stage("nbr")  # planner LLM + BM25 snippets: neighbouring events, sub-queries, terms
def st_nbr(ctx, topics, split, sub):
    from .metadata.llm_meta import EXPAND_SCHEMA, expand_messages, generate, normalise_expansion

    base = load_stage(ctx.domain, split, "1a", "bm25")
    if base is None:
        raise RuntimeError(f"nbr needs the 1a bm25 stage of {ctx.domain}/{split}")
    parents = _parent_queries(topics)
    convs = []
    for qid, q in parents.items():
        docs = [d for d, _ in base.get(qid, [])[:5]]
        texts = ctx.store.get_many(docs)
        convs.append(
            expand_messages(q.text, [" ".join(texts.get(d, "").split()[:60]) for d in docs])
        )
    outs = generate(model("llm"), convs, EXPAND_SCHEMA, 700, cache=True)
    ex = {qid: normalise_expansion(o) for qid, o in zip(parents, outs)}
    return {t.qid: ex[t.query.id] for t in topics}


def _doc_meta(ctx: DomainContext) -> dict[str, dict]:
    """The domain's LLM document metadata; refuses to run on an incomplete extraction."""
    if getattr(ctx, "_meta", None) is None:
        from .metadata.llm_meta import load_doc_meta

        m = load_doc_meta(ctx.domain)
        _, docs = ctx.bm25.passage_doc_index()
        if len(m) < 0.98 * len(docs):
            raise RuntimeError(f"LLM metadata incomplete for {ctx.domain}: {len(m)}/{len(docs)}")
        ctx._meta = m
    return ctx._meta


def _doc_periods(ctx: DomainContext) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) year per document of the BM25 document list; -1 = no period."""
    if getattr(ctx, "_per", None) is None:
        m = _doc_meta(ctx)
        _, docs = ctx.bm25.passage_doc_index()
        sy = np.array([(m.get(d) or {}).get("sy") or -1 for d in docs], dtype=np.int32)
        ey = np.array([(m.get(d) or {}).get("ey") or -1 for d in docs], dtype=np.int32)
        ctx._per = (sy, ey)
    return ctx._per


TIME_TOL = 2  # years of slack around the query period


def query_window(t: Topic, qmeta: dict) -> tuple[int, int] | None:
    """LLM query period, else the explicit years of the query (if they form a usable window)."""
    m = qmeta.get(t.qid) or {}
    if m.get("sy"):
        return m["sy"], m["ey"]
    if t.years and t.years[-1] - t.years[0] <= 60:
        return t.years[0], t.years[-1]
    return None


def period_doc_mask(ctx: DomainContext, window: tuple[int, int]) -> np.ndarray:
    """Documents whose LLM period overlaps the window (± TIME_TOL), or that have no period."""
    sy, ey = _doc_periods(ctx)
    a, b = window[0] - TIME_TOL, window[1] + TIME_TOL
    return (sy < 0) | ((sy <= b) & (ey >= a))


def _add(w: dict[str, float], extra: dict[str, float], scale: float = 1.0) -> dict[str, float]:
    w = dict(w)
    for k, v in extra.items():
        w[k] = w.get(k, 0.0) + scale * v
    return w


@stage("meta_kw")  # BM25: query + small-LLM events / entities / keywords
def st_meta_kw(ctx, topics, split, sub):
    from .metadata.llm_meta import meta_terms

    qm = need(ctx, split, sub, "qmeta")
    qw = ctx.bm25.query_weights
    return {t.qid: _rank(ctx, _add(qw(t.text), qw(meta_terms(qm.get(t.qid))))) for t in topics}


def meta_index(ctx: DomainContext) -> BM25Index:
    """Document-level BM25 over title + LLM metadata (events, entities, keywords, topic)."""
    from .metadata.llm_meta import meta_terms

    path = ctx.dir / "meta_bm25"
    if (path / "pids.json").exists():
        return BM25Index.load(path)
    m = _doc_meta(ctx)
    ids = list(m)
    texts = []
    for d in ids:
        p = ctx.profiles.get(d)
        texts.append(f"{p.title[:150] if p is not None else ''} {meta_terms(m[d])}")
    idx = BM25Index().build([f"{d}#0" for d in ids], texts)
    idx.save(path)
    return idx


@stage("meta_field")
def st_meta_field(ctx, topics, split, sub):
    from .metadata.llm_meta import meta_terms

    idx = meta_index(ctx)
    qm = need(ctx, split, sub, "qmeta")
    out = {}
    for t in topics:
        w = _add(idx.query_weights(t.text), idx.query_weights(meta_terms(qm.get(t.qid))))
        out[t.qid] = [[d, round(float(s), 5)] for d, s in idx.search_weighted(w, DEPTH)]
    return out


@stage("meta_tf_bm25")  # BM25 restricted to documents whose LLM period fits the query's
def st_meta_tf_bm25(ctx, topics, split, sub):
    qm = need(ctx, split, sub, "qmeta")
    base = need(ctx, split, sub, "bm25")
    pdoc, _ = ctx.bm25.passage_doc_index()
    out = {}
    for t in topics:
        win = query_window(t, qm)
        if win is None:  # no period: the plain BM25 ranking
            out[t.qid] = base.get(t.qid, [])
            continue
        mask = period_doc_mask(ctx, win)[pdoc].astype(np.float32)
        out[t.qid] = _rank(ctx, ctx.bm25.query_weights(t.text), mask=mask)
    return out


@stage("meta_tf_dense")  # dense retrieval with the same period filter
def st_meta_tf_dense(ctx, topics, split, sub):
    qm = need(ctx, split, sub, "qmeta")
    base = need(ctx, split, sub, "dense")
    pids, emb = passage_embeddings(ctx)
    ctx.bm25  # noqa: B018 - sets ctx.doc_index
    pd = np.fromiter((ctx.doc_index[p.rsplit("#", 1)[0]] for p in pids), np.int64, len(pids))
    ds = DenseSearcher(pids, emb)
    todo = [t for t in topics if query_window(t, qm) is not None]
    q = model("embed").queries([rerank_query(t) for t in todo]) if todo else []
    out = {t.qid: base.get(t.qid, []) for t in topics}
    for t, qv in zip(todo, q):
        mask = period_doc_mask(ctx, query_window(t, qm))[pd]
        out[t.qid] = [[d, round(s, 5)] for d, s in ds.search(qv, mask=mask)]
    del ds
    return out


def _nbr_queries(e: dict) -> list[str]:
    qs = list(e.get("sub_queries", []))
    qs += [f"{n['event']} {n['year'] or ''}".strip() for n in e.get("neighbors", [])]
    return [q for q in qs if q.strip()]


@stage("nbr_bm25")  # BM25: query (x2) ⊕ each sub-query / neighbouring event ⊕ query + terms
def st_nbr_bm25(ctx, topics, split, sub):
    ex = need(ctx, split, sub, "nbr")
    base = need(ctx, split, sub, "bm25")
    qw = ctx.bm25.query_weights
    out = {}
    for t in topics:
        e = ex.get(t.qid, {})
        runs = {"q": base.get(t.qid, [])}
        if e.get("terms"):
            runs["terms"] = ctx.bm25.search_weighted(_add(qw(t.text), qw(" ".join(e["terms"]))))
        for i, s in enumerate(_nbr_queries(e)):
            runs[str(i)] = ctx.bm25.search_weighted(qw(s), DEPTH)
        out[t.qid] = [[d, round(s, 6)] for d, s in rrf(runs, {"q": 2.0}, top=DEPTH)]
    return out


@stage("nbr_hybrid")  # as nbr_bm25, plus dense retrieval for the query and every sub-query
def st_nbr_hybrid(ctx, topics, split, sub):
    ex = need(ctx, split, sub, "nbr")
    lex = load_stage(ctx.domain, split, sub, "nbr_bm25")
    dense = load_stage(ctx.domain, split, sub, "dense")
    if lex is None or dense is None:
        raise RuntimeError("nbr_hybrid needs the nbr_bm25 and dense stages")
    subs = {t.qid: _nbr_queries(ex.get(t.qid, {})) for t in topics}
    flat = [s for t in topics for s in subs[t.qid]]
    vecs = iter(model("embed").queries(flat)) if flat else iter([])
    pids, emb = passage_embeddings(ctx)
    ds = DenseSearcher(pids, emb)
    out = {}
    for t in topics:
        runs = {"lex": lex.get(t.qid, []), "dense": dense.get(t.qid, [])}
        for i, _ in enumerate(subs[t.qid]):
            runs[f"d{i}"] = ds.search(next(vecs))
        out[t.qid] = [[d, round(s, 6)] for d, s in rrf(runs, {"lex": 2.0, "dense": 2.0}, top=DEPTH)]
    del ds
    return out


# ------------------------------------------- ReasonIR + LLM reasoning ----
@stage("reason")  # planner LLM: free-text reasoning about the question (BRIGHT/TEMPO style)
def st_reason(ctx, topics, split, sub):
    from .retrieval.reasonir import REASON_PROMPT

    parents = _parent_queries(topics)
    convs = [
        [{"role": "user", "content": REASON_PROMPT.format(query=q.text[:3000])}]
        for q in parents.values()
    ]
    outs = model("llm").chat_many(convs, max_tokens=600)
    r = {qid: " ".join(o.split()[:350]) for qid, o in zip(parents, outs)}
    return {t.qid: {"reasoning": r[t.query.id]} for t in topics}


def rir_query(t: Topic, reasoning: str = "") -> str:
    """ReasonIR query: the full question (1b: + the step, like TEMPO's Query+Step) and,
    optionally, the LLM's reasoning."""
    q = " ".join(t.query.text.split()[:400])
    if t.instruction:
        q += f"\nStep: {t.instruction}"
    return q + (f"\n{reasoning}" if reasoning else "")


def _rir_stage(with_reasoning: bool) -> StageFn:
    def fn(ctx, topics, split, sub):
        from .retrieval.reasonir import load_corpus_emb

        rs = need(ctx, split, sub, "reason") if with_reasoning else {}
        pids, emb = load_corpus_emb(ctx.domain)
        q = model("rir").queries(
            [rir_query(t, (rs.get(t.qid) or {}).get("reasoning", "")) for t in topics]
        )
        ds = DenseSearcher(pids, emb)
        out = {t.qid: [[d, round(s, 5)] for d, s in ds.search(qv)] for t, qv in zip(topics, q)}
        del ds, emb
        return out

    return fn


STAGES["rir"] = _rir_stage(False)
STAGES["rir_reason"] = _rir_stage(True)


# ------------------------------------------------- first-stage pools ----
SELECTION = "results/metrics/pool_selection.json"
POOL_A = ["bm25", "dense"]  # fixed hybrid pool: rerankers are compared on it


def selection() -> dict:
    p = resolve_path(SELECTION)
    return json.loads(p.read_text()) if p.exists() else {}


def _pool_stage(weights_fn) -> StageFn:
    def fn(ctx, topics, split, sub):
        weights = weights_fn(sub)
        runs = {c: load_stage(ctx.domain, split, sub, c) for c in weights}
        missing = [c for c, r in runs.items() if r is None]
        if missing:
            raise RuntimeError(f"pool needs stages {missing} for {ctx.domain}/{split}_{sub}")
        out = {}
        for t in topics:
            fused = rrf({c: r.get(t.qid, []) for c, r in runs.items()}, weights, top=DEPTH)
            out[t.qid] = [[d, round(s, 6)] for d, s in fused]
        return out

    return fn


def _selected_channels(sub: str) -> dict[str, float]:
    sel = selection()
    if sub not in sel:
        raise RuntimeError("run `python -m chronofuse.bench select` first")
    return sel[sub]["weights"]


STAGES["pool"] = _pool_stage(lambda sub: dict.fromkeys(POOL_A, 1.0))  # BM25 ⊕ dense
STAGES["pool_sel"] = _pool_stage(_selected_channels)  # train-selected channels


# ---------------------------------------------------------- rerankers ----
def _rerank_stage(name: str, pool_stage: str, top_n: int = DEPTH) -> StageFn:
    def fn(ctx, topics, split, sub):
        m = model(name)
        pool = load_stage(ctx.domain, split, sub, pool_stage)
        if pool is None:
            raise RuntimeError(f"{pool_stage} missing for {ctx.domain}/{split}_{sub}")
        # A (topic, document) score does not depend on the pool: reuse the scores computed on
        # the BM25 ⊕ dense pool and only score documents that are new in this pool.
        known = {}
        if pool_stage != "pool":
            prev = load_stage(ctx.domain, split, sub, name) or {}
            known = {q: dict(map(tuple, r)) for q, r in prev.items()}
        out, n_new = {}, 0
        t0 = time.time()
        for t in topics:
            docs = [d for d, _ in pool.get(t.qid, [])[:top_n]]
            have = known.get(t.qid, {})
            todo = [d for d in docs if d not in have]
            scores = dict(have)
            if todo:
                texts = ctx.candidate_texts(t, todo)
                q = rerank_query(t)
                if RERANKERS[name][0] == "ce":
                    s = m.predict([(q, x) for x in texts], batch_size=32, show_progress_bar=False)
                    s = [float(x) for x in s]
                else:
                    s = m.score(q, texts)
                scores.update(zip(todo, s))
                n_new += len(todo)
            out[t.qid] = sorted(([d, round(scores[d], 6)] for d in docs), key=lambda r: -r[1])
        log.info(
            "[%s] %s %d topics, %d new pairs, %.1fs",
            ctx.domain,
            name,
            len(topics),
            n_new,
            time.time() - t0,
        )
        return out

    return fn


for _n in RERANKERS:
    STAGES[_n] = _rerank_stage(_n, "pool")  # on BM25 ⊕ dense
    STAGES[f"{_n}@sel"] = _rerank_stage(_n, "pool_sel")  # on the train-selected pool


def final_rerank(sub: str) -> tuple[str, bool]:
    """Train-selected pointwise reranker (name, fused-with-pool?)."""
    best = selection().get(sub, {}).get("best_rerank", "q3rr4+pool")
    return best.replace("+pool", ""), best.endswith("+pool")


def listwise_base(domain: str, split: str, sub: str) -> dict[str, list[str]]:
    """Listwise input: the selected pool reranked by the train-selected pointwise reranker."""
    name, fused = final_rerank(sub)
    pool = load_stage(domain, split, sub, "pool_sel") or {}
    rr = load_stage(domain, split, sub, f"{name}@sel") or {}
    out = {}
    for q, p in pool.items():
        head = rr.get(q, [])
        if fused:
            ranked = [d for d, _ in rrf({"pool": p, "rr": head}, top=DEPTH)]
        else:
            seen = {d for d, _ in head}
            ranked = [d for d, _ in head] + [d for d, _ in p if d not in seen]
        out[q] = ranked
    return out


@stage("listwise")  # RankGPT sliding window (20/10) over the top-40, batched across topics
def st_listwise(ctx, topics, split, sub, top_n: int = 40, window: int = 20, step: int = 10):
    llm = model("llm")
    base = listwise_base(ctx.domain, split, sub)
    order = {t.qid: base.get(t.qid, [])[:top_n] for t in topics}
    texts = {
        t.qid: dict(zip(order[t.qid], ctx.candidate_texts(t, order[t.qid], 90))) for t in topics
    }
    tmap = {t.qid: t for t in topics}
    end = top_n
    while True:
        start = max(0, end - window)
        convs, keys = [], []
        for qid, docs in order.items():
            win = docs[start:end]
            if len(win) < 2:
                continue
            listing = "\n".join(f"[{i + 1}] {texts[qid][d]}" for i, d in enumerate(win))
            prompt = (
                f"Question:\n{rerank_query(tmap[qid], 150)}\n\nPassages:\n{listing}\n\n"
                "Rank the passages by how useful they are for answering the question. A passage "
                "on the right topic but about the wrong time period or events is less useful. "
                "Answer only with the identifiers in order, e.g. [3] > [1] > [2]."
            )
            convs.append([{"role": "user", "content": prompt}])
            keys.append(qid)
        outs = llm.chat_many(convs, max_tokens=200) if convs else []
        for qid, txt in zip(keys, outs):
            win = order[qid][start:end]
            new, seen = [], set()
            for m in re.findall(r"\[(\d+)\]", txt):
                i = int(m) - 1
                if 0 <= i < len(win) and i not in seen:
                    new.append(win[i])
                    seen.add(i)
            new += [d for i, d in enumerate(win) if i not in seen]
            order[qid][start:end] = new
        if start == 0:
            break
        end -= step
    return {q: [[d, float(len(docs) - r)] for r, d in enumerate(docs)] for q, docs in order.items()}


# ================================================================= CLI ====
def run_stages(stages: list[str], split: str, sub: str, domains: list[str], force: bool) -> None:
    for dom in domains:
        ctx = DomainContext(dom)
        topics = load_topics(dom, split, sub)
        for name in stages:
            if not force and cache_path(dom, split, sub, name).exists():
                continue
            t0 = time.time()
            try:
                res = STAGES[name](ctx, topics, split, sub)
            except Exception:  # noqa: BLE001 - unattended runs: log, skip, keep going
                log.exception("[%s/%s_%s] stage %s FAILED", dom, split, sub, name)
                continue
            save_stage(dom, split, sub, name, res)
            log.info(
                "[%s/%s_%s] %s: %d topics in %.1fs",
                dom,
                split,
                sub,
                name,
                len(res),
                time.time() - t0,
            )
        del ctx
        gc.collect()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(prog="chronofuse.bench")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stage")
    s.add_argument("--stage", nargs="+", required=True)
    s.add_argument("--split", nargs="+", default=["dev"])
    s.add_argument("--subtrack", nargs="+", default=["1a"])
    s.add_argument("--domains", nargs="+", default=["all"])
    s.add_argument("--force", action="store_true")
    sp.add_parser("select")
    r = sp.add_parser("report")
    r.add_argument("--split", default="dev")
    a = ap.parse_args()
    if a.cmd == "stage":
        doms = TRACK1_DOMAINS if a.domains == ["all"] else a.domains
        for split in a.split:
            for sub in a.subtrack:
                run_stages(a.stage, split, sub, doms, a.force)
    elif a.cmd == "select":
        from .systems import select_pool

        print(json.dumps(select_pool(), indent=1))
    else:
        from .systems import report

        report(a.split)


if __name__ == "__main__":
    main()
    # vLLM's engine process can keep the interpreter alive after the work is done; results are
    # already on disk, so exit hard instead of waiting for it.
    from .llm.vllm_client import hard_exit

    hard_exit()
