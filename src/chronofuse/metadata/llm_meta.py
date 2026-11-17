"""LLM-extracted temporal and topical metadata, and neighbouring-event query expansion.

Rule-based date extraction (:mod:`chronofuse.metadata.temporal`) only sees explicit years.
Here a *small* LLM (default Qwen3-1.7B) reads every document's opening and every query and
returns the same compact record for both::

    {"sy": start year, "ey": end year,        # period the text is ABOUT (implicit ones too)
     "ev": [events], "en": [entities], "kw": [keywords], "tp": "subtopic"}

Documents and queries share the schema, so their periods can be intersected (time filter /
prior) and their event / entity / keyword fields matched with BM25 (a metadata field index).

A larger LLM (the query planner's, default Qwen3-8B) additionally expands each query with
*neighbouring events* (causes, preceding / following / parallel events, each with a year),
period-anchored sub-queries and expansion terms, seeing the top BM25 snippets
(LLM pseudo-relevance feedback).

Corpus-scale extraction is sharded (10k documents per shard, one JSONL per shard under
``data/indexes/<domain>/llm_meta/``), resumable, and split over GPUs by shard number::

    python -m chronofuse.metadata.llm_meta docs --worker 0 --workers 4
    python -m chronofuse.metadata.llm_meta status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

import orjson

from ..config import resolve_path

log = logging.getLogger("chronofuse.llm_meta")

SHARD = 10_000
DOC_WORDS = 220
YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-3][0-9])$")

META_SYSTEM = """You extract search metadata from a text taken from a StackExchange forum \
or the documents it cites. Return ONLY a JSON object with these keys:
"start_year", "end_year": the period the text is ABOUT, as years. Infer implicit periods \
too (e.g. "after the 2008 crisis" -> 2008-2010, "during the Weimar Republic" -> 1919-1933). \
Use null for both if no period can be inferred.
"events": at most 5 short phrases naming the events, changes or releases the text discusses
"entities": at most 6 named people, organisations, products, laws, coins or places
"keywords": at most 6 topical search terms that are not entity names
"topic": a 2-5 word subtopic label"""

_YEAR = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
_STRS = {"type": "array", "items": {"type": "string"}}
META_SCHEMA = {
    "type": "object",
    "properties": {
        "start_year": _YEAR,
        "end_year": _YEAR,
        "events": _STRS,
        "entities": _STRS,
        "keywords": _STRS,
        "topic": {"type": "string"},
    },
    "required": ["start_year", "end_year", "events", "entities", "keywords", "topic"],
}

EXPAND_SYSTEM = """You help a search engine find evidence for a time-sensitive question. \
Relevant documents often describe related events and mechanisms rather than repeating the \
question's words or dates. Return ONLY a JSON object with these keys:
"period": [first_year, last_year] the evidence is about, or null
"neighbor_events": 4-6 events closely connected to the question (causes, preceding and \
following events, parallel developments), each {"event": short phrase with names, \
"year": int or null, "relation": one of cause, before, same_time, after, effect}
"sub_queries": 4 short self-contained search queries; each targets one event or aspect and \
names its period
"expansion_terms": 10 terms likely to appear in relevant documents (names, technical terms, \
versions, synonyms) that are not already in the question"""

EXPAND_SCHEMA = {
    "type": "object",
    "properties": {
        "period": {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}]},
        "neighbor_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "year": _YEAR,
                    "relation": {"type": "string"},
                },
                "required": ["event", "year", "relation"],
            },
        },
        "sub_queries": _STRS,
        "expansion_terms": _STRS,
    },
    "required": ["period", "neighbor_events", "sub_queries", "expansion_terms"],
}


# ------------------------------------------------------------- parsing ----
def _year(x) -> int | None:
    try:
        y = int(x)
    except (TypeError, ValueError):
        return None
    return y if 1000 <= y <= 2035 else None


def _strs(x, n: int, max_len: int = 80) -> list[str]:
    if not isinstance(x, list):
        return []
    out, seen = [], set()
    for s in x:
        if isinstance(s, str) and (s := " ".join(s.split())[:max_len]) and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:n]


def _loads(text: str) -> dict:
    """JSON object from (possibly truncated) model output; truncated arrays are recovered."""
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        pass
    out: dict = {}
    for k in ("start_year", "end_year"):
        m = re.search(rf'"{k}"\s*:\s*(\d{{4}})', text)
        if m:
            out[k] = int(m.group(1))
    for k in ("events", "entities", "keywords", "sub_queries", "expansion_terms"):
        m = re.search(rf'"{k}"\s*:\s*\[(.*?)(\]|$)', text, re.S)
        if m:
            out[k] = re.findall(r'"([^"]+)"', m.group(1))
    m = re.search(r'"topic"\s*:\s*"([^"]*)"', text)
    if m:
        out["topic"] = m.group(1)
    m = re.search(r'"period"\s*:\s*\[\s*(\d{4})\s*,\s*(\d{4})\s*\]', text)
    if m:
        out["period"] = [int(m.group(1)), int(m.group(2))]
    out["neighbor_events"] = [
        {"event": e, "year": int(y) if y.isdigit() else None, "relation": ""}
        for e, y in re.findall(r'"event"\s*:\s*"([^"]+)"\s*,\s*"year"\s*:\s*(\d{4}|null)', text)
    ]
    return out


def normalise_meta(raw: dict | str) -> dict:
    d = _loads(raw) if isinstance(raw, str) else (raw or {})
    sy, ey = _year(d.get("start_year")), _year(d.get("end_year"))
    if sy is None:
        sy = ey
    if ey is None:
        ey = sy
    if sy is not None and ey < sy:
        sy, ey = ey, sy
    if sy is not None and ey - sy > 200:  # "all of history" carries no information
        sy = ey = None
    tp = d.get("topic")
    return {
        "sy": sy,
        "ey": ey,
        "ev": _strs(d.get("events"), 5),
        "en": _strs(d.get("entities"), 6),
        "kw": _strs(d.get("keywords"), 6),
        "tp": " ".join(tp.split())[:60] if isinstance(tp, str) else "",
    }


def normalise_expansion(raw: dict | str) -> dict:
    d = _loads(raw) if isinstance(raw, str) else (raw or {})
    period = None
    p = d.get("period")
    if isinstance(p, list) and len(p) == 2 and _year(p[0]) and _year(p[1]):
        a, b = sorted((_year(p[0]), _year(p[1])))
        period = [a, b] if b - a <= 60 else None
    nb = []
    for e in d.get("neighbor_events") or []:
        if isinstance(e, dict) and isinstance(e.get("event"), str) and e["event"].strip():
            rel = e.get("relation")
            nb.append(
                {
                    "event": " ".join(e["event"].split())[:120],
                    "year": _year(e.get("year")),
                    "relation": rel if isinstance(rel, str) else "",
                }
            )
    return {
        "period": period,
        "neighbors": nb[:6],
        "sub_queries": _strs(d.get("sub_queries"), 5, 200),
        "terms": _strs(d.get("expansion_terms"), 12),
    }


def meta_terms(m: dict | None) -> str:
    """Bag of searchable metadata words (events, entities, keywords, topic)."""
    if not m:
        return ""
    return " ".join(m.get("ev", []) + m.get("en", []) + m.get("kw", []) + [m.get("tp", "")])


# ------------------------------------------------------------- prompts ----
MAX_CHARS = 1600  # hashes / code / URLs can make 220 "words" thousands of tokens


def meta_messages(text: str, max_words: int = DOC_WORDS) -> list[dict]:
    body = " ".join(w[:40] for w in text.split()[:max_words])[:MAX_CHARS]
    return [
        {"role": "system", "content": META_SYSTEM},
        {"role": "user", "content": "Text:\n" + body},
    ]


def expand_messages(question: str, feedback: list[str] | None = None) -> list[dict]:
    user = "Question:\n" + question[:1500]
    if feedback:
        user += "\n\nTop search results (may be noisy; reuse their terms and dates):\n" + (
            "\n".join(f"- {s}" for s in feedback)
        )
    return [{"role": "system", "content": EXPAND_SYSTEM}, {"role": "user", "content": user}]


def generate(llm, convs: list[list[dict]], schema: dict, max_tokens: int, cache: bool) -> list:
    """Batched JSON generation with vLLM (guided) or, on CPU, llama.cpp one by one."""
    if hasattr(llm, "chat_many"):
        try:
            return llm.chat_many(convs, max_tokens=max_tokens, schema=schema, cache=cache)
        except Exception:  # noqa: BLE001 - guided decoding unavailable: free decoding + parser
            log.exception("guided decoding failed; falling back to free decoding")
            return llm.chat_many(convs, max_tokens=max_tokens, cache=cache)
    return [llm.chat(c, json_mode=True, max_tokens=max_tokens) for c in convs]


def small_llm():
    """The metadata extractor: a small model, run with a large batch on GPU."""
    try:
        import torch

        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    if gpu:
        from ..llm.vllm_client import VLLMChat

        return VLLMChat(
            model=os.getenv("CF_SMALL_LLM", "Qwen/Qwen3-1.7B"),
            max_model_len=4096,
            gpu_memory_utilization=float(os.getenv("CF_SMALL_GPU_MEM", "0.85")),
            max_num_seqs=512,
        )
    from ..llm.local import LocalLLM

    return LocalLLM()


# ------------------------------------------------------------- corpus ----
def meta_dir(domain: str) -> Path:
    return resolve_path("data/indexes") / domain / "llm_meta"


def _shards(domain: str, mine) -> Iterator[tuple[int, list[tuple[str, str]]]]:
    """(shard number, [(doc id, opening words)]) for the shards ``mine(shard)`` accepts."""
    from ..data.reteco import load_corpus

    buf: list[tuple[str, str]] = []
    n = 0
    for doc in load_corpus(resolve_path("data/raw/reteco"), domain):
        s = n // SHARD
        if mine(s):
            buf.append((doc.id, " ".join(doc.content.split()[:DOC_WORDS])))
        n += 1
        if n % SHARD == 0 and buf:
            yield s, buf
            buf = []
    if buf:
        yield (n - 1) // SHARD, buf


def extract_corpus(
    domains: list[str], worker: int = 0, workers: int = 1, limit: int | None = None
) -> None:
    """Extract metadata for every document; shard ``g`` (numbered over all domains, in order)
    is done by worker ``g % workers``. Finished shards are skipped, so jobs can be restarted
    with any number of workers."""
    llm = None
    counts = {d: _n_docs(d) for d in domains}
    base = 0
    for dom in domains:
        n_sh = -(-counts[dom] // SHARD)
        todo = {
            s
            for s in range(n_sh)
            if (base + s) % workers == worker
            and not (meta_dir(dom) / f"part-{s:05d}.jsonl").exists()
        }
        base += n_sh
        if not todo:
            continue
        for s, buf in _shards(dom, todo.__contains__):
            if limit:
                buf = buf[:limit]
            if llm is None:
                llm = small_llm()
            t0 = time.time()
            outs = generate(
                llm, [meta_messages(t) for _, t in buf], META_SCHEMA, 160, cache=bool(limit)
            )
            name = "smoke.jsonl" if limit else f"part-{s:05d}.jsonl"  # a smoke test never
            path = meta_dir(dom) / name  # marks a shard as done
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for (d, _), o in zip(buf, outs):
                    f.write(orjson.dumps({"id": d, **normalise_meta(o)}) + b"\n")
            os.replace(tmp, path)
            dt = time.time() - t0
            log.info(
                "[%s] shard %d: %d docs in %.0fs (%.1f/s)", dom, s, len(buf), dt, len(buf) / dt
            )
            if limit:
                return


def _n_docs(domain: str) -> int:
    """Number of documents of a domain (one profile line per document)."""
    p = resolve_path("data/indexes") / domain / "profiles.jsonl"
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def load_doc_meta(domain: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(meta_dir(domain).glob("part-*.jsonl")):
        with open(p, "rb") as f:
            for line in f:
                r = orjson.loads(line)
                out[r.pop("id")] = r
    return out


def status(domains: list[str]) -> dict:
    res = {}
    for d in domains:
        n = _n_docs(d)
        done = sum(1 for _ in meta_dir(d).glob("part-*.jsonl"))
        res[d] = f"{done}/{-(-n // SHARD)} shards ({n} docs)"
    return res


def main() -> None:
    from ..data.reteco import TRACK1_DOMAINS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(prog="chronofuse.metadata.llm_meta")
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("docs")
    d.add_argument("--worker", type=int, default=0)
    d.add_argument("--workers", type=int, default=1)
    d.add_argument("--domains", nargs="+", default=["all"])
    d.add_argument("--limit", type=int, default=None, help="smoke test: N docs of one shard")
    s = sp.add_parser("status")
    s.add_argument("--domains", nargs="+", default=["all"])
    a = ap.parse_args()
    doms = TRACK1_DOMAINS if a.domains == ["all"] else a.domains
    if a.cmd == "docs":
        extract_corpus(doms, a.worker, a.workers, a.limit)
    else:
        print(json.dumps(status(doms), indent=1))


if __name__ == "__main__":
    main()
    from ..llm.vllm_client import hard_exit

    hard_exit()
