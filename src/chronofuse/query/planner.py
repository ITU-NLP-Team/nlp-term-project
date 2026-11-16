"""LLM temporal query planner: event chain, multi-query, expansion and query2doc.

One LLM call per query returns a JSON plan:

* ``events``         chronological event chain ("olay örgüsü") the answer depends
                     on, each with an (optional) year — later used as sub-queries
                     and as the query's year window;
* ``keywords``       expansion terms (synonyms, entity / version names, implied dates);
* ``sub_queries``    2-4 self-contained search queries (multi-query retrieval);
* ``answer_passage`` a short hypothetical answer (HyDE / query2doc);
* ``year_window``, ``temporal_intent``.

The single demonstration is *distilled from a train-split guidance record*
(events, retrieval plan, summary, reasoning) — the benchmark's own annotation
schema — and is never taken from the evaluated split. ``feedback`` optionally
adds the top BM25 snippets to the prompt (LLM-based pseudo-relevance feedback,
"corpus-steered" expansion), so the model can lift vocabulary and dates that the
collection actually uses.
"""

from __future__ import annotations

import json
import re

from ..data.reteco import Query

_SYSTEM = """You help a search engine answer time-sensitive questions from a StackExchange \
forum. Read the question and return ONLY a JSON object with these keys:
"temporal_intent": one of when, period_definition, timeline, ongoing_status, duration, \
before_after, order, none
"year_window": [first_year, last_year] the evidence should be about, or null
"events": chronological list of events the answer depends on, each {"year": int or null, \
"event": short phrase}
"keywords": 8-15 search keywords not already in the question (synonyms, names, versions, dates)
"sub_queries": 3 short search queries, each covering one part of the question
"answer_passage": a 60-90 word passage that would answer the question, with concrete names \
and dates"""


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "temporal_intent": {"type": "string"},
        "year_window": {
            "anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "null"}]
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "year": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "event": {"type": "string"},
                },
                "required": ["year", "event"],
            },
        },
        "keywords": {"type": "array", "items": {"type": "string"}},
        "sub_queries": {"type": "array", "items": {"type": "string"}},
        "answer_passage": {"type": "string"},
    },
    "required": [
        "temporal_intent",
        "year_window",
        "events",
        "keywords",
        "sub_queries",
        "answer_passage",
    ],
}


def _demo(q: Query, guidance: dict, evidence: str = "") -> tuple[str, str]:
    """Rewrite a train query's guidance (+ its gold evidence) into the plan schema.

    events / sub-queries / intent come from the benchmark's guidance; the hypothetical answer
    and the keywords come from the query's top gold document (train labels), so the
    demonstration shows a *content* answer rather than a description of the information need.
    """
    from ..metadata.profile import extract_entities

    g = guidance.get("query_guidance", {})
    events = [{"year": None, "event": e} for e in g.get("query_temporal_events", [])][:5]
    anchors = g.get("key_time_anchors", [])
    years = sorted(
        {int(y) for a in anchors for y in re.findall(r"\b(1[0-9]{3}|20[0-3][0-9])\b", str(a))}
    )
    target = {
        "temporal_intent": g.get("temporal_intent", "none"),
        "year_window": [years[0], years[-1]] if years else None,
        "events": events,
        "keywords": [e for e in extract_entities(evidence, 20) if e.lower() not in q.text.lower()][
            :10
        ]
        + [str(a) for a in anchors][:3],
        "sub_queries": [s.get("action", "") for s in g.get("retrieval_plan", [])][:3],
        "answer_passage": " ".join(evidence.split()[:80]) or g.get("query_summary", ""),
    }
    return q.text[:1200], json.dumps(target)


class QueryPlanner:
    def __init__(self, llm, demo: tuple | None = None, max_chars: int = 1500):
        self.llm = llm
        self.demo = _demo(*demo) if demo else None
        self.max_chars = max_chars

    def messages(self, q: Query, feedback: list[str] | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": _SYSTEM}]
        if self.demo:
            msgs += [
                {"role": "user", "content": "Question:\n" + self.demo[0]},
                {"role": "assistant", "content": self.demo[1]},
            ]
        user = "Question:\n" + q.text[: self.max_chars]
        if feedback:
            user += "\n\nTop search results (may be noisy; reuse their terms and dates):\n" + (
                "\n".join(f"- {s}" for s in feedback)
            )
        msgs.append({"role": "user", "content": user})
        return msgs

    def plan(self, q: Query, feedback: list[str] | None = None) -> dict:
        try:
            out = self.llm.chat_json(self.messages(q, feedback), max_tokens=450)
        except Exception:  # noqa: BLE001 - a failed plan degrades to "no expansion"
            out = {}
        return normalise_plan(out)


def parse_plan_text(text: str) -> dict:
    """Parse a plan; if the JSON is truncated or malformed, recover the fields one by one."""
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            out = json.loads(m.group(0))
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
    out: dict = {}
    m = re.search(r'"temporal_intent"\s*:\s*"([^"]*)"', text)
    if m:
        out["temporal_intent"] = m.group(1)
    m = re.search(r'"year_window"\s*:\s*\[\s*(\d{4})\s*,\s*(\d{4})\s*\]', text)
    if m:
        out["year_window"] = [int(m.group(1)), int(m.group(2))]
    out["events"] = [
        {"year": int(y) if y.isdigit() else None, "event": e}
        for y, e in re.findall(r'"year"\s*:\s*(\d{4}|null)\s*,\s*"event"\s*:\s*"([^"]+)"', text)
    ]
    for key in ("keywords", "sub_queries"):
        m = re.search(rf'"{key}"\s*:\s*\[(.*?)(\]|$)', text, re.S)
        if m:
            out[key] = re.findall(r'"([^"]+)"', m.group(1))
    m = re.search(r'"answer_passage"\s*:\s*"(.*?)("\s*\}|$)', text, re.S)
    if m:
        out["answer_passage"] = m.group(1)
    return out


def _strs(x, n: int) -> list[str]:
    if not isinstance(x, list):
        return []
    return [s.strip() for s in x if isinstance(s, str) and s.strip()][:n]


def normalise_plan(out: dict) -> dict:
    yw = out.get("year_window")
    window = None
    if isinstance(yw, list) and len(yw) == 2:
        try:
            a, b = int(yw[0]), int(yw[1])
            if 1000 <= a <= b <= 2035 and b - a <= 60:
                window = [a, b]
        except (TypeError, ValueError):
            window = None
    events = []
    for e in out.get("events", []) if isinstance(out.get("events"), list) else []:
        if isinstance(e, dict) and isinstance(e.get("event"), str):
            y = e.get("year")
            events.append(
                {
                    "year": y if isinstance(y, int) and 1000 <= y <= 2035 else None,
                    "event": e["event"].strip(),
                }
            )
        elif isinstance(e, str):
            events.append({"year": None, "event": e.strip()})
    if window is None:
        ys = sorted(e["year"] for e in events if e["year"])
        if ys and ys[-1] - ys[0] <= 30:
            window = [ys[0], ys[-1]]
    ap = out.get("answer_passage")
    if isinstance(ap, str):
        ap = " ".join(ap.split()[:120])
    return {
        "temporal_intent": out.get("temporal_intent")
        if isinstance(out.get("temporal_intent"), str)
        else "none",
        "year_window": window,
        "events": events[:6],
        "keywords": _strs(out.get("keywords"), 15),
        "sub_queries": _strs(out.get("sub_queries"), 4),
        "answer_passage": ap.strip() if isinstance(ap, str) else "",
    }
