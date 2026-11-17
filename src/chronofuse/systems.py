"""System recipes over cached stages: selection on train, evaluation on dev, significance.

A *system* maps (domain, split, sub-track) to a run {qid: ranking} using only
cached stages (:mod:`chronofuse.bench`). Every choice that is not fixed a
priori — RM3 α, prior weight λ, which first-stage channels enter the rerank
pool, pure vs. fused reranking, the final configuration — is made on the
**train** split; dev is only evaluated. Evaluation follows the official
protocol: nDCG@10 per query (topics missing from a run score 0), mean per
domain, macro mean over the 13 domains.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

from .bench import RERANKERS, load_stage, load_topics
from .config import resolve_path
from .data.reteco import TRACK1_DOMAINS, load_qrels
from .eval.metrics import per_query_ndcg, recall_at
from .metadata.temporal import year_overlap
from .retrieval.fusion import rrf

Run = dict[str, list[tuple[str, float]]]
OFFICIAL = {  # starter_kit/BASELINE_RESULTS.md, data v1.1 (document-level Lucene BM25)
    ("dev", "1a"): 0.1147,
    ("dev", "1b"): 0.1177,
    ("train", "1a"): 0.1075,
    ("train", "1b"): 0.1024,
}
SUBS = ("1a", "1b")


class MissingStage(KeyError):
    pass


def _get(dom, split, sub, name) -> Run:
    raw = load_stage(dom, split, sub, name)
    if raw is None:
        raise MissingStage(f"{dom}/{split}_{sub}/{name}")
    return {q: [(d, s) for d, s in r] for q, r in raw.items()}


def complete(head: Run, base: Run) -> Run:
    """Reranked head followed by the rest of the base ranking."""
    out = {}
    for q, b in base.items():
        h = head.get(q, [])
        seen = {d for d, _ in h}
        out[q] = list(h) + [(d, -1e3 - i) for i, (d, _) in enumerate(b) if d not in seen]
    return out


def fuse(runs: dict[str, Run], weights: dict[str, float] | None = None) -> Run:
    qids = set().union(*[r.keys() for r in runs.values()])
    return {q: rrf({n: r.get(q, []) for n, r in runs.items()}, weights, top=100) for q in qids}


_YEARS: dict = {}


def _profiles(dom) -> dict[str, list[int]]:
    """doc id -> years (memoised for all domains; only the years are kept, to save memory)."""
    if dom not in _YEARS:
        import orjson

        path = resolve_path("data/indexes") / dom / "profiles.jsonl"
        with open(path, "rb") as f:
            _YEARS[dom] = {d["id"]: d["years"] for d in map(orjson.loads, f) if d["years"]}
    return _YEARS[dom]


def temporal_prior(run: Run, windows: dict, profiles, lam: float) -> Run:
    if lam == 0:
        return run
    out = {}
    for q, r in run.items():
        w = windows.get(q)
        if not r or not w:
            out[q] = r
            continue
        ys = list(range(w[0], w[-1] + 1)) if w[-1] - w[0] <= 50 else [w[0], w[-1]]
        vals = [s for _, s in r]
        lo, span = min(vals), (max(vals) - min(vals)) or 1.0
        rs = []
        for d, s in r:
            dy = profiles.get(d)
            t = year_overlap(ys, dy) if dy else 0.0
            rs.append((d, (1 - lam) * (s - lo) / span + lam * t))
        out[q] = sorted(rs, key=lambda x: -x[1])
    return out


# ------------------------------------------------------------- systems ----
SystemFn = Callable[[str, str, str, dict], Run]


def s_stage(name: str) -> SystemFn:
    return lambda dom, split, sub, hp: _get(dom, split, sub, name)


def s_prior(source: str) -> SystemFn:
    def fn(dom, split, sub, hp):
        base = _get(dom, split, sub, "bm25")
        win = {
            t.qid: ([t.years[0], t.years[-1]] if t.years else None)
            for t in load_topics(dom, split, sub)
        }
        if source == "tprf":
            inf = load_stage(dom, split, sub, "tprf") or {}
            win = {q: (w or inf.get(q)) for q, w in win.items()}
        return temporal_prior(base, win, _profiles(dom), hp.get("lam", 0.0))

    return fn


def s_fuse(names: list[str], weights: dict | None = None) -> SystemFn:
    return lambda dom, split, sub, hp: fuse({n: _get(dom, split, sub, n) for n in names}, weights)


def s_rerank(name: str, fused: bool, pool_stage: str = "pool") -> SystemFn:
    """Pointwise reranking of a pool's top-100: pure (sorted by reranker) or RRF with the pool."""
    stage_name = name if pool_stage == "pool" else f"{name}@sel"

    def fn(dom, split, sub, hp):
        pool = _get(dom, split, sub, pool_stage)
        rr = complete(_get(dom, split, sub, stage_name), pool)
        return fuse({"pool": pool, "rr": rr}) if fused else rr

    return fn


def s_late(name: str) -> SystemFn:
    """Selected pool ⊕ (BM25 ⊕ dense top-100 reranked by ``name``), equal-weight RRF.

    Reranking is done once on a fixed candidate set; the train-selected pool contributes the
    recall of its LLM / RAPTOR / dense channels without re-scoring its new documents.
    """

    def fn(dom, split, sub, hp):
        sel = _get(dom, split, sub, "pool_sel")
        rr = complete(_get(dom, split, sub, name), _get(dom, split, sub, "pool"))
        return fuse({"sel": sel, "rr": rr})

    return fn


_PERIODS: dict = {}


def _periods(dom) -> dict[str, tuple[int, int]]:
    """doc id -> LLM-extracted (start, end) period (memoised; documents without one omitted)."""
    if dom not in _PERIODS:
        from .metadata.llm_meta import load_doc_meta

        _PERIODS[dom] = {
            d: (m["sy"], m["ey"]) for d, m in load_doc_meta(dom).items() if m.get("sy")
        }
    return _PERIODS[dom]


def period_score(window: tuple[int, int], period: tuple[int, int] | None) -> float:
    """1 when the document's period overlaps the query's, decaying with the gap in years;
    documents without a period are neutral (0.5)."""
    if period is None:
        return 0.5
    gap = max(0, period[0] - window[1], window[0] - period[1])
    return 1.0 / (1.0 + gap / 2.0)


def s_meta_prior(base: str = "pool") -> SystemFn:
    """A ranking re-scored with the LLM-period match: (1-λ)·minmax(score) + λ·period_score."""

    def fn(dom, split, sub, hp):
        from .bench import query_window

        run = _get(dom, split, sub, base)
        lam = hp.get("lam", 0.0)
        if lam == 0:
            return run
        qm = load_stage(dom, split, sub, "qmeta")
        if qm is None:
            raise MissingStage(f"{dom}/{split}_{sub}/qmeta")
        per = _periods(dom)
        win = {t.qid: query_window(t, qm) for t in load_topics(dom, split, sub)}
        out = {}
        for q, r in run.items():
            w = win.get(q)
            if not r or w is None:
                out[q] = r
                continue
            vals = [s for _, s in r]
            lo, span = min(vals), (max(vals) - min(vals)) or 1.0
            rs = [
                (d, (1 - lam) * (s - lo) / span + lam * period_score(w, per.get(d))) for d, s in r
            ]
            out[q] = sorted(rs, key=lambda x: -x[1])
        return out

    return fn


def s_listwise(dom, split, sub, hp) -> Run:
    from .bench import listwise_base

    base_order = listwise_base(dom, split, sub)
    base = {q: [(d, -i) for i, d in enumerate(v)] for q, v in base_order.items()}
    return complete(_get(dom, split, sub, "listwise"), base)


# first-stage channels considered for the rerank pool (forward selection on train)
CHANNELS = [
    "bm25",
    "dense",
    "rm3_a03",
    "llm_kw",
    "llm_q2d",
    "llm_multi",
    "llm_kw_fb",
    "llm_q2d_fb",
    "llm_multi_fb",
    "dense_doc",
    "raptor",
    "raptor_t",
    "kg",
    "timefilter",
    # small-LLM metadata and neighbouring-event expansion
    "meta_kw",
    "meta_field",
    "meta_tf_bm25",
    "meta_tf_dense",
    "nbr_bm25",
    "nbr_hybrid",
    # ReasonIR-8B, plain and with LLM-reasoning queries
    "rir",
    "rir_reason",
]
META = ["bm25", "dense", "meta_kw", "meta_field", "meta_tf_bm25", "meta_tf_dense"]

# (id, group, label, fn, grid)
SYSTEMS: list[tuple[str, str, str, SystemFn | None, dict]] = [
    ("bm25", "lexical", "BM25 (passage, MaxP)", s_stage("bm25"), {}),
    ("bm25_full", "lexical", "BM25, 1b input = full query + step", s_stage("bm25_full"), {}),
    ("rm3", "lexical", "BM25 + RM3 PRF", None, {"stage": ["rm3_a03", "rm3_a05", "rm3_a07"]}),
    ("bm25_y05", "temporal", "BM25, year tokens x0.5", s_stage("bm25_y05"), {}),
    ("bm25_y0", "temporal", "BM25, year tokens removed", s_stage("bm25_y0"), {}),
    (
        "prior_exp",
        "temporal",
        "BM25 + temporal prior (explicit years)",
        s_prior("explicit"),
        {"lam": [0.0, 0.05, 0.1, 0.2, 0.3]},
    ),
    (
        "prior_tprf",
        "temporal",
        "BM25 + temporal prior (explicit or temporal PRF)",
        s_prior("tprf"),
        {"lam": [0.0, 0.05, 0.1, 0.2, 0.3]},
    ),
    ("timefilter", "temporal", "year-index filtered BM25 (alone)", s_stage("timefilter"), {}),
    (
        "bm25+timefilter",
        "temporal",
        "BM25 ⊕ year-index filtered BM25",
        s_fuse(["bm25", "timefilter"]),
        {},
    ),
    ("kg", "temporal", "temporal KG (personalised PageRank)", s_stage("kg"), {}),
    ("bm25+kg", "temporal", "BM25 ⊕ temporal KG", s_fuse(["bm25", "kg"]), {}),
    ("dense", "dense", "dense (Qwen3-Embedding-0.6B, passages)", s_stage("dense"), {}),
    ("dense_doc", "dense", "dense, document vectors", s_stage("dense_doc"), {}),
    ("raptor", "dense", "RAPTOR tree (collapsed, no time)", s_stage("raptor"), {}),
    ("raptor_t", "dense", "temporal RAPTOR tree", s_stage("raptor_t"), {}),
    ("bm25+dense", "dense", "BM25 ⊕ dense", s_fuse(["bm25", "dense"]), {}),
    ("bm25+raptor_t", "dense", "BM25 ⊕ temporal RAPTOR", s_fuse(["bm25", "raptor_t"]), {}),
    ("llm_kw", "llm", "LLM keyword expansion", s_stage("llm_kw"), {}),
    ("llm_q2d", "llm", "LLM query2doc", s_stage("llm_q2d"), {}),
    ("llm_multi", "llm", "LLM multi-query + event chain", s_stage("llm_multi"), {}),
    ("llm_kw_fb", "llm", "LLM-PRF keyword expansion", s_stage("llm_kw_fb"), {}),
    ("llm_q2d_fb", "llm", "LLM-PRF query2doc", s_stage("llm_q2d_fb"), {}),
    ("llm_multi_fb", "llm", "LLM-PRF multi-query + event chain", s_stage("llm_multi_fb"), {}),
    ("meta_kw", "meta", "BM25 + small-LLM events/entities/keywords", s_stage("meta_kw"), {}),
    ("meta_field", "meta", "BM25 over LLM document metadata", s_stage("meta_field"), {}),
    ("meta_tf_bm25", "meta", "BM25, LLM-period filter", s_stage("meta_tf_bm25"), {}),
    ("meta_tf_dense", "meta", "dense, LLM-period filter", s_stage("meta_tf_dense"), {}),
    (
        "meta_prior",
        "meta",
        "BM25 ⊕ dense + LLM-period prior",
        s_meta_prior("pool"),
        {"lam": [0.0, 0.05, 0.1, 0.2, 0.3]},
    ),
    ("meta_hybrid", "meta", "BM25 ⊕ dense ⊕ metadata channels (filtered)", s_fuse(META), {}),
    ("nbr_bm25", "meta", "neighbouring events + sub-queries (BM25)", s_stage("nbr_bm25"), {}),
    (
        "nbr_hybrid",
        "meta",
        "neighbouring events + sub-queries (BM25 ⊕ dense)",
        s_stage("nbr_hybrid"),
        {},
    ),
    ("rir", "dense", "ReasonIR-8B (passages, full query)", s_stage("rir"), {}),
    ("rir_reason", "dense", "ReasonIR-8B + LLM reasoning query", s_stage("rir_reason"), {}),
    (
        "rir+rir_reason",
        "dense",
        "ReasonIR ⊕ ReasonIR + reasoning",
        s_fuse(["rir", "rir_reason"]),
        {},
    ),
    ("pool_sel", "pool", "train-selected hybrid pool (RRF)", s_stage("pool_sel"), {}),
]
RR_LABEL = {
    "ce_minilm": "MiniLM-L12 cross-encoder",
    "ce_bge": "BGE-reranker-v2-m3",
    "q3rr06": "Qwen3-Reranker-0.6B",
    "q3rr4": "Qwen3-Reranker-4B",
}
for _r in RERANKERS:
    SYSTEMS.append((f"{_r}", "rerank", f"BM25 ⊕ dense → {RR_LABEL[_r]}", s_rerank(_r, False), {}))
    SYSTEMS.append(
        (f"{_r}+pool", "rerank", f"BM25 ⊕ dense ⊕ {RR_LABEL[_r]}", s_rerank(_r, True), {})
    )


def evaluate(fn: SystemFn, split: str, sub: str, hp: dict) -> dict:
    root = resolve_path("data/raw/reteco")
    per_dom, per_q, rec = {}, {}, {}
    try:
        for d in TRACK1_DOMAINS:
            qrels = load_qrels(root, d, split, steps=(sub == "1b"))
            run = fn(d, split, sub, hp)
            s = per_query_ndcg(run, qrels)
            per_dom[d] = float(np.mean(list(s.values()))) if s else 0.0
            per_q.update({f"{d}/{q}": v for q, v in s.items()})
            rec[d] = recall_at(run, qrels, 100)
    except MissingStage:
        return {}
    return {
        "macro": round(float(np.mean(list(per_dom.values()))), 4),
        "per_domain": {k: round(v, 4) for k, v in per_dom.items()},
        "recall100": round(float(np.mean(list(rec.values()))), 4),
        "per_query": per_q,
    }


def randomization_test(a: dict, b: dict, n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired randomisation test on per-query nDCG@10 (pooled over domains)."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return 1.0
    diff = np.array([a[k] - b[k] for k in keys])
    obs = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(diff)))
    return float((np.abs((signs * diff).mean(1)) >= obs - 1e-12).mean())


def _tuned(sid, fn, grid, split, sub, tune_split="train"):
    """Evaluate a system, selecting its grid value on train first."""
    if not grid:
        return evaluate(fn, split, sub, {}), {}
    ((name, values),) = grid.items()
    scores = {}
    for v in values:
        f = s_stage(v) if name == "stage" else fn
        r = evaluate(f, tune_split, sub, {} if name == "stage" else {name: v})
        if r:
            scores[v] = r["macro"]
    if not scores:
        return {}, {}
    best = max(scores, key=scores.get)
    f = s_stage(best) if name == "stage" else fn
    r = evaluate(f, split, sub, {} if name == "stage" else {name: best})
    return r, {name: best, "train_grid": scores}


# ----------------------------------------------------------- selection ----
def select_pool(split: str = "train") -> dict:
    """Train-split selection of (i) the first-stage channels and (ii) the pointwise reranker.

    (i) Greedy forward selection over (channel, RRF weight in {1, 0.5}), starting from the
        best single channel; objective macro nDCG@10, ties broken by recall@100.
    (ii) Among the rerankers computed on the fixed BM25 ⊕ dense pool, pure or fused with the
        pool, the one with the best macro nDCG@10 on **train 1a**; it is used for both
        sub-tracks (the 4B reranker is too costly to also run on the 2.7k train steps).
    """
    out: dict = {}
    rr_scores = {}
    for r in RERANKERS:
        for fused in (False, True):
            e = evaluate(s_rerank(r, fused), split, "1a", {})
            if e:
                rr_scores[f"{r}{'+pool' if fused else ''}"] = e["macro"]
    path = resolve_path("results/metrics") / "pool_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    for sub in SUBS:
        single = {}
        for c in CHANNELS:
            e = evaluate(s_stage(c), split, sub, {})
            if e:
                single[c] = (e["macro"], e["recall100"])
        first = max(single, key=single.get)
        weights, best_key, trace = {first: 1.0}, single[first], []
        while True:
            cand = None
            for c in single:
                if c in weights:
                    continue
                for w in (1.0, 0.5):
                    trial = {**weights, c: w}
                    r = evaluate(s_fuse(list(trial), trial), split, sub, {})
                    key = (r["macro"], r["recall100"])
                    trace.append(
                        {"weights": trial, "macro": r["macro"], "recall100": r["recall100"]}
                    )
                    if key > best_key and (cand is None or key > cand[1]):
                        cand = (trial, key)
            if cand is None:
                break
            weights, best_key = cand
        res = {
            "single_train": {c: v[0] for c, v in single.items()},
            "channels": list(weights),
            "weights": weights,
            "train_macro": best_key[0],
            "train_recall100": best_key[1],
            "trace": trace,
        }
        if rr_scores:
            res["rerank_train_1a"] = rr_scores
            res["best_rerank"] = max(rr_scores, key=rr_scores.get)
        out[sub] = res
        print(sub, {k: v for k, v in res.items() if k != "trace"}, flush=True)
    path.write_text(json.dumps(out, indent=1))
    return {s: {k: v for k, v in r.items() if k != "trace"} for s, r in out.items()}


# -------------------------------------------------------------- report ----
def report(split: str = "dev") -> dict:
    sel_path = resolve_path("results/metrics") / "pool_selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    out: dict = {}
    perq: dict = {}
    for sub in SUBS:
        res: dict = {"official_bm25": OFFICIAL.get((split, sub))}
        perq[sub] = {}
        base_q = dense_q = None
        systems = list(SYSTEMS)
        systems += _final_systems(sel.get(sub, {}), sub)
        for sid, group, label, fn, grid in systems:
            if sid == "bm25_full" and sub != "1b":
                continue
            r, chosen = _tuned(sid, fn, grid, split, sub)
            if not r:
                continue
            if sid == "bm25":
                base_q = r["per_query"]
            if sid == "dense":
                dense_q = r["per_query"]
            p = randomization_test(r["per_query"], base_q) if base_q and sid != "bm25" else None
            perq[sub][sid] = r["per_query"]
            if sid == "chronofuse":
                chosen = {**chosen, "reranker": FINAL_CHOICE.get(sub)}
            if group in ("final", "pool") and dense_q:
                chosen = {**chosen, "p_vs_dense": randomization_test(r["per_query"], dense_q)}
            res[sid] = {
                "label": label,
                "group": group,
                "macro": r["macro"],
                "recall100": r["recall100"],
                "per_domain": r["per_domain"],
                "p_vs_bm25": p,
                **chosen,
            }
            print(
                f"{sub} {sid:22s} {r['macro']:.4f}  R@100 {r['recall100']:.4f}  p={p}", flush=True
            )
        out[sub] = res
    path = resolve_path("results/metrics") / f"bench_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    (path.parent / f"bench_{split}_perquery.json").write_text(json.dumps(perq))
    return out


FINAL_CHOICE: dict[str, str] = {}


def _final_systems(sel: dict, sub: str) -> list:
    """Final systems: the selected pool fused with each reranker's ranking of the BM25 ⊕ dense
    top-100, and ChronoFuse = the variant with the best macro nDCG@10 on train for this
    sub-track (rerankers not run on that train split are not candidates)."""
    out, train, fns = [], {}, {}
    for r in RERANKERS:
        for sid, label, fn in (
            (f"late_{r}", f"selected pool ⊕ {RR_LABEL[r]}", s_late(r)),
            # the selected pool's own top-100 reranked (needs the ``<r>@sel`` stage)
            (f"sel_{r}", f"selected pool → {RR_LABEL[r]}, ⊕ pool", s_rerank(r, True, "pool_sel")),
        ):
            out.append((sid, "final", label, fn, {}))
            e = evaluate(fn, "train", sub, {})
            if e:
                train[sid], fns[sid] = e["macro"], fn
    if train:
        best = max(train, key=train.get)
        FINAL_CHOICE[sub] = best
        out.append(("chronofuse", "final", "ChronoFuse (final, train-selected)", fns[best], {}))
    return out
