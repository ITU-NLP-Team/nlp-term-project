"""Turn benchmark JSONs into the paper's LaTeX tables and figures.

    python scripts/make_tables.py

Reads  results/metrics/bench_dev.json, bench_dev_perquery.json, bench_train.json (optional),
       data_analysis_train.json, pool_selection.json, results/cache/*/dev_1a/plan*.json
Writes paper/tables/*.tex, paper/figures/*.pdf, results/metrics/analysis_dev.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from chronofuse.bench import load_stage  # noqa: E402
from chronofuse.config import resolve_path  # noqa: E402
from chronofuse.data.reteco import TRACK1_DOMAINS, load_guidance  # noqa: E402

M = resolve_path("results/metrics")
T = resolve_path("paper/tables")
F = resolve_path("paper/figures")
OFFICIAL_DEV = {  # starter_kit/BASELINE_RESULTS.md (v1.1), 1a dev / 1b dev
    "bitcoin": (0.0370, 0.0299),
    "cardano": (0.1027, 0.0627),
    "economics": (0.0467, 0.0463),
    "genealogy": (0.1949, 0.2183),
    "history": (0.0873, 0.1015),
    "hsm": (0.2489, 0.2140),
    "iota": (0.3309, 0.4195),
    "law": (0.0574, 0.0549),
    "monero": (0.0320, 0.0113),
    "politics": (0.2625, 0.2371),
    "quant": (0.0237, 0.0403),
    "travel": (0.0442, 0.0645),
    "workplace": (0.0230, 0.0297),
}
GROUP_TITLES = {
    "lexical": "Lexical and pseudo-relevance feedback",
    "temporal": "Temporal metadata (prior, year index, KG)",
    "dense": "Dense retrieval and RAPTOR trees",
    "llm": "LLM query planning (Qwen3-8B)",
    "pool": "Train-selected hybrid pool",
    "rerank": r"Pointwise reranking of the BM25 $\oplus$ dense top-100",
    "final": r"Selected pool $\oplus$ reranked BM25 $\oplus$ dense top-100 (final fusion)",
}


def esc(s: str) -> str:
    return (
        s.replace("⊕", r"$\oplus$")
        .replace("→", r"$\rightarrow$")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("x0.5", r"$\times$0.5")
    )


def fmt(v, p=None, best=False):
    if v is None:
        return "--"
    s = f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"
    if p is not None and p < 0.05:
        s += r"$^{\dagger}$"
    return rf"\textbf{{{s}}}" if best else s


def full_table(dev: dict) -> str:
    a, b = dev["1a"], dev["1b"]
    ids = [k for k in a if isinstance(a[k], dict) and "group" in a[k]]
    ids += [k for k in b if k not in ids and isinstance(b[k], dict) and "group" in b[k]]
    best_a = max(a[k]["macro"] for k in ids if k in a)
    best_b = max(b[k]["macro"] for k in ids if k in b)
    rows = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{1a (whole query)}} & "
        r"\multicolumn{2}{c}{\textbf{1b (step-wise)}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"& \textbf{System} & nDCG@10 & R@100 & nDCG@10 & R@100 \\",
        r"\midrule",
        rf"& BM25, document level (official baseline) & {fmt(a['official_bm25'])} & -- & "
        rf"{fmt(b['official_bm25'])} & -- \\",
    ]
    group = None
    for k in ids:
        ra, rb = a.get(k), b.get(k)
        r0 = ra or rb
        if r0["group"] != group:
            group = r0["group"]
            rows.append(r"\midrule")
            rows.append(
                rf"\multicolumn{{6}}{{l}}{{\textit{{{GROUP_TITLES.get(group, group)}}}}} \\"
            )
        extra = ""
        if ra and "lam" in ra:
            extra = rf" ($\lambda{{=}}{ra['lam']}$)"
        if ra and "stage" in ra:
            extra = rf" ($\alpha{{=}}0.{ra['stage'][-1]}$)"
        cells = []
        for r, best in ((ra, best_a), (rb, best_b)):
            if r is None:
                cells += ["--", "--"]
            else:
                cells += [
                    fmt(r["macro"], r.get("p_vs_bm25"), r["macro"] == best),
                    fmt(r["recall100"]),
                ]
        rows.append(rf"& {esc(r0['label'])}{extra} & " + " & ".join(cells) + r" \\")
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Macro nDCG@10 and recall@100 on the RETECO Track~1 \textbf{dev} split "
        r"(13 domains, v1.1). Hyper-parameters, the pool composition and the final "
        r"configuration were selected on \textbf{train}. $^{\dagger}$: different from our "
        r"passage BM25 at $p<0.05$ (paired randomisation test over queries/steps). "
        r"$\oplus$: reciprocal rank fusion.}",
        r"\label{tab:full}",
        r"\end{table*}",
    ]
    return "\n".join(rows)


def main_table(dev: dict, train: dict | None) -> str:
    keys = [
        ("bm25", "BM25 passages + MaxP"),
        ("dense", "Dense (Qwen3-Emb.-0.6B)"),
        ("pool_sel", "Hybrid pool (train-selected)"),
        ("chronofuse", r"\sys (final)"),
    ]
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{1a}} & \multicolumn{2}{c}{\textbf{1b}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{System} & train & dev & train & dev \\",
        r"\midrule",
        r"BM25 (official) & .108 & .115 & .102 & .118 \\",
    ]
    for k, lab in keys:
        cells = []
        for sub in ("1a", "1b"):
            for src in (train, dev):
                r = (src or {}).get(sub, {}).get(k)
                cells.append(fmt(r["macro"]) if r else "--")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Macro nDCG@10 on RETECO Track~1 (v1.1).}",
        r"\label{tab:main}",
        r"\end{table}",
    ]
    return "\n".join(rows)


RR_PRETTY = {
    "ce_minilm": "MiniLM-L12",
    "ce_bge": "BGE-reranker-v2-m3",
    "q3rr06": "Qwen3-Reranker-0.6B",
    "q3rr4": "Qwen3-Reranker-4B",
}
RR_NAMES = [
    ("ce_minilm", "MiniLM-L12 (33M)", "enc."),
    ("ce_bge", "BGE-v2-m3 (568M)", "enc."),
    ("q3rr06", "Qwen3-Rerank. 0.6B", "dec."),
    ("q3rr4", "Qwen3-Rerank. 4B", "dec."),
]


def rerank_table(dev: dict, train: dict | None) -> str:
    """Cross-encoders vs cross-decoders on the same BM25 ⊕ dense top-100."""
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{1a dev}} & \multicolumn{2}{c}{\textbf{1b dev}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"\textbf{Reranker} & & pure & $\oplus$pool & pure & $\oplus$pool \\",
        r"\midrule",
    ]
    a, b = dev["1a"], dev["1b"]
    pool_a = (a.get("bm25+dense") or {}).get("macro")
    pool_b = (b.get("bm25+dense") or {}).get("macro")
    rows.append(
        rf"none (BM25 $\oplus$ dense) & & \multicolumn{{2}}{{c}}{{{fmt(pool_a)}}} & "
        rf"\multicolumn{{2}}{{c}}{{{fmt(pool_b)}}} \\"
    )
    for k, lab, kind in RR_NAMES:
        cells = []
        for r in (a, b):
            for key in (k, f"{k}+pool"):
                v = r.get(key)
                cells.append(fmt(v["macro"], v.get("p_vs_bm25")) if v else "--")
        rows.append(f"{lab} & {kind} & " + " & ".join(cells) + r" \\")
    rows += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Pointwise reranking of the same top-100 (BM25 $\oplus$ dense, RRF): "
        r"encoder cross-encoders vs.\ decoder LLM rerankers (``cross-decoders''). "
        r"\emph{pure}: sorted by the reranker; $\oplus$\emph{pool}: RRF of reranker and "
        r"pool ranks. Macro nDCG@10, dev.}",
        r"\label{tab:rerank}",
        r"\end{table}",
    ]
    return "\n".join(rows)


def per_domain_fig(dev: dict) -> None:
    a = dev["1a"]
    final = a.get("chronofuse") or a.get("pool_sel") or a.get("dense")
    x = np.arange(len(TRACK1_DOMAINS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.bar(
        x - w,
        [OFFICIAL_DEV[d][0] for d in TRACK1_DOMAINS],
        w,
        label="BM25 (official)",
        color="#bbbbbb",
    )
    ax.bar(
        x,
        [a["bm25"]["per_domain"][d] for d in TRACK1_DOMAINS],
        w,
        label="BM25 passages+MaxP",
        color="#6699cc",
    )
    ax.bar(
        x + w,
        [final["per_domain"][d] for d in TRACK1_DOMAINS],
        w,
        label="ChronoFuse",
        color="#cc5533",
    )
    ax.set_xticks(x, TRACK1_DOMAINS, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("nDCG@10 (1a dev)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "per_domain.pdf")
    plt.close(fig)


def intent_analysis(perq: dict) -> dict:
    """nDCG@10 by gold temporal intent (dev guidance used for analysis only)."""
    root = resolve_path("data/raw/reteco")
    intent = {}
    for d in TRACK1_DOMAINS:
        for qid, g in load_guidance(root, d, "dev").items():
            intent[f"{d}/{qid}"] = g.get("query_guidance", {}).get("temporal_intent", "none")
    out: dict = {}
    for sid in ("bm25", "dense", "pool_sel", "chronofuse"):
        scores = perq["1a"].get(sid)
        if not scores:
            continue
        by = defaultdict(list)
        for k, v in scores.items():
            by[intent.get(k, "none")].append(v)
        out[sid] = {i: (round(float(np.mean(v)), 4), len(v)) for i, v in by.items()}
    return out


def intent_fig(ia: dict) -> None:
    if "bm25" not in ia:
        return
    short = {
        "event_verification_and_authenticity": "event verification",
        "period_definition": "period definition",
        "ongoing_status": "ongoing status",
    }
    top = [i for i, (_, n) in sorted(ia["bm25"].items(), key=lambda kv: -kv[1][1]) if n >= 15]
    systems = [
        (s, lab, c)
        for s, lab, c in (
            ("bm25", "BM25", "#6699cc"),
            ("dense", "Dense", "#88aa55"),
            ("chronofuse", "ChronoFuse", "#cc5533"),
        )
        if s in ia
    ]
    y = np.arange(len(top))
    h = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for j, (s, lab, c) in enumerate(systems):
        ax.barh(
            y + (j - (len(systems) - 1) / 2) * h, [ia[s][i][0] for i in top], h, label=lab, color=c
        )
    ax.set_yticks(
        y, [f"{short.get(i, i).replace('_', ' ')} ({ia['bm25'][i][1]})" for i in top], fontsize=6.5
    )
    ax.invert_yaxis()
    ax.set_xlabel("nDCG@10 (1a dev)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "intent.pdf")
    plt.close(fig)


def planner_accuracy() -> dict:
    """Agreement of the LLM plan with the gold dev guidance (analysis only)."""
    root = resolve_path("data/raw/reteco")
    agree = n = win_hit = win_n = 0
    n_events = []
    for d in TRACK1_DOMAINS:
        plans = load_stage(d, "dev", "1a", "plan") or {}
        guid = load_guidance(root, d, "dev")
        for qid, p in plans.items():
            g = guid.get(qid, {}).get("query_guidance", {})
            if not g:
                continue
            n += 1
            agree += p.get("temporal_intent") == g.get("temporal_intent")
            n_events.append(len(p.get("events", [])))
            import re

            years = {
                int(y)
                for a in g.get("key_time_anchors", [])
                for y in re.findall(r"\b(1[0-9]{3}|20[0-3][0-9])\b", str(a))
            }
            if years:
                win_n += 1
                w = p.get("year_window")
                win_hit += bool(w and any(w[0] - 1 <= y <= w[1] + 1 for y in years))
    return {
        "queries": n,
        "intent_acc": round(agree / max(1, n), 3),
        "window_hit_when_gold_has_year": round(win_hit / max(1, win_n), 3),
        "gold_with_year": win_n,
        "avg_events": round(float(np.mean(n_events or [0])), 2),
    }


def data_table() -> str:
    a = json.loads((M / "data_analysis_train.json").read_text())["summary"]
    pct = lambda k: f"{100 * a[k]:.1f}\\%"  # noqa: E731
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        rf"Train queries & {a['queries']} \\",
        rf"\quad with an explicit year (rule extractor) & {pct('pct_queries_explicit_year')} \\",
        rf"\quad with a relative time expression & {pct('pct_queries_relative_expr')} \\",
        rf"\quad marked temporal in the guidance & {pct('pct_queries_guidance_temporal')} \\",
        rf"Gold documents per query & {a['gold_per_query']:.2f} \\",
        r"\midrule",
        rf"Corpus documents with $\geq$1 year & {pct('pct_docs_with_year')} \\",
        rf"Gold documents with $\geq$1 year & {pct('pct_gold_with_year')} \\",
        rf"BM25 top-10 non-gold with $\geq$1 year & {pct('pct_bm25top10_nongold_with_year')} \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Queries with an explicit year:}} \\",
        rf"\quad gold docs overlapping a query year & {pct('pct_gold_overlap_query_years')} \\",
        rf"\quad BM25 top-10 non-gold overlapping & "
        rf"{pct('pct_bm25top10_nongold_overlap_query_years')} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Temporal statistics of RETECO Track~1 (train). Gold evidence carries "
        r"\emph{fewer} explicit years, and overlaps the query's years \emph{less} often, than "
        r"the non-relevant documents BM25 ranks highly.}",
        r"\label{tab:data}",
        r"\end{table}",
    ]
    return "\n".join(rows)


def numbers_tex(dev: dict, train: dict | None, sel: dict, analysis: dict) -> str:
    """LaTeX macros for every number quoted in the text (regenerated with the tables)."""

    def g(sub, key, field="macro", src=dev):
        r = (src or {}).get(sub, {}).get(key)
        return r.get(field) if isinstance(r, dict) else None

    def f3(v):
        return "--" if v is None else f"{v:.3f}".lstrip("0")

    names = {
        "bm25": "Bm",
        "dense": "Dense",
        "dense_doc": "DenseDoc",
        "raptor": "Raptor",
        "raptor_t": "RaptorT",
        "bm25+dense": "BmDense",
        "pool_sel": "Pool",
        "rm3": "Rm",
        "prior_exp": "PriorExp",
        "prior_tprf": "PriorTprf",
        "bm25+timefilter": "BmTime",
        "bm25+kg": "BmKg",
        "kg": "Kg",
        "llm_q2d": "Qtd",
        "llm_q2d_fb": "QtdFb",
        "llm_multi": "Multi",
        "llm_multi_fb": "MultiFb",
        "llm_kw": "Kw",
        "llm_kw_fb": "KwFb",
        "ce_minilm": "MiniLM",
        "ce_minilm+pool": "MiniLMPool",
        "ce_bge": "Bge",
        "ce_bge+pool": "BgePool",
        "q3rr06": "QsmallRR",
        "q3rr06+pool": "QsmallRRPool",
        "q3rr4": "QbigRR",
        "q3rr4+pool": "QbigRRPool",
        "listwise": "Listwise",
        "chronofuse": "Final",
        "bm25_y05": "BmYhalf",
        "bm25_y0": "BmYzero",
        "late_ce_minilm": "LateMiniLM",
        "late_ce_bge": "LateBge",
        "late_q3rr06": "LateQsmall",
        "late_q3rr4": "LateQbig",
    }

    pretty = {
        "dense": "dense",
        "dense_doc": "dense document vectors",
        "bm25": "BM25",
        "llm_q2d_fb": "LLM-PRF query2doc",
        "llm_q2d": "LLM query2doc",
        "llm_multi": "LLM multi-query",
        "llm_multi_fb": "LLM-PRF multi-query",
        "llm_kw": "LLM keywords",
        "llm_kw_fb": "LLM-PRF keywords",
        "raptor_t": "temporal RAPTOR",
        "raptor": "RAPTOR",
        "kg": "temporal KG",
        "rm3_a03": "RM3",
        "timefilter": "year-filtered BM25",
    }

    def chan(c: str, w: float) -> str:
        c = pretty.get(c, c.replace("_", "\\_"))
        return c if w == 1 else c + " ($\\times$" + str(w) + ")"

    m = {}
    for sub in ("1a", "1b"):
        S = sub[1].upper()
        off = dev[sub]["official_bm25"]
        for key, nm in names.items():
            m[f"r{nm}{S}"] = f3(g(sub, key))
        fin = g(sub, "chronofuse")
        m[f"rOfficial{S}"] = f3(off)
        m[f"gainOfficial{S}"] = m[f"gainBm{S}"] = "--"
        if fin:
            m[f"gainOfficial{S}"] = f"{100 * (fin / off - 1):.0f}"
            m[f"gainBm{S}"] = f"{100 * (fin / g(sub, 'bm25') - 1):.0f}"
        m[f"rRecallFinal{S}"] = f3(g(sub, "chronofuse", "recall100"))
        m[f"rRecallBm{S}"] = f3(g(sub, "bm25", "recall100"))
        m[f"rRecallPool{S}"] = f3(g(sub, "pool_sel", "recall100"))
        m[f"rTrainFinal{S}"] = f3(g(sub, "chronofuse", src=train))
        ch = sel.get(sub, {}).get("weights", {})
        m[f"poolChannels{S}"] = ", ".join(chan(c, w) for c, w in ch.items())
        rr = (dev[sub].get("chronofuse") or {}).get("reranker") or "--"
        m[f"finalRerank{S}"] = RR_PRETTY.get(rr, rr)
        pd = (dev[sub].get("chronofuse") or {}).get("p_vs_dense")
        m[f"pFinalDense{S}"] = (
            "--" if pd is None else (r"$<$\,0.001" if pd < 0.001 else rf"$=$\,{pd:.3f}")
        )
    a = json.loads((M / "data_analysis_train.json").read_text())["summary"]
    m["pctdocyear"] = f"{100 * a['pct_docs_with_year']:.0f}\\%"
    m["pctGoldOverlap"] = f"{100 * a['pct_gold_overlap_query_years']:.0f}\\%"
    m["pctTopOverlap"] = f"{100 * a['pct_bm25top10_nongold_overlap_query_years']:.0f}\\%"
    pl = analysis.get("planner", {})
    m["plannerIntentAcc"] = f"{100 * pl.get('intent_acc', 0):.0f}\\%"
    m["plannerWindowHit"] = f"{100 * pl.get('window_hit_when_gold_has_year', 0):.0f}\\%"
    m["plannerEvents"] = f"{pl.get('avg_events', 0):.1f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    return "% generated by scripts/make_tables.py\n" + "\n".join(lines) + "\n"


def exact_macros(bench: dict, perq: dict) -> None:
    """Replace the 4-decimal macro scores by exact values from the per-query scores, so that
    3-decimal table entries are rounded once (no double rounding)."""
    for sub, systems in perq.items():
        for sid, scores in systems.items():
            by = defaultdict(list)
            for k, v in scores.items():
                by[k.split("/", 1)[0]].append(v)
            if sid in bench.get(sub, {}) and by:
                bench[sub][sid]["macro"] = float(np.mean([np.mean(v) for v in by.values()]))


def main() -> None:
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    dev = json.loads((M / "bench_dev.json").read_text())
    trp = M / "bench_train.json"
    train = json.loads(trp.read_text()) if trp.exists() else None
    perq = json.loads((M / "bench_dev_perquery.json").read_text())
    exact_macros(dev, perq)
    trq = M / "bench_train_perquery.json"
    if train and trq.exists():
        exact_macros(train, json.loads(trq.read_text()))
    (T / "full.tex").write_text(full_table(dev))
    (T / "main.tex").write_text(main_table(dev, train))
    (T / "rerank.tex").write_text(rerank_table(dev, train))
    (T / "data.tex").write_text(data_table())
    per_domain_fig(dev)
    ia = intent_analysis(perq)
    intent_fig(ia)
    out = {"intent": ia, "planner": planner_accuracy()}
    (M / "analysis_dev.json").write_text(json.dumps(out, indent=1))
    selp = M / "pool_selection.json"
    sel = json.loads(selp.read_text()) if selp.exists() else {}
    (T / "numbers.tex").write_text(numbers_tex(dev, train, sel, out))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
