"""Temporal statistics of RETECO Track 1 (train split by default).

For every domain: share of queries with explicit years / relative expressions,
guidance intent distribution, and — the key diagnostic for a temporal prior —
how often gold documents vs. BM25 top-10 non-gold documents overlap the query's
years. Writes ``results/metrics/data_analysis_<split>.json``.

    python scripts/analyze_data.py --split train
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from chronofuse.config import resolve_path
from chronofuse.data.reteco import TRACK1_DOMAINS, load_guidance, load_qrels, load_queries
from chronofuse.eval.metrics import read_trec
from chronofuse.metadata.profile import read_profiles
from chronofuse.metadata.temporal import extract_temporal, year_overlap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--run", default="results/runs/e0_bm25")
    args = ap.parse_args()
    root = resolve_path("data/raw/reteco")
    out: dict = {"per_domain": {}, "intents": Counter()}
    tot = Counter()
    for dom in TRACK1_DOMAINS:
        qs = load_queries(root, dom, args.split)
        qrels = load_qrels(root, dom, args.split)
        guid = load_guidance(root, dom, args.split)
        prof = read_profiles(resolve_path("data/indexes") / dom / "profiles.jsonl")
        run = read_trec(resolve_path(args.run) / f"1a_{args.split}" / f"{dom}.trec")
        c = Counter()
        for q in qs:
            tp = extract_temporal(q.text)
            g = guid.get(q.id, {}).get("query_guidance", {})
            out["intents"][g.get("temporal_intent", "none")] += 1
            c["queries"] += 1
            c["q_years"] += bool(tp.years)
            c["q_relative"] += bool(tp.relative)
            c["guid_temporal"] += bool(g.get("is_temporal_query"))
            gold = [d for d, r in qrels.get(q.id, {}).items() if r > 0]
            c["gold"] += len(gold)
            c["gold_has_year"] += sum(bool(prof[d].years) for d in gold if d in prof)
            top = [d for d, _ in run.get(q.id, [])[:10] if d not in set(gold)]
            c["top_nongold"] += len(top)
            c["top_has_year"] += sum(bool(prof[d].years) for d in top if d in prof)
            if tp.years:
                c["gold_yq"] += len(gold)
                c["gold_overlap"] += sum(
                    year_overlap(tp.years, prof[d].years) > 0 for d in gold if d in prof
                )
                c["top_yq"] += len(top)
                c["top_overlap"] += sum(
                    year_overlap(tp.years, prof[d].years) > 0 for d in top if d in prof
                )
        c["docs"] = len(prof)
        c["docs_has_year"] = sum(bool(p.years) for p in prof.values())
        tot.update(c)
        out["per_domain"][dom] = dict(c)
        print(dom, dict(c), flush=True)

    def r(a, b):
        return round(tot[a] / max(1, tot[b]), 4)

    out["summary"] = {
        "queries": tot["queries"],
        "pct_queries_explicit_year": r("q_years", "queries"),
        "pct_queries_relative_expr": r("q_relative", "queries"),
        "pct_queries_guidance_temporal": r("guid_temporal", "queries"),
        "gold_per_query": r("gold", "queries"),
        "pct_docs_with_year": r("docs_has_year", "docs"),
        "pct_gold_with_year": r("gold_has_year", "gold"),
        "pct_bm25top10_nongold_with_year": r("top_has_year", "top_nongold"),
        "pct_gold_overlap_query_years": r("gold_overlap", "gold_yq"),
        "pct_bm25top10_nongold_overlap_query_years": r("top_overlap", "top_yq"),
    }
    out["intents"] = dict(out["intents"].most_common())
    print(json.dumps(out["summary"], indent=2), json.dumps(out["intents"]))
    path = resolve_path("results/metrics") / f"data_analysis_{args.split}.json"
    Path(path).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
