"""Pseudo-relevance feedback (PRF) on top of the BM25 index.

* :func:`rm3_weights` — classic RM3 (Lavrenko & Croft 2001; Abdul-Jaleel et al.
  2004): the top-``fb_docs`` documents of a first-pass BM25 ranking define a
  relevance model P(t|R) = sum_d P(d|q) P(t|d); the ``fb_terms`` most likely
  terms are interpolated with the original query,
  w(t) = (1-alpha) P(t|q) + alpha P(t|R).
  This is the statistical answer to "can patterns be mined from BM25 hits?":
  co-occurring vocabulary *and* year tokens of the feedback set are surfaced.

* :func:`temporal_feedback` — temporal PRF: the year distribution of the
  feedback documents (weighted by retrieval score and contrasted with the
  corpus year prior) yields an *inferred* year window for queries that state no
  explicit date (cf. Jones & Diaz 2007, temporal profiles of queries).

* :func:`downweight_years` — de-emphasise year tokens in a weighted query.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..indexing.base import Ranking
from ..metadata.profile import DocProfile

_YEAR_TOK = re.compile(r"^(1[0-9]{3}|20[0-3][0-9])$")


def is_year_token(t: str) -> bool:
    return bool(_YEAR_TOK.match(t))


def rm3_weights(
    q_weights: dict[str, float],
    ranking: Ranking,
    doc_tokens: dict[str, list[str]],
    fb_docs: int = 10,
    fb_terms: int = 20,
    alpha: float = 0.5,
    min_len: int = 2,
) -> dict[str, float]:
    top = [(d, s) for d, s in ranking[:fb_docs] if d in doc_tokens and doc_tokens[d]]
    if not top:
        return dict(q_weights)
    # P(d|q): softmax-free normalisation of positive BM25 scores (standard in Anserini RM3).
    z = sum(max(s, 0.0) for _, s in top) or 1.0
    rel: Counter = Counter()
    for d, s in top:
        toks = doc_tokens[d]
        tf = Counter(t for t in toks if len(t) >= min_len)
        n = len(toks)
        for t, c in tf.items():
            rel[t] += (max(s, 0.0) / z) * c / n
    fb = dict(rel.most_common(fb_terms))
    zf = sum(fb.values()) or 1.0
    zq = sum(q_weights.values()) or 1.0
    out: dict[str, float] = {}
    for t, w in q_weights.items():
        out[t] = (1 - alpha) * w / zq
    for t, w in fb.items():
        out[t] = out.get(t, 0.0) + alpha * w / zf
    return out


def downweight_years(weights: dict[str, float], gamma: float) -> dict[str, float]:
    return {t: (w * gamma if is_year_token(t) else w) for t, w in weights.items()}


def corpus_year_prior(profiles: dict[str, DocProfile]) -> dict[int, float]:
    df: Counter = Counter()
    for p in profiles.values():
        df.update(set(p.years))
    n = max(1, len(profiles))
    return {y: c / n for y, c in df.items()}


def temporal_feedback(
    ranking: Ranking,
    profiles: dict[str, DocProfile],
    prior: dict[int, float],
    fb_docs: int = 20,
    min_support: int = 3,
    max_span: int = 10,
) -> tuple[int, int] | None:
    """Infer a year window from the feedback set; ``None`` if the evidence is weak.

    Each feedback document spreads its (rank-discounted) weight over its years;
    a year's score is that mass times log(1 + 1/prior) so ubiquitous years
    (e.g. the corpus' crawl year) do not dominate. The window is grown around
    the best year while neighbours keep >= 50% of its score.
    """
    mass: Counter = Counter()
    support: Counter = Counter()
    for r, (d, _) in enumerate(ranking[:fb_docs]):
        p = profiles.get(d)
        if p is None or not p.years or len(p.years) > 40:
            continue
        w = 1.0 / math.log2(r + 2)
        for y in p.years:
            mass[y] += w / len(p.years)
            support[y] += 1
    scored = {
        y: m * math.log1p(1.0 / max(prior.get(y, 1e-4), 1e-4))
        for y, m in mass.items()
        if support[y] >= min_support
    }
    if not scored:
        return None
    best = max(scored, key=scored.get)
    lo = hi = best
    thr = 0.5 * scored[best]
    while hi - lo < max_span and scored.get(hi + 1, 0) >= thr:
        hi += 1
    while hi - lo < max_span and scored.get(lo - 1, 0) >= thr:
        lo -= 1
    return lo, hi
