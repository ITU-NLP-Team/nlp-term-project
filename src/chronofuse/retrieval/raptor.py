"""Temporal RAPTOR: a recursive cluster tree over a domain's documents.

RAPTOR (Sarthi et al., 2024) clusters chunk embeddings recursively and
summarises each cluster with an LLM; retrieval over the "collapsed tree"
(all nodes at once) works best in their study. We keep the tree and the
collapsed-tree search but adapt three things to TEMPO and to a CPU budget:

1. **Temporal clustering.** Level-1 clusters are formed on
   ``[embedding ; beta * year-encoding]`` so a node groups documents that are
   about the same topic *and* period; every node stores its members' year span.
2. **Centroid nodes instead of LLM summaries.** A node is represented by the
   normalised centroid of its members (plus an extractive "summary": member
   titles closest to the centroid). Generating ~10^4 LLM summaries per domain is
   infeasible on CPU; the centroid is what the summary embedding approximates.
3. **Ranking documents, not contexts.** Collapsed-tree hits are mapped back to
   documents: a document's score is its own similarity smoothed with its
   ancestors' (cluster-based score regularisation, Kurland & Lee 2004), and
   the children of the best nodes are added to the candidate set (recall).
   Ancestors whose year span matches the query window get a bonus.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..indexing.base import Ranking
from ..metadata.profile import DocProfile

_BINS = np.arange(1900, 2031, 5)  # 5-year bins for the temporal part of the clustering space


def year_encoding(years: list[int]) -> np.ndarray:
    v = np.zeros(len(_BINS), dtype=np.float32)
    ys = [y for y in years if 1900 <= y <= 2035]
    if not ys:
        return v
    for y in ys:
        v += np.exp(-0.5 * ((_BINS + 2.5 - y) / 5.0) ** 2)
    return v / (np.linalg.norm(v) or 1.0)


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class TemporalRaptor:
    def __init__(self, target_size: int = 10, levels: int = 2, beta: float = 0.5, seed: int = 0):
        self.target_size, self.levels, self.beta, self.seed = target_size, levels, beta, seed

    def build(
        self, doc_ids: list[str], doc_emb: np.ndarray, profiles: dict[str, DocProfile]
    ) -> TemporalRaptor:
        self.doc_ids = list(doc_ids)
        self.doc_emb = _normalise(doc_emb.astype(np.float32))
        years = [profiles[d].years if d in profiles else [] for d in doc_ids]
        self.node_emb: list[np.ndarray] = []  # per level: (n_nodes, dim)
        self.parent: list[np.ndarray] = []  # per level: child index -> node index
        self.span: list[list[tuple[int, int] | None]] = []
        cur, cur_years = self.doc_emb, years
        for lvl in range(self.levels):
            k = max(2, len(cur) // self.target_size)
            if k >= len(cur):
                break
            feats = cur
            if lvl == 0 and self.beta > 0:
                ye = np.stack([year_encoding(y) for y in cur_years])
                feats = np.hstack([cur, self.beta * ye])
            km = MiniBatchKMeans(
                n_clusters=k, random_state=self.seed, batch_size=4096, n_init=1, max_iter=50
            )
            lab = km.fit_predict(feats)
            cent = np.zeros((k, cur.shape[1]), dtype=np.float32)
            np.add.at(cent, lab, cur)
            members: list[list[int]] = [[] for _ in range(k)]
            for i, c in enumerate(lab):
                members[c].append(i)
            nyears = [sorted({y for i in m for y in cur_years[i]}) for m in members]
            self.node_emb.append(_normalise(cent))
            self.parent.append(lab.astype(np.int32))
            self.span.append(
                [(ys[len(ys) // 10], ys[-1 - len(ys) // 10]) if ys else None for ys in nyears]
            )
            cur, cur_years = self.node_emb[-1], nyears
        # children of level-1 nodes (documents), for expansion
        self.children: dict[int, np.ndarray] = {}
        if self.parent:
            order = np.argsort(self.parent[0], kind="stable")
            bounds = np.searchsorted(self.parent[0][order], np.arange(len(self.node_emb[0]) + 1))
            for c in range(len(self.node_emb[0])):
                self.children[c] = order[bounds[c] : bounds[c + 1]]
        return self

    def search(
        self,
        q_emb: np.ndarray,
        k: int = 100,
        window: tuple[int, int] | None = None,
        eta: float = 0.3,
        top_nodes: int = 10,
        time_bonus: float = 0.05,
    ) -> Ranking:
        q = q_emb.reshape(-1).astype(np.float32)
        s0 = self.doc_emb @ q
        if not self.node_emb:
            top = np.argsort(-s0)[:k]
            return [(self.doc_ids[i], float(s0[i])) for i in top]
        s1 = self.node_emb[0] @ q
        if window is not None:
            a, b = window
            for c, sp in enumerate(self.span[0]):
                if sp is not None and sp[0] <= b + 1 and a - 1 <= sp[1]:
                    s1[c] += time_bonus
        anc = s1[self.parent[0]]
        if len(self.node_emb) > 1:
            s2 = self.node_emb[1] @ q
            anc = 0.7 * anc + 0.3 * s2[self.parent[1]][self.parent[0]]
        score = (1 - eta) * s0 + eta * anc
        cand = set(np.argpartition(-s0, min(k, len(s0) - 1))[:k].tolist())
        for c in np.argsort(-s1)[:top_nodes]:  # collapsed-tree hits -> expand to children
            cand.update(self.children.get(int(c), np.empty(0, np.int32)).tolist())
        cand = np.fromiter(cand, dtype=np.int64)
        best = cand[np.argsort(-score[cand])][:k]
        return [(self.doc_ids[i], float(score[i])) for i in best]
