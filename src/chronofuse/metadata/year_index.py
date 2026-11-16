"""Temporal inverted index: year -> documents, and passage masks for BM25.

Every document's normalised years (from its :class:`DocProfile`) are posted
under each year. Given a query window [a, b] (+- tolerance) the index returns
the set of documents that mention a year in the window, as a passage-level 0/1
mask that the weighted BM25 scorer multiplies in. This turns "BM25 restricted
to the right period" into a single vectorised operation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .profile import DocProfile


class YearIndex:
    def __init__(self, pdoc: np.ndarray, doc_ids: list[str], profiles: dict[str, DocProfile]):
        self.pdoc = pdoc
        post: dict[int, list[int]] = defaultdict(list)
        for i, d in enumerate(doc_ids):
            p = profiles.get(d)
            if p is not None:
                for y in p.years:
                    post[y].append(i)
        self.postings = {y: np.asarray(v, dtype=np.int32) for y, v in post.items()}
        self.n_docs = len(doc_ids)

    def doc_mask(self, window: tuple[int, int], tolerance: int = 1) -> np.ndarray:
        m = np.zeros(self.n_docs, dtype=bool)
        a, b = window
        for y in range(a - tolerance, b + tolerance + 1):
            ids = self.postings.get(y)
            if ids is not None:
                m[ids] = True
        return m

    def passage_mask(self, window: tuple[int, int], tolerance: int = 1) -> np.ndarray:
        return self.doc_mask(window, tolerance)[self.pdoc].astype(np.float32)
