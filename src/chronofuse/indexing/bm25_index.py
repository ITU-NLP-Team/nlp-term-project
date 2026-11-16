"""Passage-level BM25 via ``bm25s`` (Lucene scoring variant), with MaxP to documents.

Two ways to build the same index:

* :meth:`BM25Index.build` — in-memory, via ``bm25s`` directly (small corpora, tests).
* :meth:`BM25Index.build_streaming` — memory-lean: passages are tokenised in
  batches and kept as compact ``int32`` arrays; the sparse score matrix is
  assembled with NumPy/SciPy. It produces exactly the ``bm25s`` "lucene" scores
  (Lucene IDF, Robertson TF component) and saves in the ``bm25s`` format, so
  :meth:`load` and :meth:`search` are shared. Needed for the largest TEMPO
  domains (e.g. *history*: 200k documents, 112M words) on a ~8 GB RAM machine.
"""

from __future__ import annotations

import gc
import itertools
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import bm25s
import numpy as np
import orjson
import scipy.sparse as sp
import Stemmer
from bm25s.tokenization import Tokenizer

from .base import Ranking, Retriever, maxp


def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(itertools.islice(it, n)):
        yield batch


class BM25Index(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 0.9, b: float = 0.4, stemmer: str = "english"):
        self.k1, self.b = k1, b
        self.stemmer = Stemmer.Stemmer(stemmer) if stemmer else None
        self.model: bm25s.BM25 | None = None
        self.pids: list[str] = []

    # ------------------------------------------------------------------ build
    def build(self, pids: list[str], texts: list[str]) -> BM25Index:
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.model = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self.model.index(tokens, show_progress=False)
        self.pids = list(pids)
        return self

    def build_streaming(
        self, passages: Iterable[tuple[str, str]], batch_size: int = 20000
    ) -> BM25Index:
        """Build from an iterable of ``(pid, text)`` without holding all texts in memory."""
        tok = Tokenizer(stopwords="en", stemmer=self.stemmer)
        flat_batches: list[np.ndarray] = []  # concatenated token ids per batch
        len_batches: list[np.ndarray] = []  # passage lengths per batch
        self.pids = []
        for batch in _batched(passages, batch_size):
            self.pids.extend(pid for pid, _ in batch)
            ids = tok.tokenize(
                [t for _, t in batch], update_vocab=True, return_as="ids", show_progress=False
            )
            # The Tokenizer class maps token-less passages to the empty token "" (id 0);
            # bm25s.tokenize gives them length 0. Drop id 0 to match its scores exactly.
            ids = [[x for x in doc if x != 0] for doc in ids]
            len_batches.append(np.fromiter((len(x) for x in ids), dtype=np.int32, count=len(ids)))
            flat_batches.append(np.fromiter(itertools.chain.from_iterable(ids), dtype=np.int32))
            del ids, batch
        vocab: dict[str, int] = dict(tok.get_vocab_dict())
        assert vocab.get("") == 0, "bm25s Tokenizer is expected to reserve id 0 for ''"
        n_docs, n_vocab = len(self.pids), len(vocab)
        doc_len = np.concatenate(len_batches) if len_batches else np.zeros(0, np.int32)
        avg_len = float(doc_len.mean()) if n_docs else 1.0

        # Pass 1: unique (passage, token) pairs with term frequencies; document frequencies.
        rows, cols, tfs = [], [], []
        df = np.zeros(n_vocab, dtype=np.int64)
        offset = 0
        for flat, lens in zip(flat_batches, len_batches):
            local = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
            keys, tf = np.unique(local * n_vocab + flat, return_counts=True)
            r = (keys // n_vocab).astype(np.int32) + offset
            c = (keys % n_vocab).astype(np.int32)
            df += np.bincount(c, minlength=n_vocab)
            rows.append(r)
            cols.append(c)
            tfs.append(tf.astype(np.float32))
            offset += len(lens)
            del local, keys, tf
        del flat_batches
        gc.collect()

        # Pass 2: BM25 (lucene) score per non-zero = idf(df) * tf / (k1*(1-b+b*dl/avgdl) + tf)
        idf = np.array([math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for d in df], dtype=np.float32)
        norm = (self.k1 * ((1 - self.b) + self.b * doc_len / avg_len)).astype(np.float32)
        for i in range(len(rows)):
            tf = tfs[i]
            tfs[i] = idf[cols[i]] * tf / (norm[rows[i]] + tf)
        row = np.concatenate(rows)
        del rows
        col = np.concatenate(cols)
        del cols
        data = np.concatenate(tfs)
        del tfs
        gc.collect()
        mat = sp.csc_matrix((data, (row, col)), shape=(n_docs, n_vocab), dtype=np.float32)
        del row, col, data
        gc.collect()

        self.model = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
        self.model.scores = {
            "data": mat.data,
            "indices": mat.indices,
            "indptr": mat.indptr,
            "num_docs": n_docs,
        }
        self.model.vocab_dict = vocab
        self.model.nonoccurrence_array = None
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        self.model.save(str(path), show_progress=False)
        (path / "pids.json").write_bytes(orjson.dumps(self.pids))

    @classmethod
    def load(cls, path: str | Path, stemmer: str = "english", mmap: bool = True) -> BM25Index:
        path = Path(path)
        obj = cls(stemmer=stemmer)
        obj.model = bm25s.BM25.load(str(path), mmap=mmap)
        obj.pids = orjson.loads((path / "pids.json").read_bytes())
        return obj

    # ------------------------------------------------------- weighted queries
    def analyze(self, text: str) -> list[str]:
        """Tokenise like the index (lowercase, English stopwords, Snowball stem)."""
        toks = bm25s.tokenize(
            [text], stopwords="en", stemmer=self.stemmer, show_progress=False, return_ids=False
        )
        return list(toks[0]) if toks else []

    def passage_doc_index(self) -> tuple[np.ndarray, list[str]]:
        """``(doc_idx per passage, doc ids)`` — cached; used for masks and pooling."""
        if getattr(self, "_pdoc", None) is None:
            docs: dict[str, int] = {}
            pdoc = np.empty(len(self.pids), dtype=np.int32)
            for i, pid in enumerate(self.pids):
                pdoc[i] = docs.setdefault(pid.rsplit("#", 1)[0], len(docs))
            self._pdoc, self._docs = pdoc, list(docs)
        return self._pdoc, self._docs

    def score_terms(self, weights: dict[str, float], mask: np.ndarray | None = None) -> np.ndarray:
        """Passage scores for a weighted bag of (analysed) terms: sum_t w_t * BM25(t, p)."""
        assert self.model is not None
        s = self.model.scores
        data, indptr, indices = s["data"], s["indptr"], s["indices"]
        vocab = self.model.vocab_dict
        out = np.zeros(s["num_docs"], dtype=np.float32)
        for term, w in weights.items():
            t = vocab.get(term)
            if t is None or w == 0:
                continue
            a, b = indptr[t], indptr[t + 1]
            out[indices[a:b]] += w * data[a:b]
        if mask is not None:
            out *= mask
        return out

    def search_weighted(
        self, weights: dict[str, float], k: int = 100, mask: np.ndarray | None = None
    ) -> Ranking:
        scores = self.score_terms(weights, mask)
        n = min(k * 5, len(scores))
        top = np.argpartition(-scores, n - 1)[:n] if n < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [
            (d, s) for d, s in maxp([self.pids[i] for i in top], scores[top].tolist(), k) if s > 0
        ]

    def query_weights(self, text: str) -> dict[str, float]:
        """Raw term counts of the analysed query (Lucene sums duplicate clauses)."""
        w: dict[str, float] = {}
        for t in self.analyze(text):
            w[t] = w.get(t, 0.0) + 1.0
        return w

    # ----------------------------------------------------------------- search
    def search(
        self, queries: list[str], k: int = 100, passage_k: int | None = None
    ) -> list[Ranking]:
        assert self.model is not None, "index not built/loaded"
        passage_k = min(passage_k or k * 5, len(self.pids))
        q_tokens = bm25s.tokenize(
            queries, stopwords="en", stemmer=self.stemmer, show_progress=False
        )
        idx, scores = self.model.retrieve(q_tokens, k=passage_k, show_progress=False)
        out: list[Ranking] = []
        for row_i, row_s in zip(idx, scores):
            pids = [self.pids[i] for i in row_i]
            out.append([(d, s) for d, s in maxp(pids, list(row_s), k) if s > 0])
        return out
