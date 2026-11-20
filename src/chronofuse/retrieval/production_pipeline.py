"""Production Hybrid Retrieval and Temporal Reranking Pipeline.

Combines:
- Dynamic Hugging Face corpus discovery and loading (track1_tempo in DataScience-UIBK/RETECO-SemEval2027)
- Temporal metadata extraction & [YEAR: YYYY] anchor injection
- Sparse indexing with BM25Okapi (rank_bm25)
- Dense semantic bi-encoder embeddings via BAAI/bge-base-en-v1.5 (with .npy caching/loading)
- Reciprocal Rank Fusion (RRF, c=60)
- Cross-Encoder reranking via BAAI/bge-reranker-large
- GPU acceleration (torch.cuda) with automatic CPU fallback
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger("chronofuse.production_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_default_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Kullanılan cihaz: %s", device)
    if device == "cuda":
        logger.info("GPU Modeli: %s", torch.cuda.get_device_name(0))
    return device


def extract_year_from_text(text: str) -> str:
    """Extract a 4-digit year (1000-2035) from text using regex."""
    matches = re.findall(r"\b(1[0-9]{3}|20[0-3][0-9])\b", text)
    return matches[0] if matches else ""


class TemporalHybridSearchEngine:
    """Production hybrid search engine with BM25, Dense embeddings, RRF, and CrossEncoder reranker."""

    def __init__(
        self,
        repo_id: str = "DataScience-UIBK/RETECO-SemEval2027",
        dense_model_name: str = "BAAI/bge-base-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-large",
        device: str | None = None,
        cache_dir: str | Path = "data/embeddings",
    ) -> None:
        self.repo_id = repo_id
        self.dense_model_name = dense_model_name
        self.reranker_model_name = reranker_model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 1. Device check
        self.device = device if device is not None else get_default_device()

        # 2. Models
        logger.info("[1/5] Modeller yükleniyor...")
        self.dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
        self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)

        # 3. Storage
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.dense_embeddings: np.ndarray | None = None

    # -------------------------------------------------------------------------
    # Corpus preparation & Temporal Metadata Injection
    # -------------------------------------------------------------------------
    def fetch_corpus_files(self) -> list[str]:
        """Discover track1_tempo corpus / documents files from Hugging Face."""
        api = HfApi()
        logger.info("[2/5] Depodaki Doküman Depoları (Corpus) taranıyor...")
        repo_files = api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")

        corpus_files = [
            f for f in repo_files
            if "corpus" in f and (f.endswith(".parquet") or f.endswith(".jsonl"))
        ]

        if not corpus_files:
            corpus_files = [
                f for f in repo_files
                if ("examples_train" in f or "documents" in f)
                and (f.endswith(".parquet") or f.endswith(".jsonl"))
            ]

        logger.info("Bulunan Doküman Dosyası Sayısı: %d", len(corpus_files))
        for f in corpus_files[:5]:
            logger.info(" -> %s", f)

        return corpus_files

    def load_and_process_corpus(
        self,
        corpus_files: list[str] | None = None,
        max_files: int = 3,
        max_docs_per_file: int | None = None,
    ) -> tuple[list[str], list[str], list[list[str]]]:
        """Download corpus files and inject [YEAR: YYYY] temporal tags."""
        if corpus_files is None:
            corpus_files = self.fetch_corpus_files()

        documents: list[str] = []
        doc_ids: list[str] = []
        tokenized_corpus: list[list[str]] = []

        logger.info("[3/5] Dokümanlar işleniyor ve zamansal metadata ekleniyor...")
        target_files = corpus_files[:max_files] if max_files else corpus_files

        for file_path in target_files:
            logger.info(" -> Okunuyor: %s", file_path)
            local_path = hf_hub_download(
                repo_id=self.repo_id, filename=file_path, repo_type="dataset"
            )

            if file_path.endswith(".parquet"):
                df = pd.read_parquet(local_path)
            else:
                df = pd.read_json(local_path, lines=True)

            if max_docs_per_file is not None:
                df = df.iloc[:max_docs_per_file]

            for idx, row in df.iterrows():
                doc_id = str(row.get("id") or row.get("doc_id") or f"{file_path}_{idx}")
                
                # Check for explicit year column, else regex extract from text
                text_val = str(row.get("text") or row.get("content") or row.get("passage") or "")
                year = row.get("year") if "year" in row and pd.notna(row["year"]) else ""
                if not year:
                    year = extract_year_from_text(text_val)

                title = str(row.get("title", "")) if "title" in row and pd.notna(row["title"]) else ""

                # Temporal Anchor Injection: [YEAR: YYYY]
                year_tag = f"[YEAR: {year}] " if year else ""
                full_text = f"{year_tag}{title} {text_val}".strip()

                if full_text:
                    documents.append(full_text)
                    doc_ids.append(doc_id)
                    tokenized_corpus.append(full_text.lower().split())

        self.documents = documents
        self.doc_ids = doc_ids
        self.tokenized_corpus = tokenized_corpus

        logger.info("Toplam Hazırlanan Doküman Sayısı: %d", len(self.documents))
        return self.documents, self.doc_ids, self.tokenized_corpus

    # -------------------------------------------------------------------------
    # BM25 Sparse Index
    # -------------------------------------------------------------------------
    def build_bm25_index(self) -> BM25Okapi:
        """Create BM25Okapi index over the tokenized corpus."""
        assert self.tokenized_corpus, "Corpus is empty. Call load_and_process_corpus first."
        logger.info("[5/5] BM25 İndeksi oluşturuluyor...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return self.bm25

    # -------------------------------------------------------------------------
    # Dense Embeddings Extraction & Caching
    # -------------------------------------------------------------------------
    def extract_dense_embeddings(
        self,
        batch_size: int = 256,
        save_path: str | Path | None = None,
        save_ids_path: str | Path | None = None,
    ) -> np.ndarray:
        """Encode documents with BAAI/bge-base-en-v1.5 and save .npy to disk."""
        assert self.documents, "Documents list is empty. Call load_and_process_corpus first."
        logger.info("[4/5] GPU/CPU ile Dense Embedding'ler çıkarılıyor...")
        dense_embeddings = self.dense_model.encode(
            self.documents,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # Cosine Sim için L2 Norm
        )
        self.dense_embeddings = np.array(dense_embeddings, dtype=np.float32)

        # Save to specified paths, or default to root and cache_dir
        emb_file = Path(save_path) if save_path else self.cache_dir / "full_dense_embeddings.npy"
        ids_file = Path(save_ids_path) if save_ids_path else self.cache_dir / "full_doc_ids.npy"
        docs_file = emb_file.parent / "full_documents.npy"

        np.save(emb_file, self.dense_embeddings)
        np.save(ids_file, np.array(self.doc_ids))
        np.save(docs_file, np.array(self.documents, dtype=object))

        # Also write root copies if cache_dir is different
        if emb_file.resolve() != Path("full_dense_embeddings.npy").resolve():
            try:
                np.save("full_dense_embeddings.npy", self.dense_embeddings)
                np.save("full_doc_ids.npy", np.array(self.doc_ids))
            except Exception:
                pass

        logger.info(" -> [KAYDEDİLDİ] %s ve %s diske yazıldı.", emb_file, ids_file)
        return self.dense_embeddings

    # -------------------------------------------------------------------------
    # Load Pre-saved .npy Vectors
    # -------------------------------------------------------------------------
    def load_saved_npy(
        self,
        embeddings_path: str | Path = "full_dense_embeddings.npy",
        doc_ids_path: str | Path = "full_doc_ids.npy",
        documents_path: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-computed .npy vector matrices and document IDs (takes seconds)."""
        emb_p = Path(embeddings_path)
        ids_p = Path(doc_ids_path)

        # Fallback to cache_dir if not found in root
        if not emb_p.exists() and (self.cache_dir / emb_p.name).exists():
            emb_p = self.cache_dir / emb_p.name
        if not ids_p.exists() and (self.cache_dir / ids_p.name).exists():
            ids_p = self.cache_dir / ids_p.name

        if not emb_p.exists():
            raise FileNotFoundError(f"Vektör dosyası bulunamadı: {emb_p}")
        if not ids_p.exists():
            raise FileNotFoundError(f"Doc ID dosyası bulunamadı: {ids_p}")

        logger.info("Önceden kaydedilen NPY vektörleri yükleniyor: %s ...", emb_p)
        self.dense_embeddings = np.load(emb_p)
        self.doc_ids = np.load(ids_p).tolist()

        logger.info("Yüklenen Vektör Matrisi Şekli: %s", str(self.dense_embeddings.shape))
        logger.info("Yüklenen Doc ID Sayısı: %d", len(self.doc_ids))

        # Attempt to load saved documents if available
        doc_candidates = [
            Path(documents_path) if documents_path else None,
            emb_p.parent / "full_documents.npy",
            Path("full_documents.npy"),
            self.cache_dir / "full_documents.npy",
        ]
        for cand in doc_candidates:
            if cand and cand.exists():
                self.documents = np.load(cand, allow_pickle=True).tolist()
                self.tokenized_corpus = [doc.lower().split() for doc in self.documents]
                self.build_bm25_index()
                logger.info("Kaydedilmiş doküman metinleri ve BM25 indeksi yüklendi (%d doküman).", len(self.documents))
                break

        return self.dense_embeddings, np.array(self.doc_ids)

    # -------------------------------------------------------------------------
    # SOTA Hybrid Search & Reranking
    # -------------------------------------------------------------------------
    def production_hybrid_search(
        self,
        query_text: str,
        top_k_candidates: int = 100,
        final_top_k: int = 10,
        rrf_c: int = 60,
    ) -> list[dict[str, Any]]:
        """Execute hybrid search using BM25, Dense embeddings, RRF, and CrossEncoder."""
        assert self.dense_embeddings is not None, "Dense embeddings yüklenmedi."
        n_docs = len(self.doc_ids)
        if n_docs == 0:
            return []

        # Adjust candidate count if corpus is smaller
        k_candidates = min(top_k_candidates, n_docs)

        # 1. BM25 Adayları (BM25 mevcutsa kullanılır, yoksa dense kullanılır)
        bm25_top_indices: list[int] = []
        if self.bm25 is not None:
            query_lower = query_text.lower().split()
            bm25_scores = self.bm25.get_scores(query_lower)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:k_candidates].tolist()

        # 2. Dense Vector Adayları
        query_embedding = self.dense_model.encode(query_text, normalize_embeddings=True)
        dense_scores = np.dot(self.dense_embeddings, query_embedding)
        dense_top_indices = np.argsort(dense_scores)[::-1][:k_candidates].tolist()

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}

        if bm25_top_indices:
            for rank, idx in enumerate(bm25_top_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_c + rank + 1))

        for rank, idx in enumerate(dense_top_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_c + rank + 1))

        candidate_indices = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[:k_candidates]

        # 4. Cross-Encoder Reranking
        doc_texts = self.documents if self.documents else [str(self.doc_ids[idx]) for idx in range(n_docs)]
        pairs = [[query_text, doc_texts[idx]] for idx in candidate_indices]
        rerank_scores = self.reranker.predict(pairs, batch_size=32)

        final_ranks = np.argsort(rerank_scores)[::-1][:final_top_k]

        results: list[dict[str, Any]] = []
        for rank_idx in final_ranks:
            doc_idx = candidate_indices[rank_idx]
            results.append(
                {
                    "doc_id": self.doc_ids[doc_idx],
                    "score": float(rerank_scores[rank_idx]),
                    "rrf_score": float(rrf_scores.get(doc_idx, 0.0)),
                    "text": doc_texts[doc_idx],
                }
            )

        return results

    def search(self, query: str, top_k: int = 10, top_k_candidates: int = 100) -> list[dict[str, Any]]:
        """Convenience alias for production_hybrid_search."""
        return self.production_hybrid_search(
            query_text=query,
            top_k_candidates=top_k_candidates,
            final_top_k=top_k,
        )
