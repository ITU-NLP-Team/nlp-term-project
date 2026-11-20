#!/usr/bin/env python3
"""Main execution and test script for ChronoFuse Temporal Hybrid Search.

Supports:
- Direct loading of precomputed NPY vectors (full_dense_embeddings.npy, full_doc_ids.npy)
- Dynamic corpus fetching and embedding generation if NPY files are not present
- Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion (c=60)
- BAAI/bge-reranker-large Cross-Encoder reranking
- Display of formatted search results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add 'src' to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
from chronofuse.retrieval.production_pipeline import (
    TemporalHybridSearchEngine,
    get_default_device,
)


def find_npy_files(cache_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Look for full_dense_embeddings.npy and full_doc_ids.npy in root or cache_dir."""
    candidates_emb = [
        PROJECT_ROOT / "full_dense_embeddings.npy",
        cache_dir / "full_dense_embeddings.npy",
    ]
    candidates_ids = [
        PROJECT_ROOT / "full_doc_ids.npy",
        cache_dir / "full_doc_ids.npy",
    ]
    candidates_docs = [
        PROJECT_ROOT / "full_documents.npy",
        cache_dir / "full_documents.npy",
    ]

    emb_path = next((p for p in candidates_emb if p.exists()), None)
    ids_path = next((p for p in candidates_ids if p.exists()), None)
    docs_path = next((p for p in candidates_docs if p.exists()), None)
    return emb_path, ids_path, docs_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ChronoFuse Temporal Hybrid Search & Reranking."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What major technical changes occurred before 2020?",
        help="Temporal query to search.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=50,
        help="Top K candidates for RRF fusion before Cross-Encoder rerank (default: 50).",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=5,
        help="Number of final reranked results to display (default: 5).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/embeddings",
        help="Embeddings directory (default: data/embeddings).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force downloading corpus from Hugging Face and extracting dense embeddings.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Max number of corpus files to process when building (default: 3).",
    )
    parser.add_argument(
        "--max-docs-per-file",
        type=int,
        default=None,
        help="Optional limit on docs per file when building.",
    )
    args = parser.parse_args()

    cache_dir = PROJECT_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize engine
    engine = TemporalHybridSearchEngine(cache_dir=cache_dir)

    # 2. Check for pre-saved NPY files
    emb_file, ids_file, docs_file = find_npy_files(cache_dir)

    if emb_file and ids_file and not args.build:
        print("\n" + "=" * 60)
        print("[NPY MODU] Önceden kaydedilen .npy vektörleri yükleniyor...")
        print(f" -> Vektör dosyası : {emb_file}")
        print(f" -> Doc ID dosyası : {ids_file}")
        print("=" * 60)
        engine.load_saved_npy(
            embeddings_path=emb_file,
            doc_ids_path=ids_file,
            documents_path=docs_file,
        )
    else:
        print("\n" + "=" * 60)
        if args.build:
            print("[BUILD MODU] Hugging Face'ten korpus indirilip embedding çıkarılıyor...")
        else:
            print("[BİLGİ] .npy dosyaları henüz bulunamadı. Korpus hazırlanıyor...")
        print("=" * 60)
        engine.load_and_process_corpus(
            max_files=args.max_files,
            max_docs_per_file=args.max_docs_per_file,
        )
        engine.extract_dense_embeddings()
        engine.build_bm25_index()

    # 3. Test Sorgusu
    sample_query = args.query
    print("\n" + "=" * 60)
    print(f"SORGUMUZ: {sample_query}")
    print("=" * 60 + "\n")

    results = engine.production_hybrid_search(
        query_text=sample_query,
        top_k_candidates=args.candidates,
        final_top_k=args.final_top_k,
    )

    # 4. Sonuçları Yazdır
    for rank, res in enumerate(results, 1):
        print(f"Sıra {rank} | Doc ID: {res['doc_id']} | Reranker Score: {res['score']:.4f}")
        text_preview = str(res.get("text", "")).replace("\n", " ")
        if len(text_preview) > 250:
            text_preview = text_preview[:250] + "..."
        print(f"Metin: {text_preview}\n" + "-" * 50)


if __name__ == "__main__":
    main()
