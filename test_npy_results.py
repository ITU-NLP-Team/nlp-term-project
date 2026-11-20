#!/usr/bin/env python3
"""Utility script to inspect and test precomputed .npy embeddings and doc IDs.

Usage:
    python test_npy_results.py
    python test_npy_results.py --query "What happened in 2018 regarding IOTA consensus?"
    python test_npy_results.py --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


def find_files() -> tuple[Path | None, Path | None, Path | None]:
    candidates_emb = [
        PROJECT_ROOT / "full_dense_embeddings.npy",
        PROJECT_ROOT / "data" / "embeddings" / "full_dense_embeddings.npy",
    ]
    candidates_ids = [
        PROJECT_ROOT / "full_doc_ids.npy",
        PROJECT_ROOT / "data" / "embeddings" / "full_doc_ids.npy",
    ]
    candidates_docs = [
        PROJECT_ROOT / "full_documents.npy",
        PROJECT_ROOT / "data" / "embeddings" / "full_documents.npy",
    ]

    emb_path = next((p for p in candidates_emb if p.exists()), None)
    ids_path = next((p for p in candidates_ids if p.exists()), None)
    docs_path = next((p for p in candidates_docs if p.exists()), None)
    return emb_path, ids_path, docs_path


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and test .npy embeddings.")
    parser.add_argument(
        "--query",
        type=str,
        default="What major technical changes occurred before 2020?",
        help="Query to test against the .npy embeddings.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=True,
        help="Apply BAAI/bge-reranker-large cross-encoder reranking.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("      CHRONOFUSE: .NPY VEKTÖR VE SONUÇ TEST ARACI")
    print("=" * 70)

    emb_path, ids_path, docs_path = find_files()

    if not emb_path or not ids_path:
        print("\n[UYARI] .npy dosyaları henüz dizinde bulunamadı!")
        print("\nLütfen .npy dosyalarınızı şu konumlardan birine yerleştirin:")
        print(f" 1. {PROJECT_ROOT}/full_dense_embeddings.npy")
        print(f"    {PROJECT_ROOT}/full_doc_ids.npy")
        print(" VEYA")
        print(f" 2. {PROJECT_ROOT}/data/embeddings/full_dense_embeddings.npy")
        print(f"    {PROJECT_ROOT}/data/embeddings/full_doc_ids.npy")
        print("\nOpsiyonel (metinleri de görmek için):")
        print(f"    {PROJECT_ROOT}/full_documents.npy")
        print("\nDosyaları ekledikten sonra bu betiği tekrar çalıştırabilirsiniz:")
        print("    python test_npy_results.py")
        sys.exit(0)

    # 1. Dosya Bilgileri
    print(f"\n[1] Dosyalar Bulundu:")
    print(f" -> Vektör Dosyası : {emb_path.relative_to(PROJECT_ROOT)} ({format_bytes(emb_path.stat().st_size)})")
    print(f" -> Doc ID Dosyası : {ids_path.relative_to(PROJECT_ROOT)} ({format_bytes(ids_path.stat().st_size)})")
    if docs_path:
        print(f" -> Doküman Metinleri : {docs_path.relative_to(PROJECT_ROOT)} ({format_bytes(docs_path.stat().st_size)})")

    # 2. Vektörlerin Yüklenmesi
    print("\n[2] NPY Dosyaları Yükleniyor...")
    embeddings = np.load(emb_path)
    doc_ids = np.load(ids_path)
    documents = np.load(docs_path, allow_pickle=True) if docs_path else None

    n_docs, dim = embeddings.shape
    print(f" -> Toplam Vektör Sayısı : {n_docs:,}")
    print(f" -> Vektör Boyutu (Dim)  : {dim}")
    print(f" -> Veri Tipi (dtype)    : {embeddings.dtype}")
    print(f" -> Bellek Kullanımı     : {format_bytes(embeddings.nbytes)}")

    # Norm kontrolü (L2 normalize mi?)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f" -> L2 Norm Ortalaması   : {norms.mean():.4f} (min: {norms.min():.4f}, max: {norms.max():.4f})")

    # Örnek Doc ID'ler
    print("\n -> Örnek Doc ID'ler (ilk 5):")
    for i, d_id in enumerate(doc_ids[:5], 1):
        print(f"    {i}. {d_id}")

    # 3. Test Sorgusu Çalıştırma
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[3] Test Sorgusu Çalıştırılıyor (Cihaz: {device})...")
    print(f" -> SORGUMUZ: \"{args.query}\"")

    dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    query_emb = dense_model.encode(args.query, normalize_embeddings=True)

    # Cosine Similarity
    scores = np.dot(embeddings, query_emb)
    top_indices = np.argsort(-scores)[: max(args.top_k, 50)]

    # Cross-Encoder Reranking (eğer doküman metinleri varsa veya doc_id üzerinden)
    rerank_scores = None
    if args.rerank and documents is not None:
        print(" -> Cross-Encoder ile rerank ediliyor (BAAI/bge-reranker-large)...")
        reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)
        candidate_indices = top_indices[:50]
        pairs = [[args.query, str(documents[idx])] for idx in candidate_indices]
        ce_scores = reranker.predict(pairs, batch_size=32)
        rerank_order = np.argsort(-ce_scores)
        final_indices = [candidate_indices[i] for i in rerank_order[:args.top_k]]
        final_scores = [float(ce_scores[i]) for i in rerank_order[:args.top_k]]
        is_reranked = True
    else:
        final_indices = top_indices[:args.top_k]
        final_scores = [float(scores[i]) for i in final_indices]
        is_reranked = False

    # 4. Sonuçları Göster
    score_label = "Reranker Score" if is_reranked else "Cosine Similarity"
    print("\n" + "=" * 70)
    print(f"                        TEST SONUÇLARI (Top {len(final_indices)})")
    print("=" * 70)

    for rank, (idx, sc) in enumerate(zip(final_indices, final_scores), 1):
        doc_id = doc_ids[idx]
        raw_cos = float(scores[idx])
        print(f"Sıra {rank} | Doc ID: {doc_id} | {score_label}: {sc:.4f} (Cosine: {raw_cos:.4f})")
        if documents is not None:
            text_preview = str(documents[idx]).replace("\n", " ")
            if len(text_preview) > 220:
                text_preview = text_preview[:220] + "..."
            print(f"  Metin: {text_preview}")
        print("-" * 70)

    print("\n[OK] .npy test sonuçları başarıyla listelendi.")


if __name__ == "__main__":
    main()
