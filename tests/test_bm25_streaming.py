import numpy as np

from chronofuse.indexing.bm25_index import BM25Index

TEXTS = [
    "Bitcoin halving happened in 2020 and reduced the block reward.",
    "The IOTA tangle had slow transactions during the 2017 spam attack.",
    "Monero introduced RingCT in January 2017 to hide amounts.",
    "Cardano Shelley upgrade launched in 2020; staking became available.",
    "",
    "Transactions per block increased in 2023 because of ordinals inscriptions.",
] * 3
PIDS = [f"d{i}#0" for i in range(len(TEXTS))]
QUERIES = ["why did transactions per block jump in 2023", "monero ringct 2017", "staking cardano"]


def test_streaming_matches_in_memory_bm25():
    ref = BM25Index().build(PIDS, TEXTS)
    stream = BM25Index().build_streaming(zip(PIDS, TEXTS), batch_size=4)
    for q_ref, q_str in zip(ref.search(QUERIES, k=10), stream.search(QUERIES, k=10)):
        assert [d for d, _ in q_ref] == [d for d, _ in q_str]
        np.testing.assert_allclose([s for _, s in q_ref], [s for _, s in q_str], rtol=1e-5)


def test_streaming_roundtrip(tmp_path):
    idx = BM25Index().build_streaming(zip(PIDS, TEXTS), batch_size=5)
    idx.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.search(QUERIES[:1], k=3)[0][0][0] == idx.search(QUERIES[:1], k=3)[0][0][0]
