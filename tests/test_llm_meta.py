import numpy as np

from chronofuse.bench import DenseSearcher
from chronofuse.metadata.llm_meta import meta_terms, normalise_expansion, normalise_meta
from chronofuse.systems import period_score


def test_normalise_meta_repairs_years_and_dedupes():
    m = normalise_meta(
        '{"start_year": 2017, "end_year": 2015, "events": ["Mainnet", "mainnet", "Fork"],'
        ' "entities": ["IOTA Foundation"], "keywords": ["tangle"], "topic": "consensus"}'
    )
    assert (m["sy"], m["ey"]) == (2015, 2017)
    assert m["ev"] == ["Mainnet", "Fork"]
    assert "IOTA Foundation" in meta_terms(m)


def test_normalise_meta_recovers_truncated_output():
    m = normalise_meta('{"start_year": null, "end_year": 2019, "events": ["Chrysalis", "Tri')
    assert (m["sy"], m["ey"]) == (2019, 2019)
    assert m["ev"] == ["Chrysalis"]
    assert normalise_meta('{"start_year": 1000, "end_year": 2020}')["sy"] is None  # too wide


def test_normalise_expansion():
    e = normalise_expansion(
        '{"period": [2019, 2017], "neighbor_events": [{"event": "wallet hack", "year": 2020,'
        ' "relation": "after"}], "sub_queries": ["q1"], "expansion_terms": ["Hornet"]}'
    )
    assert e["period"] == [2017, 2019]
    assert e["neighbors"][0] == {"event": "wallet hack", "year": 2020, "relation": "after"}


def test_period_score():
    assert period_score((2010, 2012), (2011, 2015)) == 1.0
    assert period_score((2010, 2012), None) == 0.5
    assert period_score((2010, 2012), (2016, 2018)) < period_score((2010, 2012), (2013, 2014))


def test_dense_search_mask():
    emb = np.eye(3, dtype=np.float16)
    ds = DenseSearcher(["a#0", "b#0", "c#0"], emb)
    q = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    assert [d for d, _ in ds.search(q, k=3)] == ["a", "b", "c"]
    masked = ds.search(q, k=3, mask=np.array([False, True, True]))
    assert [d for d, _ in masked] == ["b", "c"]
