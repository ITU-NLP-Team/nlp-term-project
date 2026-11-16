import numpy as np

from chronofuse.metadata.profile import DocProfile
from chronofuse.metadata.year_index import YearIndex
from chronofuse.query.planner import normalise_plan, parse_plan_text
from chronofuse.retrieval.fusion import rrf
from chronofuse.retrieval.prf import (
    corpus_year_prior,
    downweight_years,
    is_year_token,
    rm3_weights,
    temporal_feedback,
)
from chronofuse.retrieval.raptor import TemporalRaptor, year_encoding


def _prof(i, years):
    return DocProfile(id=i, title=i, description="", years=years)


def test_rm3_interpolates_query_and_feedback():
    q = {"halv": 1.0, "bitcoin": 1.0}
    ranking = [("d1", 10.0), ("d2", 5.0)]
    toks = {"d1": ["halv", "reward", "reward", "2020"], "d2": ["block", "reward"]}
    w = rm3_weights(q, ranking, toks, fb_docs=2, fb_terms=3, alpha=0.5)
    assert w["reward"] > 0 and "halv" in w
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_year_tokens():
    assert is_year_token("2017") and not is_year_token("20170")
    w = downweight_years({"2017": 2.0, "fork": 1.0}, 0.5)
    assert w == {"2017": 1.0, "fork": 1.0}


def test_temporal_feedback_finds_dominant_window():
    profiles = {f"d{i}": _prof(f"d{i}", [2017, 2018]) for i in range(5)}
    profiles["x"] = _prof("x", [1999])
    prior = corpus_year_prior(profiles)
    win = temporal_feedback([(f"d{i}", 1.0) for i in range(5)] + [("x", 1.0)], profiles, prior)
    assert win is not None and win[0] >= 2017 and win[1] <= 2018


def test_year_index_passage_mask():
    pdoc = np.array([0, 0, 1, 2], dtype=np.int32)
    profiles = {"a": _prof("a", [2010]), "b": _prof("b", [2020]), "c": _prof("c", [])}
    yi = YearIndex(pdoc, ["a", "b", "c"], profiles)
    assert yi.passage_mask((2019, 2021), 0).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_temporal_raptor_returns_ranked_docs():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 8)).astype(np.float32)
    ids = [f"d{i}" for i in range(60)]
    profiles = {d: _prof(d, [2000 + i % 20]) for i, d in enumerate(ids)}
    tree = TemporalRaptor(target_size=5, levels=2, beta=0.5).build(ids, emb, profiles)
    res = tree.search(emb[3], k=10, window=(2003, 2003))
    assert len(res) == 10 and res[0][0] == "d3"
    assert year_encoding([]).sum() == 0 and abs(np.linalg.norm(year_encoding([2015])) - 1) < 1e-6


def test_plan_parser_recovers_truncated_json():
    text = '{"temporal_intent": "when", "year_window": [2014, 2015], "events": [{"year": 2014, '
    text += '"event": "Mt. Gox collapse"}], "keywords": ["Mt. Gox"], "answer_passage": "In 2014'
    plan = normalise_plan(parse_plan_text(text))
    assert plan["year_window"] == [2014, 2015]
    assert plan["events"][0]["event"] == "Mt. Gox collapse"
    assert plan["answer_passage"].startswith("In 2014")


def test_weighted_rrf_respects_weights():
    a = [("x", 1.0), ("y", 0.5)]
    b = [("y", 1.0), ("x", 0.5)]
    assert rrf({"a": a, "b": b}, {"a": 1.0, "b": 0.5})[0][0] == "x"
    assert rrf({"a": a, "b": b}, {"a": 0.5, "b": 1.0})[0][0] == "y"
