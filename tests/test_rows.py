import pytest

from decide.rows import ask_row, jev_answer, jev_to_rows, parse_option, ranked


def test_parse_option_plain_and_described():
    assert parse_option("billing") == {"id": "billing", "description": "billing"}
    assert parse_option("yes=asks for a refund") == {"id": "yes", "description": "asks for a refund"}
    assert parse_option("x=") == {"id": "x", "description": "x"}


@pytest.mark.parametrize("state,options,message", [
    ("  ", ["a", "b"], "state is empty"),
    ("s", ["a"], "at least two"),
    ("s", ["a", "a=dup"], "unique"),
    ("s", ["=nothing", "b"], "empty option id"),
])
def test_ask_row_rejects_bad_input(state, options, message):
    with pytest.raises(ValueError, match=message):
        ask_row(state, "q?", options)


def test_ranked_orders_best_first():
    result = {"option_ids": ["a", "b", "c"], "probabilities": [0.2, 0.7, 0.1]}
    assert ranked(result) == [("b", 0.7), ("a", 0.2), ("c", 0.1)]


def test_jev_mapping_matches_jevbench_adapter():
    rows = jev_to_rows({"state": "S", "questions": {
        "ok": {"type": "noul", "instructions": "Allowed?", "criteria": {"true": "Yes it is", "false": "No"}},
        "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Payments", "tech": ""}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "High"]},
    }})
    (_, k1, noul), (_, k2, choice), (_, k3, score) = rows
    assert (k1, k2, k3) == ("noul", "choice", "score")
    assert noul["options"] == [{"id": "true", "description": "true: Yes it is"},
                               {"id": "false", "description": "false: No"}]
    assert choice["options"][1] == {"id": "tech", "description": "tech: tech"}
    assert [o["id"] for o in score["options"]] == ["0", "1"]


@pytest.mark.parametrize("request_body", [
    {"state": "S", "questions": {}},
    {"questions": {"a": {"type": "noul"}}},
    {"state": "S", "questions": {"a": {"type": "nope"}}},
    {"state": "S", "questions": {"a": {"type": "score", "criteria": ["only one"]}}},
])
def test_jev_mapping_rejects_bad_requests(request_body):
    with pytest.raises(ValueError):
        jev_to_rows(request_body)


def test_jev_answers():
    assert jev_answer("noul", {"option_ids": ["true", "false"], "probabilities": [0.9, 0.1]}) == \
        {"type": "noul", "noul": 0.9}
    choice = jev_answer("choice", {"option_ids": ["a", "b"], "probabilities": [0.3, 0.7]})
    assert choice["choice"] == "b"
    score = jev_answer("score", {"option_ids": ["0", "1", "2"], "probabilities": [0.0, 0.5, 0.5]})
    assert score["score"] == pytest.approx(1.5)
