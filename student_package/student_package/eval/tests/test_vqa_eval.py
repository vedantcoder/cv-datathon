import json
from pathlib import Path

import pytest

from eval.normalize import normalize_answer
from eval.vqa_eval import SubmissionError, evaluate, score_question

ROOT = Path(__file__).resolve().parents[2]
VAL_GT = ROOT / "data/processed/val/annotations.json"


def gt_item(qid, answers, at="other", qt="what"):
    return {"question_id": qid, "image_id": 1, "question_type": qt, "answer_type": at,
            "multiple_choice_answer": answers[0], "answers": answers}


# ---------- per-question metric (hand-computed leave-one-out values) ----------

@pytest.mark.parametrize("k, expected", [
    (0, 0.0),
    (1, (1 * 0 + 9 * (1 / 3)) / 10),          # 0.3
    (2, (2 * (1 / 3) + 8 * (2 / 3)) / 10),    # 0.6
    (3, (3 * (2 / 3) + 7 * 1) / 10),          # 0.9
    (4, 1.0),
    (10, 1.0),
])
def test_leave_one_out(k, expected):
    answers = ["cat"] * k + ["dog"] * (10 - k)
    if k == 10:
        answers = ["cat"] * 10
    assert score_question("cat", answers) == pytest.approx(expected)


def test_normalization_applied_when_gt_disagrees():
    gt = ["two"] * 5 + ["2"] * 4 + ["three"]
    assert score_question("2", gt) == pytest.approx(1.0)
    assert score_question("Two!", gt) == pytest.approx(1.0)
    assert score_question("the dog.", ["dog"] * 9 + ["cat"]) == pytest.approx(1.0)


def test_unanimous_gt_is_exact_match_official_quirk():
    # Official code skips normalization when all 10 GT answers are identical.
    assert score_question("yes", ["yes"] * 10) == 1.0
    assert score_question(" yes\n", ["yes"] * 10) == 1.0     # whitespace cleaning still applies
    assert score_question("Yes", ["yes"] * 10) == 0.0
    assert score_question("2", ["two"] * 10) == 0.0


def test_article_answer_a():
    # "a" normalizes to "" (article removal); vocab writes it as "a" and it must still score.
    assert normalize_answer("a") == ""
    assert score_question("a", ["a"] * 6 + ["b"] * 4) == pytest.approx(1.0)


# ---------- aggregation + submission validation ----------

GT = [gt_item(1, ["yes"] * 10, at="yes/no"), gt_item(2, ["2"] * 10, at="number"), gt_item(3, ["red"] * 10)]


def test_perfect_and_breakdown():
    preds = [{"question_id": 1, "answer": "yes"}, {"question_id": 2, "answer": "2"},
             {"question_id": 3, "answer": "blue"}]
    r = evaluate(preds, GT)
    assert r["overall"] == pytest.approx(66.67)
    assert r["perAnswerType"] == {"number": 100.0, "other": 0.0, "yes/no": 100.0}


def test_missing_scores_zero():
    r = evaluate([{"question_id": 1, "answer": "yes"}], GT)
    assert r["n_missing"] == 2 and r["overall"] == pytest.approx(33.33)


@pytest.mark.parametrize("preds, msg", [
    ([{"question_id": 1, "answer": "yes"}, {"question_id": 1, "answer": "no"}], "duplicate"),
    ([{"question_id": 99, "answer": "yes"}], "not in this split"),
    ([{"question_id": "1", "answer": "yes"}], "must be an int"),
    ([{"question_id": True, "answer": "yes"}], "must be an int"),
    ([{"question_id": 1, "answer": 3}], "must be a string"),
    ([{"question_id": 1}], "exactly the keys"),
    ([{"question_id": 1, "answer": "yes", "extra": 0}], "exactly the keys"),
    ({"1": "yes"}, "must be a JSON list"),
    (["yes"], "expected an object"),
])
def test_malformed_submissions(preds, msg):
    with pytest.raises(SubmissionError, match=msg):
        evaluate(preds, GT)


def test_invalid_json_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[{'question_id': 1,")
    with pytest.raises(SubmissionError, match="not valid JSON"):
        evaluate(p, GT)


# ---------- real-data sanity (skipped if the processed split is absent) ----------

@pytest.fixture(scope="module")
def val_gt():
    if not VAL_GT.exists():
        pytest.skip("processed val split not present")
    return json.load(open(VAL_GT))


def test_val_majority_answer_close_to_but_below_100(val_gt):
    preds = [{"question_id": a["question_id"], "answer": a["multiple_choice_answer"]} for a in val_gt]
    r = evaluate(preds, val_gt)
    assert 85.0 < r["overall"] < 100.0, r["overall"]


def test_val_constant_yes_prior(val_gt):
    preds = [{"question_id": a["question_id"], "answer": "yes"} for a in val_gt]
    r = evaluate(preds, val_gt)
    assert 20.0 < r["overall"] < 35.0, r["overall"]
    assert r["perAnswerType"]["number"] < 2.0 and r["perAnswerType"]["other"] < 3.0
