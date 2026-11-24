"""VQA accuracy evaluator (official definition) with strict submission validation.

    python eval/vqa_eval.py --pred predictions.json --gt annotations.json [--out scores.json]

Scoring is a faithful port of GT-Vision-Lab/VQA vqaEval.py (see eval/normalize.py for the source hash):
  * every GT answer and the prediction are whitespace-cleaned (\\n, \\t -> space, strip);
  * ONLY if the 10 GT answers are not all identical, GT answers and the prediction are additionally
    normalized (punctuation, number words -> digits, articles, contractions); if all 10 agree the
    comparison is exact (case-sensitive) after whitespace cleaning — this mirrors the official code;
  * per question: mean over the 10 leave-one-out subsets of min(#matches among the other 9 / 3, 1);
  * accuracies are reported in percent (0–100), overall and per answer_type / question_type.

Submission rules: predictions is a JSON list of {"question_id": int, "answer": str}.
  * missing question_id  -> that question scores 0 (reported as n_missing);
  * duplicate or unknown question_id, wrong types, wrong structure -> SubmissionError (hard error).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.normalize import clean_whitespace, process_digit_article, process_punctuation  # noqa: E402


class SubmissionError(ValueError):
    """Raised for malformed submissions (hard error, with a clear message)."""


def _official_norm(text: str) -> str:
    return process_digit_article(process_punctuation(text))


def score_question(pred: str, gt_answers: list) -> float:
    """Official VQA accuracy of one prediction against its 10 GT answers, in [0, 1]."""
    gts = [clean_whitespace(a) for a in gt_answers]
    res = clean_whitespace(pred)
    if len(set(gts)) > 1:
        gts = [_official_norm(a) for a in gts]
        res = _official_norm(res)
    matches = [g == res for g in gts]
    m = sum(matches)
    # leave-one-out: drop answer i, count matches among the remaining 9
    return sum(min(1.0, (m - hit) / 3.0) for hit in matches) / len(gts)


def load_gt(gt) -> dict:
    anns = json.load(open(gt)) if isinstance(gt, (str, Path)) else gt
    if isinstance(anns, dict) and "annotations" in anns:
        raise ValueError("GT looks like an official VQA file; expected our list format (see CLAUDE.md)")
    by_id = {}
    for a in anns:
        qid = a["question_id"]
        if qid in by_id:
            raise ValueError(f"GT has duplicate question_id {qid}")
        if len(a["answers"]) != 10:
            raise ValueError(f"GT question {qid} has {len(a['answers'])} answers, expected 10")
        by_id[qid] = a
    return by_id


def validate_predictions(preds, gt_ids) -> dict:
    """Return {question_id: answer}. Raises SubmissionError on any structural problem."""
    if isinstance(preds, (str, Path)):
        try:
            preds = json.load(open(preds))
        except json.JSONDecodeError as e:
            raise SubmissionError(f"predictions file is not valid JSON: {e}") from None
    if not isinstance(preds, list):
        raise SubmissionError(f"predictions must be a JSON list of objects, got {type(preds).__name__}")
    out, dups, unknown = {}, [], []
    for i, p in enumerate(preds):
        if not isinstance(p, dict):
            raise SubmissionError(f"entry {i} is {type(p).__name__}, expected an object")
        if set(p) != {"question_id", "answer"}:
            raise SubmissionError(f"entry {i} must have exactly the keys question_id and answer, got {sorted(p)}")
        qid, ans = p["question_id"], p["answer"]
        if not isinstance(qid, int) or isinstance(qid, bool):
            raise SubmissionError(f"entry {i}: question_id must be an int, got {qid!r}")
        if not isinstance(ans, str):
            raise SubmissionError(f"entry {i} (question_id {qid}): answer must be a string, got {ans!r}")
        if qid in out:
            dups.append(qid)
        elif qid not in gt_ids:
            unknown.append(qid)
        out[qid] = ans
    if dups:
        raise SubmissionError(f"{len(dups)} duplicate question_id(s), e.g. {sorted(set(dups))[:5]}")
    if unknown:
        raise SubmissionError(f"{len(unknown)} question_id(s) not in this split, e.g. {sorted(unknown)[:5]}")
    return out


def evaluate(preds, gt, ndigits: int = 2) -> dict:
    gt_by_id = load_gt(gt)
    pred_by_id = validate_predictions(preds, gt_by_id.keys())
    per_q, by_at, by_qt = {}, defaultdict(list), defaultdict(list)
    for qid, a in gt_by_id.items():
        acc = score_question(pred_by_id[qid], a["answers"]) if qid in pred_by_id else 0.0
        per_q[qid] = acc
        by_at[a["answer_type"]].append(acc)
        by_qt[a["question_type"]].append(acc)

    def pct(xs):
        return round(100.0 * sum(xs) / len(xs), ndigits)

    return {
        "overall": pct(list(per_q.values())),
        "perAnswerType": {k: pct(v) for k, v in sorted(by_at.items())},
        "perQuestionType": {k: pct(v) for k, v in sorted(by_qt.items())},
        "n_questions": len(gt_by_id),
        "n_predicted": len(pred_by_id),
        "n_missing": len(gt_by_id) - len(pred_by_id),
        "_per_question": per_q,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True, help="predictions.json")
    ap.add_argument("--gt", required=True, help="annotations.json (our list format)")
    ap.add_argument("--out", help="write scores JSON here")
    args = ap.parse_args(argv)
    try:
        res = evaluate(args.pred, args.gt)
    except SubmissionError as e:
        print(f"SUBMISSION ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    res.pop("_per_question")
    if res["n_missing"]:
        print(f"WARNING: {res['n_missing']} of {res['n_questions']} questions have no prediction (scored 0)",
              file=sys.stderr)
    print(f"Overall accuracy: {res['overall']:.2f}")
    for k, v in res["perAnswerType"].items():
        print(f"  {k:8s}: {v:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
