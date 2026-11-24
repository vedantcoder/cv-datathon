"""Composite leaderboard score.

    Score = Acc - alpha * log2(Params / P_ref) - beta * log2(FLOPs / F_ref)

Acc is overall VQA accuracy in percent (0-100) from eval/vqa_eval.py. alpha, beta, P_ref, F_ref come from
config.yaml (never hardcoded).

    python eval/score.py --pred predictions.json --gt annotations.json --complexity complexity.json
    python eval/score.py --scores scores.json --complexity complexity.json     # reuse vqa_eval --out
"""
import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from eval.vqa_eval import evaluate  # noqa: E402

REQUIRED = {"params": int, "flops": int, "checkpoint_sha256": str, "torch_version": str}


def load_score_cfg(config_path) -> dict:
    sc = yaml.safe_load(open(config_path))["score"]
    missing = [k for k in ("alpha", "beta", "P_ref", "F_ref") if sc.get(k) is None]
    if missing:
        raise ValueError(f"config.yaml score.{missing} not set (TODO). Set them before computing composite scores.")
    if sc["P_ref"] <= 0 or sc["F_ref"] <= 0:
        raise ValueError("P_ref and F_ref must be positive")
    return sc


def validate_complexity(c: dict) -> dict:
    for k, t in REQUIRED.items():
        if k not in c or not isinstance(c[k], t) or isinstance(c[k], bool):
            raise ValueError(f"complexity.json: field {k!r} missing or not {t.__name__}")
    if c["params"] <= 0 or c["flops"] <= 0:
        raise ValueError("complexity.json: params and flops must be positive")
    return c


def composite(acc: float, params: int, flops: int, sc: dict) -> dict:
    p_pen = sc["alpha"] * math.log2(params / sc["P_ref"])
    f_pen = sc["beta"] * math.log2(flops / sc["F_ref"])
    return {"accuracy": acc, "params": params, "flops": flops,
            "param_penalty": round(p_pen, 4), "flop_penalty": round(f_pen, 4),
            "score": round(acc - p_pen - f_pen, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Composite datathon score")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred", help="predictions.json (requires --gt)")
    g.add_argument("--scores", help="scores JSON written by vqa_eval.py --out")
    ap.add_argument("--gt")
    ap.add_argument("--complexity", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    sc = load_score_cfg(args.config)
    if args.pred:
        if not args.gt:
            ap.error("--pred requires --gt")
        acc = evaluate(args.pred, args.gt)["overall"]
    else:
        acc = json.load(open(args.scores))["overall"]
    c = validate_complexity(json.load(open(args.complexity)))
    res = composite(acc, c["params"], c["flops"], sc) | {"checkpoint_sha256": c["checkpoint_sha256"],
                                                         "torch_version": c["torch_version"]}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main()
