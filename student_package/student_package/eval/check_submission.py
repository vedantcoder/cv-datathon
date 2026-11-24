"""Build submission.zip and check it with the SAME ingestion program Codabench runs, on val.

    python eval/check_submission.py --model-dir baseline/simple --ckpt data/checkpoints/simple_baseline.pt \
        --out submission.zip [--n-val 2000] [--threads 2]

1. zips <model-dir> (model.py + helper files; other .pt/.pth files are skipped) + <ckpt> as checkpoint.pt
2. unzips it into a fresh folder, as Codabench does
3. runs codabench/ingestion_program/ingestion.py on val questions, CPU only, with full error messages
4. scores the predictions on val and prints accuracy, params, FLOPs, composite score, estimated test runtime
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from eval.score import composite  # noqa: E402
from eval.vqa_eval import evaluate  # noqa: E402

N_TEST_QUESTIONS = 10405
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
SKIP_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".zip"}


def build_zip(model_dir: Path, ckpt: Path, out: Path) -> list:
    files = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(model_dir.rglob("*")):
            rel = f.relative_to(model_dir)
            if f.is_dir() or SKIP_DIRS & set(rel.parts) or f.suffix in SKIP_SUFFIXES:
                continue
            z.write(f, rel)
            files.append(str(rel))
        z.write(ckpt, "checkpoint.pt")
        files.append("checkpoint.pt")
    return files


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--val-dir", default=str(ROOT / "data/processed/val"))
    ap.add_argument("--n-val", type=int, default=2000, help="val questions to run (0 = all 10,388)")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (Codabench workers may have few)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    model_dir, ckpt, out = Path(args.model_dir).resolve(), Path(args.ckpt).resolve(), Path(args.out).resolve()
    val_dir = Path(args.val_dir).resolve()

    if not (model_dir / "model.py").exists():
        sys.exit(f"ERROR: {model_dir}/model.py not found")
    files = build_zip(model_dir, ckpt, out)
    mb = out.stat().st_size / 1e6
    print(f"[1/4] wrote {out} ({mb:.1f} MB): {files}")
    limit = cfg["codabench"]["max_checkpoint_mb"]
    if mb > limit:
        sys.exit(f"ERROR: zip is {mb:.0f} MB > {limit} MB limit")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sub, inp, outp = tmp / "submission", tmp / "input", tmp / "output"
        with zipfile.ZipFile(out) as z:
            z.extractall(sub)
        print(f"[2/4] unzipped into a fresh folder")

        qs = json.load(open(val_dir / "questions.json"))
        if args.n_val and args.n_val < len(qs):
            qs = random.Random(0).sample(qs, args.n_val)
        inp.mkdir()
        (inp / "questions.json").write_text(json.dumps(qs))
        os.symlink(val_dir / "images", inp / "images")

        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", VQA_DT_VERBOSE="1",
                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   VQA_DT_THREADS=str(args.threads), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        print(f"[3/4] running the Codabench ingestion program on {len(qs)} val questions "
              f"(CPU, {args.threads} threads) ...", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "codabench/ingestion_program/ingestion.py"),
                            str(inp), str(outp), str(ROOT), str(sub)], env=env, cwd=tmp)
        status = json.load(open(outp / "status.json"))["status"] if (outp / "status.json").exists() else None
        if r.returncode != 0 or status != 0:
            sys.exit(f"FAILED: ingestion status {status} (see the error above). Codabench would show this status.")

        info = json.load(open(outp / "run_info.json"))
        c = json.load(open(outp / "complexity.json"))
        preds = json.load(open(outp / "predictions.json"))
        gt = [a for a in json.load(open(val_dir / "annotations.json")) if a["question_id"] in
              {q["question_id"] for q in qs}]
        acc = evaluate(preds, gt)["overall"]
        res = composite(acc, c["params"], c["flops"], cfg["score"])
        est = info["predict_sec"] / len(qs) * N_TEST_QUESTIONS + info.get("load_sec", 0) + info.get("complexity_sec", 0)
        print(f"[4/4] val accuracy ({len(qs)} q): {acc:.2f}   params {c['params']:,}   FLOPs {c['flops']:,}")
        print(f"      composite score on this val sample: {res['score']:.2f} "
              f"(param penalty {res['param_penalty']:+.2f}, FLOP penalty {res['flop_penalty']:+.2f})")
        cb = cfg["codabench"]
        est_cb = cb["baseline_runtime_sec"] * c["flops"] / cfg["score"]["F_ref"]
        print(f"      test runtime on THIS machine ({args.threads} CPU threads): {est / 60:.1f} min")
        print(f"      estimated test runtime on Codabench: ~{est_cb / 60:.0f} min (baseline {cb['baseline_runtime_sec'] / 60:.1f} min "
              f"x your FLOPs / baseline FLOPs; Codabench's speed differs from this machine's, e.g. ~3x faster than Colab's CPU)")
        print(f"      Codabench time limit: {cb['execution_time_limit'] / 60:.0f} min. Leave a wide margin: a timeout scores nothing.")
        if est_cb > 0.5 * cb["execution_time_limit"]:
            print("      WARNING: the estimate is over half the time limit; this submission risks timing out.")
        print(f"OK  {out.name}  sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
