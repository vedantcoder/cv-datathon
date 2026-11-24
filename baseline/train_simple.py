"""Train the student-facing simple baseline on OUR train split only; select by val accuracy.

    python baseline/train_simple.py   # defaults = best val config of baseline/sweep_simple.sh (60 ep, bs 128, lr 2e-3, no clip, dropout 0.5)

Steps
 1. words.txt (train-question words with count >= 2, + <pad>, <unk>) and answers.txt -> baseline/simple/
 2. ImageNet ResNet-18 weights -> frozen backbone; grid features [49,512] for train/val cached (fp16)
 3. train the head with soft VQA targets (official leave-one-out score per answer), BCE loss
 4. pick the epoch with best val accuracy (official metric, on cached features)
 5. reload the best checkpoint through the fixed interface (build_model / load_state_dict / predict on JPEGs)
    and score val + test; run eval/complexity.py.  -> logs/simple_results.json
"""
import argparse
import os
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.complexity import measure  # noqa: E402
from eval.normalize import normalize_answer  # noqa: E402
from eval.run_predict import load_image, load_model, predict_split  # noqa: E402
from eval.vqa_eval import evaluate  # noqa: E402

MODEL_DIR = ROOT / "baseline" / "simple"
sys.path.insert(0, str(MODEL_DIR))
import model as simple  # noqa: E402

PROC = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "data" / "features"
CKPT_DIR = ROOT / "data" / "checkpoints"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_word_vocab(questions, min_count=2):
    c = Counter(t for q in questions for t in simple.tokenize(q["question"]))
    words = ["<pad>", "<unk>"] + sorted(w for w, n in c.items() if n >= min_count)
    return words


def loo_score(m: int) -> float:
    """Official accuracy of an answer that m of the 10 humans gave."""
    return (m * min(1.0, (m - 1) / 3) + (10 - m) * min(1.0, m / 3)) / 10 if m else 0.0


def soft_targets(anns, key2idx):
    out = []
    for a in anns:
        c = Counter(normalize_answer(x) for x in a["answers"])
        out.append([(key2idx[k], loo_score(m)) for k, m in c.items() if k in key2idx])
    return out


@torch.no_grad()
def extract_features(backbone, split, image_ids, device, bs=256):
    path = FEAT_DIR / f"{split}_r18_grid.pt"
    if path.exists():
        d = torch.load(path)
        if d["image_ids"] == image_ids:
            return d["feats"]
    t0 = time.time()
    feats = torch.empty(len(image_ids), 49, 512, dtype=torch.float16)

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(image_ids)
        def __getitem__(self, i): return load_image(PROC / split / "images" / f"{image_ids[i]}.jpg")

    dl = torch.utils.data.DataLoader(DS(), batch_size=bs, num_workers=min(8, os.cpu_count() or 1))
    i = 0
    for x in dl:
        f = backbone(x.to(device))
        feats[i: i + len(f)] = f.half().cpu()
        i += len(f)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"image_ids": image_ids, "feats": feats}, path)
    log(f"features {split}: {tuple(feats.shape)} in {time.time() - t0:.1f}s")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip (<=0 disables)")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--tag", default="", help="suffix for checkpoint/results (sweeps)")
    ap.add_argument("--sweep", action="store_true", help="train + val only; skip interface scoring of test")
    args = ap.parse_args()
    seed_all(args.seed)
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tr_q = json.load(open(PROC / "train/questions.json"))
    tr_a = {a["question_id"]: a for a in json.load(open(PROC / "train/annotations.json"))}
    va_q = json.load(open(PROC / "val/questions.json"))
    va_gt = json.load(open(PROC / "val/annotations.json"))

    # 1. vocabularies shipped next to model.py
    words = build_word_vocab(tr_q)
    (MODEL_DIR / "words.txt").write_text("\n".join(words) + "\n")
    shutil.copy(PROC / "answers.txt", MODEL_DIR / "answers.txt")
    model = simple.build_model()
    keys = [normalize_answer(a) for a in model.answers]
    key2idx = {k: i for i, k in enumerate(keys)}
    log(f"{len(words)} words, {len(model.answers)} answers")

    # 2. ImageNet backbone + cached features
    r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    body = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
    model.backbone.body.load_state_dict(body.state_dict())
    model = model.to(dev)
    tr_imgs = sorted({q["image_id"] for q in tr_q})
    va_imgs = sorted({q["image_id"] for q in va_q})
    t_feat = time.time()
    tr_f = extract_features(model.backbone, "train", tr_imgs, dev)
    va_f = extract_features(model.backbone, "val", va_imgs, dev)
    feat_time = time.time() - t_feat
    tr_f, va_f = tr_f.to(dev), va_f.to(dev)
    im2row_tr = {iid: i for i, iid in enumerate(tr_imgs)}
    im2row_va = {iid: i for i, iid in enumerate(va_imgs)}

    def encode(qs, im2row):
        tok, ln = model.encode_questions([q["question"] for q in qs], "cpu")
        rows = torch.tensor([im2row[q["image_id"]] for q in qs])
        return tok.to(dev), ln.to(dev), rows.to(dev)

    tr_tok, tr_len, tr_rows = encode(tr_q, im2row_tr)
    va_tok, va_len, va_rows = encode(va_q, im2row_va)
    targets = soft_targets([tr_a[q["question_id"]] for q in tr_q], key2idx)
    log(f"train {len(tr_q)} q, val {len(va_q)} q; {sum(1 for t in targets if t) / len(targets):.1%} "
        f"train q have an in-vocab answer")

    # 3. train head
    head = model.head
    for mod in head.classifier:
        if isinstance(mod, torch.nn.Dropout):
            mod.p = args.dropout
    opt = torch.optim.Adamax(head.parameters(), lr=args.lr)
    steps_per_epoch = (len(tr_q) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    g = torch.Generator().manual_seed(args.seed)
    n_ans = len(model.answers)

    @torch.no_grad()
    def val_acc():
        head.eval()
        preds = []
        for s in range(0, len(va_q), 1024):
            sl = slice(s, s + 1024)
            logits = head(va_f[va_rows[sl]].float(), va_tok[sl], va_len[sl])
            preds += [model.answers[i] for i in logits.argmax(-1).tolist()]
        head.train()
        return evaluate([{"question_id": q["question_id"], "answer": p} for q, p in zip(va_q, preds)], va_gt)

    best, history, epoch_times = -1.0, [], []
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / f"simple_baseline{args.tag}.pt"
    head.train()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(tr_q), generator=g)
        tot = 0.0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s: s + args.batch_size]
            tgt = torch.zeros(len(idx), n_ans)
            for j, qi in enumerate(idx.tolist()):
                for ci, sc in targets[qi]:
                    tgt[j, ci] = sc
            idx_d = idx.to(dev)
            logits = head(tr_f[tr_rows[idx_d]].float(), tr_tok[idx_d], tr_len[idx_d])
            loss = F.binary_cross_entropy_with_logits(logits, tgt.to(dev)) * n_ans
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        torch.cuda.synchronize()
        dt = time.time() - t0
        epoch_times.append(dt)
        r = val_acc()
        history.append({"epoch": ep, "loss": round(tot / len(tr_q), 4), "val": r["overall"],
                        "val_types": r["perAnswerType"], "sec": round(dt, 1)})
        log(f"epoch {ep:2d} loss {tot / len(tr_q):.4f} val {r['overall']:.2f} {r['perAnswerType']} ({dt:.1f}s)")
        if r["overall"] > best:
            best = r["overall"]
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)

    if args.sweep:
        log(f"SWEEP {args.tag} best val {best:.2f} args {vars(args)}")
        (ROOT / "logs").mkdir(parents=True, exist_ok=True)
        (ROOT / f"logs/simple_sweep{args.tag}.json").write_text(json.dumps({"best_val": best, "args": vars(args),
                                                                         "history": history}, indent=2))
        return

    # 5. reload through the fixed interface, score val + test from JPEGs
    del model, head, tr_f, va_f
    torch.cuda.empty_cache()
    m = load_model(MODEL_DIR, best_path, dev)
    out = {"best_val_cached": best, "history": history, "feature_extraction_sec": round(feat_time, 1),
           "mean_epoch_sec_4090": round(sum(epoch_times) / len(epoch_times), 2), "args": vars(args)}
    # test GT exists only on the instructor's machine; students score val only
    evals = [("val", PROC / "val/annotations.json"), ("test", ROOT / "private/test_gt.json")]
    for split, gt in [(s_, g_) for s_, g_ in evals if g_.exists()]:
        qs = json.load(open(PROC / f"{split}/questions.json"))
        t0 = time.time()
        preds = predict_split(m, qs, PROC / split / "images", batch_size=128, device=dev, log_every=0)
        (ROOT / "data/preds").mkdir(parents=True, exist_ok=True)
        (ROOT / f"data/preds/simple_{split}.json").write_text(json.dumps(preds))
        r = evaluate(preds, gt)
        r.pop("_per_question"), r.pop("perQuestionType")
        out[split] = r | {"inference_sec_gpu": round(time.time() - t0, 1)}
        log(f"interface {split}: {out[split]}")
    out["complexity"] = measure(MODEL_DIR, best_path)
    log(f"complexity: {out['complexity']}")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)   # the student package has no logs/ folder
    (ROOT / f"logs/simple_results{args.tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
