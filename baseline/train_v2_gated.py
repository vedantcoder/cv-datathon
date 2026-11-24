"""Train V2 Gated VQA with coordinate features, GLU multimodal fusion, and EMA weight averaging.

    python baseline/train_v2_gated.py --epochs 45 --batch-size 128 --lr 2e-3
"""
import argparse
import copy
import json
import os
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

MODEL_DIR = ROOT / "baseline" / "v2_gated"
sys.path.insert(0, str(MODEL_DIR))
import model as v2_model  # noqa: E402

PROC = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "data" / "features"
CKPT_DIR = ROOT / "data" / "checkpoints"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_word_vocab(questions, min_count=2):
    c = Counter(t for q in questions for t in v2_model.tokenize(q["question"]))
    words = ["<pad>", "<unk>"] + sorted(w for w, n in c.items() if n >= min_count)
    return words


def loo_score(m: int) -> float:
    return (m * min(1.0, (m - 1) / 3) + (10 - m) * min(1.0, m / 3)) / 10 if m else 0.0


def soft_targets(anns, key2idx):
    out = []
    for a in anns:
        c = Counter(normalize_answer(x) for x in a["answers"])
        out.append([(key2idx[k], loo_score(m)) for k, m in c.items() if k in key2idx])
    return out


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, split_dir):
        self.image_ids = image_ids
        self.split_dir = split_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        return load_image(self.split_dir / "images" / f"{self.image_ids[i]}.jpg")


@torch.no_grad()
def extract_features(backbone, split, image_ids, device, bs=256):
    path = FEAT_DIR / f"{split}_mbv3_small_grid.pt"
    if path.exists():
        d = torch.load(path)
        if d.get("image_ids") == image_ids:
            log(f"loaded cached features {split}: {tuple(d['feats'].shape)}")
            return d["feats"]
    t0 = time.time()
    feats = torch.empty(len(image_ids), 49, 576, dtype=torch.float16)

    ds = ImageDataset(image_ids, PROC / split)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0)
    i = 0
    for x in dl:
        f = backbone.features((x - backbone.mean) / backbone.std)
        f = f.flatten(2).transpose(1, 2)
        feats[i: i + len(f)] = f.half().cpu()
        i += len(f)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"image_ids": image_ids, "feats": feats}, path)
    log(f"features {split}: {tuple(feats.shape)} in {time.time() - t0:.1f}s")
    return feats


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="suffix for checkpoint/results")
    args = ap.parse_args()
    seed_all(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tr_q = json.load(open(PROC / "train/questions.json"))
    tr_a = {a["question_id"]: a for a in json.load(open(PROC / "train/annotations.json"))}
    va_q = json.load(open(PROC / "val/questions.json"))
    va_gt = json.load(open(PROC / "val/annotations.json"))

    # 1. Vocabularies
    words = build_word_vocab(tr_q)
    (MODEL_DIR / "words.txt").write_text("\n".join(words) + "\n")
    shutil.copy(PROC / "answers.txt", MODEL_DIR / "answers.txt")
    model = v2_model.build_model()
    keys = [normalize_answer(a) for a in model.answers]
    key2idx = {k: i for i, k in enumerate(keys)}
    log(f"{len(words)} words, {len(model.answers)} answers")

    # 2. Backbone + Cached Features
    log("Loading pretrained MobileNetV3-Small backbone weights...")
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    r = torchvision.models.mobilenet_v3_small(weights=weights)
    model.backbone.features.load_state_dict(r.features.state_dict())
    model = model.to(dev)

    tr_imgs = sorted({q["image_id"] for q in tr_q})
    va_imgs = sorted({q["image_id"] for q in va_q})
    t_feat = time.time()
    tr_f_raw = extract_features(model.backbone, "train", tr_imgs, dev)
    va_f_raw = extract_features(model.backbone, "val", va_imgs, dev)
    feat_time = time.time() - t_feat

    # Append 2D coordinate grid to visual features: [B, 49, 576] -> [B, 49, 578]
    coords = model.backbone.coords.to(dev)              # [1, 49, 2]
    tr_f = torch.cat([tr_f_raw.to(dev), coords.expand(len(tr_f_raw), -1, -1)], dim=-1)
    va_f = torch.cat([va_f_raw.to(dev), coords.expand(len(va_f_raw), -1, -1)], dim=-1)
    log(f"visual features with coordinates: {tuple(tr_f.shape)}")

    im2row_tr = {iid: i for i, iid in enumerate(tr_imgs)}
    im2row_va = {iid: i for i, iid in enumerate(va_imgs)}

    def encode(qs, im2row):
        tok, ln = model.encode_questions([q["question"] for q in qs], "cpu")
        rows = torch.tensor([im2row[q["image_id"]] for q in qs])
        return tok.to(dev), ln.to(dev), rows.to(dev)

    tr_tok, tr_len, tr_rows = encode(tr_q, im2row_tr)
    va_tok, va_len, va_rows = encode(va_q, im2row_va)
    targets = soft_targets([tr_a[q["question_id"]] for q in tr_q], key2idx)
    log(f"train {len(tr_q)} q, val {len(va_q)} q; {sum(1 for t in targets if t) / len(targets):.1%} train q with target")

    # 3. Train Head with EMA
    head = model.head
    ema = EMA(head, decay=0.999)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = (len(tr_q) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.15)
    g = torch.Generator().manual_seed(args.seed)
    n_ans = len(model.answers)

    @torch.no_grad()
    def evaluate_head(state_dict=None):
        backup = copy.deepcopy(head.state_dict()) if state_dict is not None else None
        if state_dict is not None:
            head.load_state_dict(state_dict)
        head.eval()
        preds = []
        for s in range(0, len(va_q), 1024):
            sl = slice(s, s + 1024)
            logits = head(va_f[va_rows[sl]].float(), va_tok[sl], va_len[sl])
            preds += [model.answers[i] for i in logits.argmax(-1).tolist()]
        if backup is not None:
            head.load_state_dict(backup)
        head.train()
        return evaluate([{"question_id": q["question_id"], "answer": p} for q, p in zip(va_q, preds)], va_gt)

    best, history, epoch_times = -1.0, [], []
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / f"v2_gated{args.tag}.pt"
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
            ema.update()
            tot += loss.item() * len(idx)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        epoch_times.append(dt)

        # Evaluate raw and EMA
        r_raw = evaluate_head()
        r_ema = evaluate_head(ema.state_dict())
        r = r_ema if r_ema["overall"] >= r_raw["overall"] else r_raw
        source = "EMA" if r_ema["overall"] >= r_raw["overall"] else "raw"

        history.append({"epoch": ep, "loss": round(tot / len(tr_q), 4), "val": r["overall"],
                        "val_types": r["perAnswerType"], "source": source, "sec": round(dt, 1)})
        log(f"epoch {ep:2d} loss {tot / len(tr_q):.4f} val {r['overall']:.2f} [{source}] "
            f"(raw={r_raw['overall']:.2f}, ema={r_ema['overall']:.2f}) {r['perAnswerType']} ({dt:.1f}s)")

        if r["overall"] > best:
            best = r["overall"]
            # Save the full model with the winning head state
            winning_head = ema.state_dict() if source == "EMA" else head.state_dict()
            full_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            for k, v in winning_head.items():
                full_sd[f"head.{k}"] = v.cpu()
            torch.save(full_sd, best_path)

    log(f"TRAINING COMPLETE. Best val: {best:.2f}. Running interface evaluation...")
    del model, head, tr_f, va_f, tr_f_raw, va_f_raw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m = load_model(MODEL_DIR, best_path, dev)
    out = {"best_val_cached": best, "history": history, "feature_extraction_sec": round(feat_time, 1),
           "mean_epoch_sec": round(sum(epoch_times) / len(epoch_times), 2), "args": vars(args)}

    qs = json.load(open(PROC / "val/questions.json"))
    t0 = time.time()
    preds = predict_split(m, qs, PROC / "val/images", batch_size=128, device=dev, log_every=0)
    (ROOT / "data/preds").mkdir(parents=True, exist_ok=True)
    (ROOT / f"data/preds/v2_gated_val{args.tag}.json").write_text(json.dumps(preds))
    r = evaluate(preds, va_gt)
    r.pop("_per_question"), r.pop("perQuestionType")
    out["val"] = r | {"inference_sec_gpu": round(time.time() - t0, 1)}
    log(f"interface val: {out['val']}")

    out["complexity"] = measure(MODEL_DIR, best_path)
    log(f"complexity: {out['complexity']}")

    # Calculate composite score
    from eval.score import composite, load_score_cfg
    sc_cfg = load_score_cfg(ROOT / "config.yaml")
    comp = composite(out["val"]["overall"], out["complexity"]["params"], out["complexity"]["flops"], sc_cfg)
    out["composite"] = comp
    log(f"COMPOSITE SCORE: {comp}")

    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / f"logs/v2_gated_results{args.tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
