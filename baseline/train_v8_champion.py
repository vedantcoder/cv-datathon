"""Train Model v8 Champion: Ultra-Compact Fine-Tuned & Multi-Label Distilled VQA.

    python baseline/train_v8_champion.py --res 40 --top-ans 550 --tag ""
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.complexity import measure  # noqa: E402
from eval.normalize import normalize_answer  # noqa: E402
from eval.run_predict import load_model, predict_split  # noqa: E402
from eval.score import composite, load_score_cfg  # noqa: E402
from eval.vqa_eval import _official_norm, clean_whitespace, evaluate  # noqa: E402

MODEL_DIR = ROOT / "baseline" / "v8_champion"
sys.path.insert(0, str(MODEL_DIR))
import model as v8_model  # noqa: E402

PROC = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "data" / "features"
CKPT_DIR = ROOT / "data" / "checkpoints"

UNSAFE_FLIP_WORDS = {
    "left", "right", "clockwise", "counterclockwise", "time", "clock",
    "say", "says", "read", "word", "words", "letter", "letters",
    "number", "numbers", "side", "facing", "direction", "west", "east",
}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_word_vocab(questions, min_count=6):
    c = Counter(t for q in questions for t in v8_model.tokenize(q["question"]))
    return ["<pad>", "<unk>"] + sorted(w for w, n in c.items() if n >= min_count)


def loo_score(m: int) -> float:
    return (m * min(1.0, (m - 1) / 3) + (10 - m) * min(1.0, m / 3)) / 10 if m else 0.0


def load_or_cache_images(split, img_ids, target_size, cache_name):
    path = FEAT_DIR / cache_name
    if path.exists():
        d = torch.load(path)
        if d.get("image_ids") == img_ids and d["images"].shape[-1] == target_size:
            log(f"loaded cached {split} images: {tuple(d['images'].shape)}")
            return d["images"]

    t0 = time.time()
    split_dir = PROC / split / "images"

    def load_one(iid):
        with Image.open(split_dir / f"{iid}.jpg") as im:
            im = im.convert("RGB").resize((target_size, target_size), Image.Resampling.BILINEAR) if target_size != 224 else im.convert("RGB")
            return np.asarray(im)

    with ThreadPoolExecutor(max_workers=8) as ex:
        arrs = list(ex.map(load_one, img_ids))
    imgs = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).contiguous()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"image_ids": img_ids, "images": imgs}, path)
    log(f"cached {split} images {tuple(imgs.shape)} in {time.time() - t0:.1f}s")
    return imgs


def augment_batch_gpu(imgs_u8, target_res, can_flip, exact_scale=False, crop_min=0.91):
    B = imgs_u8.size(0)
    dev = imgs_u8.device
    x = imgs_u8.float() / 255.0

    if exact_scale:
        out = F.interpolate(x, size=(target_res, target_res), mode="bilinear", align_corners=False)
        do_flip = (torch.rand(B, device=dev) < 0.5) & can_flip
        if do_flip.any():
            out[do_flip] = out[do_flip].flip(-1)
        return out

    s = torch.empty(B, device=dev).uniform_(crop_min, 1.0)
    max_shift = (1.0 - s)
    tx = (torch.rand(B, device=dev) * 2.0 - 1.0) * max_shift
    ty = (torch.rand(B, device=dev) * 2.0 - 1.0) * max_shift

    do_flip = (torch.rand(B, device=dev) < 0.5) & can_flip
    flip = torch.where(do_flip, -1.0, 1.0)

    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = s * flip
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=(B, 3, target_res, target_res), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def precompute_val_score_matrix(va_q, va_gt, answers, dev):
    ans_clean = [clean_whitespace(a) for a in answers]
    ans_norm = [_official_norm(a) for a in ans_clean]
    ans_c2i, ans_n2i = {}, {}
    for idx_a, s in enumerate(ans_clean):
        ans_c2i.setdefault(s, []).append(idx_a)
    for idx_a, s in enumerate(ans_norm):
        ans_n2i.setdefault(s, []).append(idx_a)
    va_gt_by_id = {a["question_id"]: a for a in va_gt}
    mat = torch.zeros(len(va_q), len(answers), dtype=torch.float32)
    for qi, q in enumerate(va_q):
        gts = [clean_whitespace(a) for a in va_gt_by_id[q["question_id"]]["answers"]]
        if len(set(gts)) == 1:
            for ai in ans_c2i.get(gts[0], []):
                mat[qi, ai] = 1.0
        else:
            for g_norm, m in Counter(_official_norm(a) for a in gts).items():
                sc = (m * min(1.0, (m - 1) / 3.0) + (10 - m) * min(1.0, m / 3.0)) / 10.0
                for ai in ans_n2i.get(g_norm, []):
                    mat[qi, ai] = sc
    return mat.to(dev)


def n_tokens_for_res(res: int) -> int:
    sn = torchvision.models.shufflenet_v2_x0_5(weights=None)
    body = nn.Sequential(sn.conv1, sn.maxpool, sn.stage2, sn.stage3, sn.stage4)
    with torch.no_grad():
        y = body(torch.zeros(1, 3, res, res))
    return int(y.shape[2] * y.shape[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=40)
    ap.add_argument("--top-ans", type=int, default=550)
    ap.add_argument("--min-count", type=int, default=6)
    ap.add_argument("--v-dim", type=int, default=80)
    ap.add_argument("--emb-rank", type=int, default=16)
    ap.add_argument("--q-dim", type=int, default=64)
    ap.add_argument("--hid", type=int, default=104)
    ap.add_argument("--cls-rank", type=int, default=72)
    ap.add_argument("--drop", type=float, default=0.22)
    ap.add_argument("--epochs-warm", type=int, default=20)
    ap.add_argument("--epochs-ft", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--img-batch-size", type=int, default=128)
    ap.add_argument("--lr-head", type=float, default=4e-3)
    ap.add_argument("--lr-bb", type=float, default=1e-3)
    ap.add_argument("--kd-bce", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    seed_all(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sc_cfg = load_score_cfg(ROOT / "config.yaml")

    tr_q = json.load(open(PROC / "train/questions.json"))
    tr_a = {a["question_id"]: a for a in json.load(open(PROC / "train/annotations.json"))}
    va_q = json.load(open(PROC / "val/questions.json"))
    va_gt = json.load(open(PROC / "val/annotations.json"))

    n_tokens = n_tokens_for_res(args.res)
    arch_cfg = {
        "res": args.res,
        "n_tokens": n_tokens,
        "v_dim": args.v_dim,
        "emb_rank": args.emb_rank,
        "q_dim": args.q_dim,
        "hid": args.hid,
        "cls_rank": args.cls_rank,
        "drop": args.drop,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "arch.json").write_text(json.dumps(arch_cfg, indent=2) + "\n")

    words = build_word_vocab(tr_q, min_count=args.min_count)
    answers_list = (PROC / "answers.txt").read_text().splitlines()[:args.top_ans]
    (MODEL_DIR / "words.txt").write_text("\n".join(words) + "\n")
    (MODEL_DIR / "answers.txt").write_text("\n".join(answers_list) + "\n")

    model = v8_model.build_model()
    keys = [normalize_answer(a) for a in model.answers]
    key2idx = {k: i for i, k in enumerate(keys)}
    log(f"v8{args.tag} arch={arch_cfg}, {len(words)} words (min_count={args.min_count}), {len(model.answers)} answers")

    weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
    r_sn = torchvision.models.shufflenet_v2_x0_5(weights=weights)
    body = nn.Sequential(r_sn.conv1, r_sn.maxpool, r_sn.stage2, r_sn.stage3, r_sn.stage4)
    model.backbone.load_state_dict(body.state_dict())
    model = model.to(dev)

    tr_imgs = sorted({q["image_id"] for q in tr_q})
    va_imgs = sorted({q["image_id"] for q in va_q})
    tr_u8 = load_or_cache_images("train", tr_imgs, 128, "train_img128_u8.pt").to(dev)
    va_u8 = load_or_cache_images("val", va_imgs, 224, "val_img224_u8.pt").to(dev)

    with torch.no_grad():
        va_res_imgs = torch.empty(len(va_imgs), 3, args.res, args.res, dtype=torch.float32, device=dev)
        for s in range(0, len(va_imgs), 256):
            sl = slice(s, s + 256)
            va_res_imgs[sl] = F.interpolate(
                va_u8[sl].float() / 255.0, size=(args.res, args.res), mode="bilinear", align_corners=False
            )
    del va_u8

    teach_logits = None
    teach_path = FEAT_DIR / "train_teacher_logits.pt"
    if teach_path.exists() and args.kd_bce > 0:
        td = torch.load(teach_path)
        assert td["question_ids"] == [q["question_id"] for q in tr_q]
        teach_logits = td["logits"][:, :args.top_ans].to(dev)
        log(f"loaded teacher logits sliced to top-{args.top_ans}: {tuple(teach_logits.shape)}")

    im2row_tr = {iid: i for i, iid in enumerate(tr_imgs)}
    im2row_va = {iid: i for i, iid in enumerate(va_imgs)}
    tr_tok = model.encode_questions([q["question"] for q in tr_q], dev, max_len=v8_model.MAX_Q_LEN)
    va_tok = model.encode_questions([q["question"] for q in va_q], dev, max_len=v8_model.MAX_Q_LEN)
    tr_rows = torch.tensor([im2row_tr[q["image_id"]] for q in tr_q], device=dev)
    va_rows = torch.tensor([im2row_va[q["image_id"]] for q in va_q], device=dev)

    can_flip_tr = torch.tensor(
        [not bool(UNSAFE_FLIP_WORDS & set(v8_model.tokenize(q["question"]))) for q in tr_q],
        dtype=torch.bool, device=dev,
    )

    targets = []
    for q in tr_q:
        cnt = Counter(normalize_answer(x) for x in tr_a[q["question_id"]]["answers"])
        targets.append([(key2idx[k], loo_score(m)) for k, m in cnt.items() if k in key2idx])
    max_t = max(len(t) for t in targets)
    tgt_idx = torch.zeros(len(tr_q), max_t, dtype=torch.long)
    tgt_val = torch.zeros(len(tr_q), max_t, dtype=torch.float32)
    for i, t in enumerate(targets):
        for j, (ci, sc) in enumerate(t):
            tgt_idx[i, j] = ci
            tgt_val[i, j] = sc
    tgt_idx, tgt_val = tgt_idx.to(dev), tgt_val.to(dev)

    va_score_mat = precompute_val_score_matrix(va_q, va_gt, model.answers, dev)
    va_q_idx = torch.arange(len(va_q), device=dev)
    n_ans = len(model.answers)

    live_float = {
        **{f"backbone.{k}": p.data for k, p in model.backbone.named_parameters()},
        **{f"backbone.{k}": b.data for k, b in model.backbone.named_buffers() if b.is_floating_point()},
        **{f"head.{k}": p.data for k, p in model.head.named_parameters()},
        **{f"head.{k}": b.data for k, b in model.head.named_buffers() if b.is_floating_point()},
    }
    ema_sd = {k: v.clone() for k, v in live_float.items()}

    @torch.no_grad()
    def eval_val(use_ema=False):
        backup = None
        if use_ema:
            backup = {k: v.clone() for k, v in live_float.items()}
            for k, v in live_float.items():
                v.copy_(ema_sd[k])
        model.eval()
        v_feats = torch.empty(len(va_imgs), n_tokens, 192, device=dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(va_imgs), 512):
                sl = slice(s, s + 512)
                v_feats[sl] = model.extract_visual(va_res_imgs[sl]).float()
            pred_ids = []
            for s in range(0, len(va_q), 2048):
                sl = slice(s, s + 2048)
                lg = model.head(v_feats[va_rows[sl]], va_tok[sl])
                pred_ids.append(lg.argmax(-1))
        acc = va_score_mat[va_q_idx, torch.cat(pred_ids)].mean().item() * 100.0
        if backup is not None:
            for k, v in live_float.items():
                v.copy_(backup[k])
        return acc

    def compute_loss(logits, idx):
        tgt = torch.zeros(len(idx), n_ans, device=dev)
        tgt.scatter_add_(1, tgt_idx[idx], tgt_val[idx])
        loss_gt = F.binary_cross_entropy_with_logits(logits, tgt) * n_ans
        if teach_logits is None:
            return loss_gt
        t_prob = torch.sigmoid(teach_logits[idx].float())
        return (1.0 - args.kd_bce) * loss_gt + args.kd_bce * F.binary_cross_entropy_with_logits(logits, t_prob) * n_ans

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / f"v8_champion{args.tag}.pt"
    best_val, history, epoch_times = -1.0, [], []
    g = torch.Generator().manual_seed(args.seed)
    steps_per_ep = (len(tr_q) + args.batch_size - 1) // args.batch_size

    # Phase A: Frozen Backbone Warmup
    if args.epochs_warm > 0:
        log(f"--- Phase A: Frozen Backbone Warmup ({args.epochs_warm} epochs) ---")
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()
        with torch.no_grad():
            tr_warm_f = torch.empty(len(tr_imgs), n_tokens, 192, dtype=torch.float16, device=dev)
            for s in range(0, len(tr_imgs), 256):
                sl = slice(s, s + 256)
                x = F.interpolate(tr_u8[sl].float() / 255.0, size=(args.res, args.res), mode="bilinear", align_corners=False)
                tr_warm_f[sl] = model.extract_visual(x).half()

        opt_w = torch.optim.AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=1e-4)
        sched_w = torch.optim.lr_scheduler.OneCycleLR(
            opt_w, max_lr=args.lr_head, total_steps=args.epochs_warm * steps_per_ep, pct_start=0.15
        )
        for ep in range(1, args.epochs_warm + 1):
            t0 = time.time()
            model.head.train()
            perm = torch.randperm(len(tr_q), generator=g).to(dev)
            tot = 0.0
            for s in range(0, len(perm), args.batch_size):
                idx = perm[s:s + args.batch_size]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model.head(tr_warm_f[tr_rows[idx]].float(), tr_tok[idx])
                    loss = compute_loss(lg.float(), idx)
                opt_w.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
                opt_w.step()
                sched_w.step()
                tot += loss.item() * len(idx)
            for k, v in live_float.items():
                ema_sd[k].copy_(v)
            acc = eval_val(use_ema=False)
            dt = time.time() - t0
            log(f"warm ep {ep:2d} loss={tot/len(tr_q):.4f} val={acc:.2f}% ({dt:.1f}s)")
            if acc > best_val:
                best_val = acc
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)
        del tr_warm_f, opt_w, sched_w
        torch.cuda.empty_cache()

    # Phase B: Image-Grouped End-to-End Fine-Tuning
    img_qs = [[] for _ in range(len(tr_imgs))]
    for qi, q in enumerate(tr_q):
        img_qs[im2row_tr[q["image_id"]]].append(qi)
    img_can_flip = torch.tensor(
        [all(can_flip_tr[qi].item() for qi in qis) for qis in img_qs],
        dtype=torch.bool, device=dev,
    )

    img_steps_per_ep = (len(tr_imgs) + args.img_batch_size - 1) // args.img_batch_size
    log(f"--- Phase B: Image-Grouped Fine-Tuning ({args.epochs_ft} epochs, {img_steps_per_ep} steps/ep) ---")
    for p in model.backbone.parameters():
        p.requires_grad = True

    opt_ft = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_bb, "weight_decay": 1e-4},
        {"params": model.head.parameters(), "lr": args.lr_head * 0.6, "weight_decay": 1e-4},
    ])
    sched_ft = torch.optim.lr_scheduler.OneCycleLR(
        opt_ft, max_lr=[args.lr_bb, args.lr_head * 0.6], total_steps=args.epochs_ft * img_steps_per_ep, pct_start=0.10
    )

    for ep in range(1, args.epochs_ft + 1):
        t0 = time.time()
        model.train()
        exact_scale = ep > (args.epochs_ft - 6)
        img_perm = torch.randperm(len(tr_imgs), generator=g).tolist()
        tot = 0.0
        step_i = 0
        for s in range(0, len(img_perm), args.img_batch_size):
            b_imgs = img_perm[s:s + args.img_batch_size]
            b_imgs_t = torch.tensor(b_imgs, device=dev)
            local_img_rep, q_idx_list = [], []
            for li, im_idx in enumerate(b_imgs):
                qis = img_qs[im_idx]
                q_idx_list.extend(qis)
                local_img_rep.extend([li] * len(qis))
            idx = torch.tensor(q_idx_list, device=dev)
            rep = torch.tensor(local_img_rep, device=dev)

            imgs = augment_batch_gpu(
                tr_u8[b_imgs_t], args.res, img_can_flip[b_imgs_t], exact_scale=exact_scale, crop_min=0.91
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_feats = model.extract_visual(imgs)
                lg = model.head(v_feats[rep], tr_tok[idx])
                loss = compute_loss(lg.float(), idx)
            opt_ft.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_ft.step()
            sched_ft.step()
            step_i += 1
            if step_i % 2 == 0:
                with torch.no_grad():
                    for k, v in live_float.items():
                        ema_sd[k].mul_(0.975).add_(v, alpha=0.025)
            tot += loss.item() * len(idx)

        torch.cuda.synchronize()
        dt = time.time() - t0
        epoch_times.append(dt)

        acc_raw = eval_val(use_ema=False)
        acc_ema = eval_val(use_ema=True)
        use_ema = acc_ema >= acc_raw
        acc = acc_ema if use_ema else acc_raw
        src = "EMA" if use_ema else "raw"
        history.append({"epoch": ep, "loss": round(tot / len(tr_q), 4), "val_raw": round(acc_raw, 2),
                        "val_ema": round(acc_ema, 2), "val": round(acc, 2), "source": src, "sec": round(dt, 1)})
        log(f"ft   ep {ep:2d} loss={tot/len(tr_q):.4f} val={acc:.2f}% [{src}] (raw={acc_raw:.2f}, ema={acc_ema:.2f}) ({dt:.1f}s)")

        if acc > best_val:
            best_val = acc
            if use_ema:
                backup = {k: v.clone() for k, v in live_float.items()}
                for k, v in live_float.items():
                    v.copy_(ema_sd[k])
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)
                for k, v in live_float.items():
                    v.copy_(backup[k])
            else:
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)

    log(f"TRAINING COMPLETE. Best fast val: {best_val:.2f}%. Running official interface evaluation...")
    del model, tr_u8, va_res_imgs, teach_logits, va_score_mat
    torch.cuda.empty_cache()

    m = load_model(MODEL_DIR, best_path, dev)
    qs = json.load(open(PROC / "val/questions.json"))
    t0 = time.time()
    preds = predict_split(m, qs, PROC / "val/images", batch_size=256, num_workers=0, device=dev, log_every=0)
    (ROOT / "data/preds").mkdir(parents=True, exist_ok=True)
    (ROOT / f"data/preds/v8_champion_val{args.tag}.json").write_text(json.dumps(preds))
    r = evaluate(preds, va_gt)
    r.pop("_per_question"), r.pop("perQuestionType")
    val_res = r | {"inference_sec_gpu": round(time.time() - t0, 1)}
    log(f"official interface val: {val_res}")

    comp_metrics = measure(MODEL_DIR, best_path)
    log(f"complexity: {comp_metrics}")

    comp2 = composite(val_res["overall"], comp_metrics["params"], comp_metrics["flops"], sc_cfg)
    comp1_val = round(
        val_res["overall"]
        - math.log2(comp_metrics["params"] / sc_cfg["P_ref"])
        - math.log2(comp_metrics["flops"] / sc_cfg["F_ref"]),
        4,
    )
    log(f"COMPOSITE SCORE (w=2): {comp2}")
    log(f"COMPOSITE SCORE (w=1): {comp1_val}")

    out = {
        "arch": arch_cfg,
        "best_val_fast": round(best_val, 2),
        "history": history,
        "mean_epoch_sec": round(sum(epoch_times) / max(1, len(epoch_times)), 2),
        "args": vars(args),
        "val": val_res,
        "complexity": comp_metrics,
        "composite": comp2,
        "composite_w1": comp1_val,
    }
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / f"logs/v8_champion_results{args.tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
