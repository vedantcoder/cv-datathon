"""Train Model v9: End-to-End Fine-Tuned MobileNetV3-Small with ConvNeXt-Small Distillation.

Strict compliance with competition rules & Gap <= 20 audit limit:
- Trained exclusively on data/processed/train (selected on data/processed/val).
- Uses standard torchvision ImageNet-pretrained MobileNetV3-Small + cached ConvNeXt-Small teacher logits.
- Preserves raw test-time scoring interface: build_model(), weights_only=True checkpoint, images in [0, 1].
- Reports full val (10,388 q) accuracy, w=2 score, w=1 score, and Gap = Score - Acc.
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline.v9_mbv3_ft.model import MAX_Q_LEN, PAD, UNK, V9MobileNetV3FT, tokenize
from eval.complexity import measure as measure_complexity
from eval.normalize import clean_whitespace, normalize_answer
from eval.run_predict import load_model, predict_split
from eval.score import composite, load_score_cfg
from eval.vqa_eval import evaluate as vqa_evaluate, score_question

DATA = ROOT / "data" / "processed"
FEAT_DIR = ROOT / "data" / "features"
CKPT_DIR = ROOT / "data" / "checkpoints"
LOGS_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "baseline" / "v9_mbv3_ft"

UNSAFE_FLIP_WORDS = {
    "left", "right", "clockwise", "counterclockwise",
    "west", "east", "north", "south",
    "clock", "time", "read", "say", "says", "word", "words",
    "letter", "letters", "sign", "signs", "written", "writing",
    "number", "numbers", "name", "facing", "way", "side",
    "hand", "lane", "direction", "reversed", "mirror",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_answers() -> list:
    return [a.strip() for a in (DATA / "answers.txt").read_text(encoding="utf-8").splitlines() if a.strip()]


def build_vocab(questions: list, min_count: int) -> list:
    wc = Counter()
    for q in questions:
        wc.update(tokenize(q["question"]))
    words = ["<pad>", "<unk>"]
    for w, c in sorted(wc.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= min_count:
            words.append(w)
    return words


def encode_questions_tensor(questions: list, w2i: dict, max_len: int = MAX_Q_LEN) -> torch.Tensor:
    out = torch.full((len(questions), max_len), PAD, dtype=torch.long)
    for i, q in enumerate(questions):
        arr = []
        for t in tokenize(q["question"]):
            if t in w2i:
                arr.append(w2i[t])
            elif len(t) > 3 and t.endswith("s") and t[:-1] in w2i:
                arr.append(w2i[t[:-1]])
            else:
                arr.append(UNK)
        arr = arr[:max_len] or [UNK]
        out[i, :len(arr)] = torch.tensor(arr, dtype=torch.long)
    return out


def compute_can_flip_flags(questions: list, anns: list) -> torch.Tensor:
    flags = torch.ones(len(questions), dtype=torch.bool)
    for i, (q, a) in enumerate(zip(questions, anns)):
        q_toks = set(tokenize(q["question"]))
        if q_toks & UNSAFE_FLIP_WORDS:
            flags[i] = False
            continue
        ans_norms = {normalize_answer(x) for x in a["answers"]}
        if ans_norms & {"left", "right"}:
            flags[i] = False
    return flags


def build_train_targets(anns: list, a2i: dict) -> torch.Tensor:
    y = torch.zeros((len(anns), len(a2i)), dtype=torch.float32)
    for i, ann in enumerate(anns):
        cnt = Counter(normalize_answer(a) for a in ann["answers"])
        for ans, c in cnt.items():
            idx = a2i.get(ans)
            if idx is not None:
                y[i, idx] = min(c / 3.0, 1.0)
    return y


def build_val_score_matrix(val_anns: list, answers: list) -> torch.Tensor:
    """Precompute exact [N_val, N_ans] VQA score matrix matching eval.vqa_eval.score_question."""
    norm_to_indices = defaultdict(list)
    clean_to_indices = defaultdict(list)
    for idx, ans in enumerate(answers):
        norm_to_indices[normalize_answer(ans)].append(idx)
        clean_to_indices[clean_whitespace(ans)].append(idx)

    mat = torch.zeros((len(val_anns), len(answers)), dtype=torch.float32)
    for qi, ann in enumerate(val_anns):
        gt_answers = ann["answers"]
        cand_indices = set()
        for g in gt_answers:
            cand_indices.update(norm_to_indices.get(normalize_answer(g), ()))
            cand_indices.update(clean_to_indices.get(clean_whitespace(g), ()))
        for idx in cand_indices:
            s = score_question(answers[idx], gt_answers)
            if s > 0.0:
                mat[qi, idx] = s
    return mat


def load_or_cache_train_images(image_ids: list, res: int = 192) -> torch.Tensor:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEAT_DIR / f"train_img{res}_u8.pt"
    if cache_path.exists():
        print(f"Loading cached {res}x{res} train images from {cache_path}...", flush=True)
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        if blob["image_ids"] == image_ids and tuple(blob["images"].shape[-2:]) == (res, res):
            return blob["images"]

    print(f"Caching {len(image_ids)} train images at {res}x{res} uint8...", flush=True)
    t0 = time.time()
    img_dir = DATA / "train" / "images"

    def _read_one(iid: int) -> np.ndarray:
        p = img_dir / f"{iid}.jpg"
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.size != (res, res):
                im = im.resize((res, res), Image.Resampling.BILINEAR)
            return np.asarray(im, dtype=np.uint8)

    out = torch.empty((len(image_ids), 3, res, res), dtype=torch.uint8)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, arr in enumerate(ex.map(_read_one, image_ids, chunksize=64)):
            out[i].copy_(torch.from_numpy(arr).permute(2, 0, 1))
            if (i + 1) % 5000 == 0:
                print(f"  cached {i + 1}/{len(image_ids)} images ({time.time() - t0:.1f}s)", flush=True)

    torch.save({"image_ids": image_ids, "images": out}, cache_path)
    print(f"Saved {cache_path} ({out.numel() / 1e9:.2f} GB) in {time.time() - t0:.1f}s", flush=True)
    return out


def prepare_val_images_res(res: int, val_img_ids: list, dev: torch.device) -> torch.Tensor:
    """Load val_img224_u8.pt and resize cleanly to [N_val, 3, res, res] float16 matching extract_visual."""
    cache_224 = FEAT_DIR / "val_img224_u8.pt"
    if cache_224.exists():
        blob = torch.load(cache_224, map_location="cpu", weights_only=True)
        assert blob["image_ids"] == val_img_ids
        imgs224 = blob["images"]
    else:
        img_dir = DATA / "val" / "images"
        imgs224 = torch.empty((len(val_img_ids), 3, 224, 224), dtype=torch.uint8)
        for i, iid in enumerate(val_img_ids):
            with Image.open(img_dir / f"{iid}.jpg") as im:
                imgs224[i].copy_(torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.uint8)).permute(2, 0, 1))

    out = torch.empty((len(val_img_ids), 3, res, res), dtype=torch.float16)
    with torch.no_grad():
        for s in range(0, len(val_img_ids), 256):
            xb = imgs224[s:s + 256].to(dev, dtype=torch.float32) / 255.0
            if xb.shape[-1] != res or xb.shape[-2] != res:
                xb = F.interpolate(xb, size=(res, res), mode="bilinear", align_corners=False)
            out[s:s + 256].copy_(xb.half().cpu())
    del imgs224
    return out


def build_image_csr(q_img_ids: list, img_id_to_idx: dict, num_imgs: int):
    buckets = [[] for _ in range(num_imgs)]
    for qi, iid in enumerate(q_img_ids):
        buckets[img_id_to_idx[iid]].append(qi)
    ptr = [0]
    flat = []
    for b in buckets:
        flat.extend(b)
        ptr.append(len(flat))
    return buckets, torch.tensor(ptr, dtype=torch.long), torch.tensor(flat, dtype=torch.long)


def gather_image_batch_questions(img_batch: torch.Tensor, ptr: torch.Tensor, flat: torch.Tensor):
    """Given tensor [B_img] of image indices, return (q_indices [Q], local_img_idx [Q])."""
    q_chunks = []
    l_chunks = []
    for li, ii in enumerate(img_batch.tolist()):
        s, e = int(ptr[ii]), int(ptr[ii + 1])
        if e > s:
            q_chunks.append(flat[s:e])
            l_chunks.append(torch.full((e - s,), li, dtype=torch.long))
    return torch.cat(q_chunks), torch.cat(l_chunks)


def augment_images_gpu(
    imgs_u8: torch.Tensor,
    can_flip_imgs: torch.Tensor,
    out_res: int,
    crop_min: float,
    p_flip: float,
    exact_scale: bool,
) -> torch.Tensor:
    """Color-safe geometric crop + directional-safe horizontal flip on GPU."""
    x = imgs_u8.float() / 255.0
    b = x.size(0)
    dev = x.device
    do_flip = (torch.rand(b, device=dev) < p_flip) & can_flip_imgs if p_flip > 0.0 else torch.zeros(b, dtype=torch.bool, device=dev)

    if exact_scale:
        if x.shape[-1] != out_res or x.shape[-2] != out_res:
            x = F.interpolate(x, size=(out_res, out_res), mode="bilinear", align_corners=False)
        if do_flip.any():
            x[do_flip] = x[do_flip].flip(-1)
        return x

    scale = torch.empty(b, device=dev).uniform_(crop_min, 1.0)
    max_shift = 1.0 - scale
    tx = (torch.rand(b, device=dev) * 2.0 - 1.0) * max_shift
    ty = (torch.rand(b, device=dev) * 2.0 - 1.0) * max_shift
    sx = torch.where(do_flip, -scale, scale)

    theta = torch.zeros((b, 2, 3), device=dev, dtype=torch.float32)
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (b, 3, out_res, out_res), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class LiveFloatEMA:
    """Tracks EMA over all floating-point parameters AND floating-point buffers (BN stats)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        self.steps = 0

    def reset_from_model(self, model: nn.Module, decay: float = None):
        if decay is not None:
            self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.steps += 1
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        one_minus_d = 1.0 - d
        msd = model.state_dict()
        for k, sv in self.shadow.items():
            mv = msd[k]
            if mv.is_floating_point():
                sv.mul_(d).add_(mv.detach(), alpha=one_minus_d)

    def apply_to(self, model: nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def state_dict_full(self, model: nn.Module) -> dict:
        out = {}
        for k, v in model.state_dict().items():
            out[k] = self.shadow[k].detach().cpu().clone() if k in self.shadow else v.detach().cpu().clone()
        return out


@torch.no_grad()
def eval_val_fast(
    model: V9MobileNetV3FT,
    va_imgs_res_f16: torch.Tensor,
    va_tokens: torch.Tensor,
    va_q_img_idx: torch.Tensor,
    val_score_mat: torch.Tensor,
    dev: torch.device,
    img_bs: int = 256,
) -> float:
    """Exact VQA validation using pre-resized [N_val, 3, res, res] images on GPU (<0.3s)."""
    model.eval()
    n_imgs = va_imgs_res_f16.size(0)
    grid_tokens = model.head.coords.size(1)
    v_all = torch.empty((n_imgs, grid_tokens, 576), device=dev, dtype=torch.float32)

    for s in range(0, n_imgs, img_bs):
        xb = va_imgs_res_f16[s:s + img_bs].to(dev, dtype=torch.float32)
        v_all[s:s + img_bs] = model.extract_visual(xb)

    n_q = va_tokens.size(0)
    preds = torch.empty(n_q, dtype=torch.long)
    for s in range(0, n_q, 1024):
        e = min(n_q, s + 1024)
        logits = model.head(v_all[va_q_img_idx[s:e].to(dev)], va_tokens[s:e].to(dev))
        preds[s:e] = logits.argmax(-1).cpu()

    acc = val_score_mat[torch.arange(n_q), preds].mean().item() * 100.0
    return acc


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def param_groups_wd(named_params, lr: float, wd: float):
    decay_p, no_decay_p = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not n.endswith("bias"):
            decay_p.append(p)
        else:
            no_decay_p.append(p)
    groups = []
    if decay_p:
        groups.append({"params": decay_p, "lr": lr, "base_lr": lr, "weight_decay": wd})
    if no_decay_p:
        groups.append({"params": no_decay_p, "lr": lr, "base_lr": lr, "weight_decay": 0.0})
    return groups


def eval_both_live_and_ema(
    model: V9MobileNetV3FT,
    ema: LiveFloatEMA,
    va_imgs_res_f16: torch.Tensor,
    va_tokens: torch.Tensor,
    va_q_img_idx: torch.Tensor,
    val_score_mat: torch.Tensor,
    dev: torch.device,
):
    live_acc = eval_val_fast(model, va_imgs_res_f16, va_tokens, va_q_img_idx, val_score_mat, dev)
    backup = ema.apply_to(model)
    ema_acc = eval_val_fast(model, va_imgs_res_f16, va_tokens, va_q_img_idx, val_score_mat, dev)
    ema_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(backup, strict=False)
    live_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return live_acc, ema_acc, live_state, ema_state


def main():
    parser = argparse.ArgumentParser(description="Train Model v9 (End-to-End MobileNetV3-Small + KD)")
    parser.add_argument("--res", type=int, default=160)
    parser.add_argument("--grid-hw", type=int, default=5)
    parser.add_argument("--emb-rank", type=int, default=64)
    parser.add_argument("--q-dim", type=int, default=192)
    parser.add_argument("--att-dim", type=int, default=192)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--cls-hid", type=int, default=384)
    parser.add_argument("--drop", type=float, default=0.3)
    parser.add_argument("--min-word-count", type=int, default=3)

    parser.add_argument("--img-batch-size", type=int, default=64)
    parser.add_argument("--phase-a-epochs", type=int, default=24)
    parser.add_argument("--phase-b-epochs", type=int, default=26)
    parser.add_argument("--lr-a-head", type=float, default=2.5e-3)
    parser.add_argument("--lr-b-bb", type=float, default=4.0e-4)
    parser.add_argument("--lr-b-head", type=float, default=1.0e-3)
    parser.add_argument("--wd", type=float, default=1.5e-4)
    parser.add_argument("--kd-weight", type=float, default=0.30)
    parser.add_argument("--crop-min", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Model v9 on {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}) ===", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    va_q = json.loads((DATA / "val" / "questions.json").read_text(encoding="utf-8"))
    va_a = json.loads((DATA / "val" / "annotations.json").read_text(encoding="utf-8"))

    if args.eval_only:
        ckpt_path = CKPT_DIR / "v9_mbv3_ft.pt"
        eval_model = load_model(MODEL_DIR, ckpt_path, device="cpu")
        preds = predict_split(eval_model, va_q, DATA / "val" / "images", batch_size=256, num_workers=0, device="cpu", log_every=0)
        official_val = vqa_evaluate(preds, va_a)
        official_val_clean = {k: v for k, v in official_val.items() if k != "_per_question"}
        acc = official_val["overall"]
        comp = measure_complexity(MODEL_DIR, ckpt_path, config_path=ROOT / "config.yaml")
        sc_w2 = load_score_cfg(ROOT / "config.yaml")
        res_w2 = composite(acc, comp["params"], comp["flops"], sc_w2)
        sc_w1 = dict(sc_w2, alpha=1.0, beta=1.0)
        res_w1 = composite(acc, comp["params"], comp["flops"], sc_w1)
        gap = round(res_w2["score"] - acc, 2)
        arch_saved = json.loads((MODEL_DIR / "arch.json").read_text(encoding="utf-8"))
        words_saved = (MODEL_DIR / "words.txt").read_text(encoding="utf-8").splitlines()
        summary = {
            "model": "v9_mbv3_ft",
            "selected": "saved_checkpoint",
            "arch": arch_saved,
            "vocab_size": len(words_saved),
            "val": official_val_clean,
            "params": comp["params"],
            "flops": comp["flops"],
            "score_w2": res_w2,
            "score_w1": res_w1,
            "gap": gap,
            "audit_safe_under_20": gap <= 20.0,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": comp["checkpoint_sha256"],
        }
        log_path = LOGS_DIR / "v9_mbv3_ft_results.json"
        log_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("\n============================================================")
        print(f"OFFICIAL VAL OVERALL : {acc:.2f}%")
        print(f"BY TYPE              : {official_val['perAnswerType']}")
        print(f"PARAMS / FLOPS       : {comp['params']:,} / {comp['flops']:,}")
        print(f"SCORE (w=2, official): {res_w2['score']:.2f}  | Gap = {gap:.2f} (Safe <= 20: {gap <= 20.0})")
        print(f"SCORE (w=1, fallback): {res_w1['score']:.2f}")
        print(f"Results written to   : {log_path}")
        print("============================================================", flush=True)
        return

    tr_q = json.loads((DATA / "train" / "questions.json").read_text(encoding="utf-8"))
    tr_a = json.loads((DATA / "train" / "annotations.json").read_text(encoding="utf-8"))

    answers = load_answers()
    a2i = {a: i for i, a in enumerate(answers)}
    words = build_vocab(tr_q, args.min_word_count)
    w2i = {w: i for i, w in enumerate(words)}

    arch_cfg = {
        "res": args.res,
        "grid_hw": args.grid_hw,
        "emb_rank": args.emb_rank,
        "q_dim": args.q_dim,
        "att_dim": args.att_dim,
        "hid": args.hid,
        "cls_hid": args.cls_hid,
        "drop": args.drop,
    }
    (MODEL_DIR / "words.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    (MODEL_DIR / "answers.txt").write_text("\n".join(answers) + "\n", encoding="utf-8")
    (MODEL_DIR / "arch.json").write_text(json.dumps(arch_cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Vocab: {len(words)} words (min_count={args.min_word_count}), {len(answers)} answers | arch={arch_cfg}", flush=True)

    tr_tokens = encode_questions_tensor(tr_q, w2i, MAX_Q_LEN)
    va_tokens = encode_questions_tensor(va_q, w2i, MAX_Q_LEN)
    tr_y_gt = build_train_targets(tr_a, a2i)
    val_score_mat = build_val_score_matrix(va_a, answers)

    tl_blob = torch.load(FEAT_DIR / "train_teacher_logits.pt", map_location="cpu", weights_only=True)
    assert tl_blob["question_ids"] == [x["question_id"] for x in tr_q], "Teacher question_ids mismatch!"
    tr_teacher_logits = tl_blob["logits"]
    del tl_blob

    tr_img_ids = sorted({q["image_id"] for q in tr_q})
    va_img_ids = sorted({q["image_id"] for q in va_q})
    tr_img_id_to_idx = {iid: i for i, iid in enumerate(tr_img_ids)}
    va_img_id_to_idx = {iid: i for i, iid in enumerate(va_img_ids)}

    tr_buckets, tr_ptr, tr_flat = build_image_csr([q["image_id"] for q in tr_q], tr_img_id_to_idx, len(tr_img_ids))
    va_q_img_idx = torch.tensor([va_img_id_to_idx[q["image_id"]] for q in va_q], dtype=torch.long)

    q_can_flip = compute_can_flip_flags(tr_q, tr_a)
    img_can_flip = torch.tensor(
        [all(bool(q_can_flip[qi]) for qi in b) for b in tr_buckets],
        dtype=torch.bool,
    )
    print(f"Safe flip: {int(q_can_flip.sum())}/{len(tr_q)} questions, {int(img_can_flip.sum())}/{len(tr_img_ids)} images", flush=True)

    # Load / cache images
    tr_imgs_u8 = load_or_cache_train_images(tr_img_ids, res=192)
    va_imgs_res_f16 = prepare_val_images_res(args.res, va_img_ids, dev)

    # Build model & load torchvision IMAGENET1K_V1 backbone weights
    model = V9MobileNetV3FT(words, answers, cfg=arch_cfg).to(dev)
    bb_weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    bb_pretrained = torchvision.models.mobilenet_v3_small(weights=bb_weights).features.state_dict()
    model.backbone.load_state_dict(bb_pretrained, strict=True)
    del bb_pretrained

    # =========================================================================
    # PHASE A: Warmup V9GatedHead on Frozen Pretrained Features (Normal + Flipped)
    # =========================================================================
    print("\n--- Phase A: Frozen MobileNetV3-Small Feature Extraction (Orig + Safe Flip) ---", flush=True)
    model.eval()
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    n_tr_imgs = len(tr_img_ids)
    n_grid = args.grid_hw * args.grid_hw
    tr_feat_orig = torch.empty((n_tr_imgs, n_grid, 576), dtype=torch.float16)
    tr_feat_flip = torch.empty((n_tr_imgs, n_grid, 576), dtype=torch.float16)
    with torch.no_grad():
        for s in range(0, n_tr_imgs, 256):
            xb = tr_imgs_u8[s:s + 256].to(dev, dtype=torch.float32) / 255.0
            tr_feat_orig[s:s + 256] = model.extract_visual(xb).half().cpu()
            tr_feat_flip[s:s + 256] = model.extract_visual(xb.flip(-1)).half().cpu()

    print(f"Extracted features {tuple(tr_feat_orig.shape)} in Phase A.", flush=True)

    opt_a = torch.optim.AdamW(
        param_groups_wd(model.head.named_parameters(), lr=args.lr_a_head, wd=args.wd),
        betas=(0.9, 0.999),
    )
    ema_a = LiveFloatEMA(model, decay=0.992)
    steps_per_ep = math.ceil(n_tr_imgs / args.img_batch_size)
    total_steps_a = args.phase_a_epochs * steps_per_ep
    warmup_a = 3 * steps_per_ep

    best_a_acc = -1.0
    best_a_tag = "none"
    best_a_state = None
    history_a = []
    step_a = 0
    t_start_a = time.time()

    img_can_flip_cpu = img_can_flip.cpu()
    for ep in range(1, args.phase_a_epochs + 1):
        t_ep = time.time()
        model.train()
        model.backbone.eval()
        perm_imgs = torch.randperm(n_tr_imgs)
        tot_loss = 0.0
        tot_q = 0

        for b_start in range(0, n_tr_imgs, args.img_batch_size):
            img_batch = perm_imgs[b_start:b_start + args.img_batch_size]
            use_flip = (torch.rand(img_batch.size(0)) < 0.5) & img_can_flip_cpu[img_batch]

            vf_batch = tr_feat_orig[img_batch].clone()
            if use_flip.any():
                vf_batch[use_flip] = tr_feat_flip[img_batch[use_flip]]
            vf_dev = vf_batch.to(dev, dtype=torch.float32)

            q_idx, local_idx = gather_image_batch_questions(img_batch, tr_ptr, tr_flat)
            tq = tr_tokens[q_idx].to(dev)
            y_gt = tr_y_gt[q_idx].to(dev)
            y_kd = torch.sigmoid(tr_teacher_logits[q_idx].to(dev, dtype=torch.float32))

            cur_lr = cosine_warmup_lr(step_a, total_steps_a, warmup_a, args.lr_a_head, 2e-4)
            for pg in opt_a.param_groups:
                pg["lr"] = cur_lr

            logits = model.head(vf_dev[local_idx.to(dev)], tq)
            loss_gt = F.binary_cross_entropy_with_logits(logits, y_gt, reduction="none").sum(-1).mean()
            loss_kd = F.binary_cross_entropy_with_logits(logits, y_kd, reduction="none").sum(-1).mean()
            loss = (1.0 - args.kd_weight) * loss_gt + args.kd_weight * loss_kd

            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 2.0)
            opt_a.step()
            ema_a.update(model)

            tot_loss += float(loss.item()) * q_idx.numel()
            tot_q += q_idx.numel()
            step_a += 1

        if ep % 2 == 0 or ep >= args.phase_a_epochs - 4:
            live_acc, ema_acc, live_st, ema_st = eval_both_live_and_ema(
                model, ema_a, va_imgs_res_f16, va_tokens, va_q_img_idx, val_score_mat, dev
            )
            tag_ep, cur_best_st = ("ema", ema_st) if ema_acc >= live_acc else ("live", live_st)
            best_ep = max(live_acc, ema_acc)
            if best_ep > best_a_acc:
                best_a_acc = best_ep
                best_a_tag = f"ep{ep}_{tag_ep}"
                best_a_state = cur_best_st
            history_a.append({"epoch": ep, "loss": round(tot_loss / tot_q, 4), "live_val": round(live_acc, 2), "ema_val": round(ema_acc, 2)})
            print(f"[Phase A ep {ep:02d}/{args.phase_a_epochs}] loss={tot_loss / tot_q:.4f} | live={live_acc:.2f}% ema={ema_acc:.2f}% | best={best_a_acc:.2f}% ({best_a_tag}) | {time.time() - t_ep:.1f}s", flush=True)
        else:
            print(f"[Phase A ep {ep:02d}/{args.phase_a_epochs}] loss={tot_loss / tot_q:.4f} | {time.time() - t_ep:.1f}s", flush=True)

    del tr_feat_orig, tr_feat_flip
    torch.cuda.empty_cache()
    phase_a_seconds = time.time() - t_start_a
    print(f"Phase A finished in {phase_a_seconds:.1f}s | Best Phase A Val: {best_a_acc:.2f}% ({best_a_tag})", flush=True)

    # Load best Phase A weights and unfreeze backbone for Phase B
    model.load_state_dict(best_a_state, strict=True)
    del best_a_state
    for p in model.backbone.parameters():
        p.requires_grad_(True)

    # =========================================================================
    # PHASE B: End-to-End MobileNetV3-Small Fine-Tuning (Image-Grouped bs=64)
    # =========================================================================
    print("\n--- Phase B: End-to-End Fine-Tuning (Color-Safe Crop + Flip -> Exact Scale Tail) ---", flush=True)
    tr_imgs_gpu = tr_imgs_u8.to(dev)
    can_flip_gpu = img_can_flip.to(dev)
    del tr_imgs_u8

    pg_b = (
        param_groups_wd(model.backbone.named_parameters(), lr=args.lr_b_bb, wd=args.wd)
        + param_groups_wd(model.head.named_parameters(), lr=args.lr_b_head, wd=args.wd)
    )
    opt_b = torch.optim.AdamW(pg_b, betas=(0.9, 0.999))
    ema_b = LiveFloatEMA(model, decay=0.996)

    total_steps_b = args.phase_b_epochs * steps_per_ep
    warmup_b = 2 * steps_per_ep
    exact_start_ep = max(1, args.phase_b_epochs - 5)
    pure_eval_start_ep = max(1, args.phase_b_epochs - 2)

    best_b_acc = best_a_acc
    best_b_tag = f"phaseA_{best_a_tag}"
    best_b_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    top_tail_snapshots = []
    history_b = []
    step_b = 0
    t_start_b = time.time()

    for ep in range(1, args.phase_b_epochs + 1):
        t_ep = time.time()
        model.train()
        perm_imgs = torch.randperm(n_tr_imgs, device="cpu")
        tot_loss = 0.0
        tot_q = 0

        exact_scale = ep >= exact_start_ep
        p_flip = 0.0 if ep >= pure_eval_start_ep else (0.25 if exact_scale else 0.5)

        for b_start in range(0, n_tr_imgs, args.img_batch_size):
            img_batch = perm_imgs[b_start:b_start + args.img_batch_size]
            img_batch_dev = img_batch.to(dev)

            xb = augment_images_gpu(
                tr_imgs_gpu[img_batch_dev],
                can_flip_gpu[img_batch_dev],
                out_res=args.res,
                crop_min=args.crop_min,
                p_flip=p_flip,
                exact_scale=exact_scale,
            )

            q_idx, local_idx = gather_image_batch_questions(img_batch, tr_ptr, tr_flat)
            tq = tr_tokens[q_idx].to(dev)
            y_gt = tr_y_gt[q_idx].to(dev)
            y_kd = torch.sigmoid(tr_teacher_logits[q_idx].to(dev, dtype=torch.float32))

            for pg in opt_b.param_groups:
                base = pg["base_lr"]
                pg["lr"] = cosine_warmup_lr(step_b, total_steps_b, warmup_b, base, 1e-5)

            v_img = model.extract_visual(xb)
            logits = model.head(v_img[local_idx.to(dev)], tq)

            loss_gt = F.binary_cross_entropy_with_logits(logits, y_gt, reduction="none").sum(-1).mean()
            loss_kd = F.binary_cross_entropy_with_logits(logits, y_kd, reduction="none").sum(-1).mean()
            loss = (1.0 - args.kd_weight) * loss_gt + args.kd_weight * loss_kd

            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt_b.step()
            ema_b.update(model)

            tot_loss += float(loss.item()) * q_idx.numel()
            tot_q += q_idx.numel()
            step_b += 1

        live_acc, ema_acc, live_st, ema_st = eval_both_live_and_ema(
            model, ema_b, va_imgs_res_f16, va_tokens, va_q_img_idx, val_score_mat, dev
        )
        tag_ep, cur_st = ("ema", ema_st) if ema_acc >= live_acc else ("live", live_st)
        best_ep = max(live_acc, ema_acc)

        if best_ep > best_b_acc:
            best_b_acc = best_ep
            best_b_tag = f"ep{ep}_{tag_ep}"
            best_b_state = cur_st

        if ep >= args.phase_b_epochs - 7:
            top_tail_snapshots.append((best_ep, f"ep{ep}_{tag_ep}", cur_st))

        mode_str = "exact" if exact_scale else f"crop>={args.crop_min}"
        history_b.append({
            "epoch": ep,
            "mode": mode_str,
            "p_flip": p_flip,
            "loss": round(tot_loss / tot_q, 4),
            "live_val": round(live_acc, 2),
            "ema_val": round(ema_acc, 2),
        })
        print(
            f"[Phase B ep {ep:02d}/{args.phase_b_epochs} | {mode_str} flip={p_flip:.2f}] "
            f"loss={tot_loss / tot_q:.4f} | live={live_acc:.2f}% ema={ema_acc:.2f}% | "
            f"best={best_b_acc:.2f}% ({best_b_tag}) | {time.time() - t_ep:.1f}s",
            flush=True,
        )

    del tr_imgs_gpu, can_flip_gpu
    torch.cuda.empty_cache()
    phase_b_seconds = time.time() - t_start_b

    # Check if uniform top-k tail soup beats best single checkpoint
    final_state = best_b_state
    final_tag = best_b_tag
    final_fast_acc = best_b_acc

    if len(top_tail_snapshots) >= 2:
        top_sorted = sorted(top_tail_snapshots, key=lambda x: -x[0])
        for k in (2, 3):
            if len(top_sorted) >= k:
                chosen = top_sorted[:k]
                soup_st = {}
                keys = chosen[0][2].keys()
                for key in keys:
                    if chosen[0][2][key].is_floating_point():
                        soup_st[key] = sum(c[2][key].float() for c in chosen) / float(k)
                        if chosen[0][2][key].dtype != torch.float32:
                            soup_st[key] = soup_st[key].to(chosen[0][2][key].dtype)
                    else:
                        soup_st[key] = chosen[0][2][key].clone()
                model.load_state_dict(soup_st, strict=True)
                soup_acc = eval_val_fast(model, va_imgs_res_f16, va_tokens, va_q_img_idx, val_score_mat, dev)
                names = "+".join(c[1] for c in chosen)
                print(f"Tail Soup (top-{k}: {names}) -> {soup_acc:.2f}% (vs single {best_b_acc:.2f}%)", flush=True)
                if soup_acc > final_fast_acc + 1e-4:
                    final_fast_acc = soup_acc
                    final_tag = f"soup_top{k}({names})"
                    final_state = soup_st

    ckpt_path = CKPT_DIR / "v9_mbv3_ft.pt"
    torch.save(final_state, ckpt_path)
    print(f"\nSaved best checkpoint ({final_tag}, fast_val={final_fast_acc:.2f}%) -> {ckpt_path}", flush=True)

    # =========================================================================
    # Official Exact CPU Evaluation (Matches Codabench Pipeline 100%)
    # =========================================================================
    print("\n--- Running Exact Official Val Evaluation on CPU (num_workers=0) ---", flush=True)
    del va_imgs_res_f16
    torch.cuda.empty_cache()

    eval_model = load_model(MODEL_DIR, ckpt_path, device="cpu")
    preds = predict_split(eval_model, va_q, DATA / "val" / "images", batch_size=256, num_workers=0, device="cpu", log_every=0)
    official_val = vqa_evaluate(preds, va_a)
    official_val_clean = {k: v for k, v in official_val.items() if k != "_per_question"}
    acc = official_val["overall"]

    comp = measure_complexity(MODEL_DIR, ckpt_path, config_path=ROOT / "config.yaml")
    sc_w2 = load_score_cfg(ROOT / "config.yaml")
    res_w2 = composite(acc, comp["params"], comp["flops"], sc_w2)

    sc_w1 = dict(sc_w2, alpha=1.0, beta=1.0)
    res_w1 = composite(acc, comp["params"], comp["flops"], sc_w1)
    gap = round(res_w2["score"] - acc, 2)

    summary = {
        "model": "v9_mbv3_ft",
        "selected": final_tag,
        "arch": arch_cfg,
        "min_word_count": args.min_word_count,
        "vocab_size": len(words),
        "val": official_val_clean,
        "params": comp["params"],
        "flops": comp["flops"],
        "score_w2": res_w2,
        "score_w1": res_w1,
        "gap": gap,
        "audit_safe_under_20": gap <= 20.0,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": comp["checkpoint_sha256"],
        "phase_a_seconds": round(phase_a_seconds, 1),
        "phase_b_seconds": round(phase_b_seconds, 1),
        "history_a": history_a,
        "history_b": history_b,
    }
    log_path = LOGS_DIR / "v9_mbv3_ft_results.json"
    log_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n============================================================")
    print(f"OFFICIAL VAL OVERALL : {acc:.2f}%")
    print(f"BY TYPE              : {official_val['perAnswerType']}")
    print(f"PARAMS / FLOPS       : {comp['params']:,} / {comp['flops']:,}")
    print(f"SCORE (w=2, official): {res_w2['score']:.2f}  | Gap = {gap:.2f} (Safe <= 20: {gap <= 20.0})")
    print(f"SCORE (w=1, fallback): {res_w1['score']:.2f}")
    print(f"Results written to   : {log_path}")
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
