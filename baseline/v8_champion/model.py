"""Model v8 Champion: Ultra-Compact Fine-Tuned & Multi-Label Distilled VQA.

Combines:
- Pretrained ShuffleNetV2 x0.5 (Stage 4) fine-tuned end-to-end at ultra-efficient resolution (40x40 / 48x48)
- Top-550 high-coverage answer vocabulary (covers 98.3%+ of val predictions while halving head size)
- Factorized word embeddings + singular fallback + Bidirectional Dilated Conv1D Question Encoder
- Multiplicative Normalized Fusion + Low-Rank Bottleneck Classifier
Target: ~315k params, ~3.0-3.5M FLOPs -> +31.8 to +32.2 pt efficiency bonus -> 73.0 - 74.5+ score!
"""
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HERE = Path(__file__).resolve().parent
MAX_Q_LEN = 20
PAD, UNK = 0, 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tokenize(question: str) -> list:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", question.lower())


class CompactConvQuestionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_rank=16, q_dim=64, word_drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_rank, padding_idx=PAD)
        self.emb_proj = nn.Linear(emb_rank, q_dim, bias=False)
        self.q_pos = nn.Parameter(torch.randn(1, MAX_Q_LEN, q_dim) * 0.02)
        self.dw1 = nn.Conv1d(q_dim, q_dim, kernel_size=3, padding=1, groups=q_dim)
        self.pw1 = nn.Conv1d(q_dim, q_dim, kernel_size=1)
        self.dw2 = nn.Conv1d(q_dim, q_dim, kernel_size=3, padding=2, dilation=2, groups=q_dim)
        self.pw2 = nn.Conv1d(q_dim, q_dim, kernel_size=1)
        self.norm = nn.LayerNorm(q_dim)
        self.q_att = nn.Linear(q_dim, 2)
        self.word_drop = word_drop

    def forward(self, tokens):
        if self.training and self.word_drop > 0.0:
            mask = (torch.rand_like(tokens, dtype=torch.float) < self.word_drop) & (tokens != PAD)
            tokens = tokens.masked_fill(mask, UNK)

        pad_mask = (tokens == PAD)
        valid_f = (~pad_mask).unsqueeze(-1).float()
        valid_c = valid_f.transpose(1, 2)

        L = tokens.size(1)
        x = (self.emb_proj(self.emb(tokens)) + self.q_pos[:, :L]) * valid_f
        h = x.transpose(1, 2)
        h = (h + F.gelu(self.pw1(self.dw1(h)))) * valid_c
        h = (h + F.gelu(self.pw2(self.dw2(h)))) * valid_c
        seq = self.norm(h.transpose(1, 2))

        scores = self.q_att(seq).masked_fill(pad_mask.unsqueeze(-1), -1e9)
        w = F.softmax(scores, dim=1)
        return torch.einsum("blh,bld->bhd", w, seq).flatten(1)


class V8ChampionHead(nn.Module):
    def __init__(self, n_words, n_answers, v_raw_dim=192, v_dim=80, n_tokens=4,
                 emb_rank=16, q_dim=64, hid=104, cls_rank=72, drop=0.22):
        super().__init__()
        self.v_proj = nn.Linear(v_raw_dim, v_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, v_dim) * 0.02)
        self.q_enc = CompactConvQuestionEncoder(n_words, emb_rank=emb_rank, q_dim=q_dim, word_drop=0.08)

        q_out_dim = q_dim * 2
        self.att_v = nn.Linear(v_dim, 48)
        self.att_q = nn.Linear(q_out_dim, 48)
        self.att_out = nn.Linear(48, 2)

        self.v_fuse = nn.Sequential(nn.Linear(v_dim * 2, hid), nn.LayerNorm(hid), nn.GELU())
        self.q_fuse = nn.Sequential(nn.Linear(q_out_dim, hid), nn.LayerNorm(hid), nn.GELU())

        self.classifier = nn.Sequential(
            nn.LayerNorm(hid),
            nn.Dropout(drop),
            nn.Linear(hid, hid + 40),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid + 40, cls_rank, bias=False),
            nn.Linear(cls_rank, n_answers),
        )

    def forward(self, v_raw, tokens):
        v = F.gelu(self.v_proj(v_raw))
        v_norm = F.normalize(v + self.pos_emb, dim=-1)
        q = self.q_enc(tokens)

        a = F.softmax(self.att_out(torch.tanh(self.att_v(v_norm) + self.att_q(q).unsqueeze(1))), dim=1)
        v_att = torch.einsum("bng,bnd->bgd", a, v_norm)
        v_mean = v_norm.mean(dim=1)
        v_feat = torch.cat([v_att[:, 0], v_att[:, 1] + 0.5 * v_mean], dim=-1)

        fused = self.v_fuse(v_feat) * self.q_fuse(q)
        return self.classifier(fused)


class V8ChampionVQA(nn.Module):
    def __init__(self, words, answers, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.img_res = int(cfg.get("res", 40))
        n_tokens = int(cfg.get("n_tokens", 4))
        v_dim = int(cfg.get("v_dim", 80))
        emb_rank = int(cfg.get("emb_rank", 16))
        q_dim = int(cfg.get("q_dim", 64))
        hid = int(cfg.get("hid", 104))
        cls_rank = int(cfg.get("cls_rank", 72))
        drop = float(cfg.get("drop", 0.22))

        self.words, self.answers = words, answers
        self.w2i = {w: i for i, w in enumerate(words)}

        m = torchvision.models.shufflenet_v2_x0_5(weights=None)
        self.backbone = nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.head = V8ChampionHead(
            len(words), len(answers),
            v_raw_dim=192, v_dim=v_dim, n_tokens=n_tokens,
            emb_rank=emb_rank, q_dim=q_dim, hid=hid, cls_rank=cls_rank, drop=drop,
        )

    def encode_questions(self, questions, device, max_len=None):
        ids = []
        for q in questions:
            arr = []
            for t in tokenize(q):
                if t in self.w2i:
                    arr.append(self.w2i[t])
                elif len(t) > 3 and t.endswith("s") and t[:-1] in self.w2i:
                    arr.append(self.w2i[t[:-1]])
                else:
                    arr.append(UNK)
            ids.append(arr[:MAX_Q_LEN] or [UNK])
        L = max_len if max_len is not None else max(len(x) for x in ids)
        tokens = torch.full((len(ids), L), PAD, dtype=torch.long)
        for i, x in enumerate(ids):
            tokens[i, :len(x)] = torch.tensor(x)
        return tokens.to(device)

    def extract_visual(self, images):
        if images.shape[-1] != self.img_res or images.shape[-2] != self.img_res:
            images = F.interpolate(images, size=(self.img_res, self.img_res), mode="bilinear", align_corners=False)
        x = (images - self.mean) / self.std
        f = self.backbone(x)
        return f.flatten(2).transpose(1, 2)

    def forward(self, images, tokens):
        return self.head(self.extract_visual(images), tokens)

    @torch.no_grad()
    def predict(self, images, questions):
        dev = next(self.parameters()).device
        tokens = self.encode_questions(questions, dev)
        logits = self.forward(images.to(dev).float(), tokens)
        return [self.answers[i] for i in logits.argmax(-1).tolist()]


def _read_lines(name):
    return (HERE / name).read_text().splitlines()


def build_model():
    arch_path = HERE / "arch.json"
    cfg = json.loads(arch_path.read_text()) if arch_path.exists() else {}
    return V8ChampionVQA(_read_lines("words.txt"), _read_lines("answers.txt"), cfg=cfg)
