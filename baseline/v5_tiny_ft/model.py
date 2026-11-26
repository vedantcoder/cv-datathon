"""Model v5: Ultra-Compact End-to-End Fine-Tuned & Distilled VQA Architecture.

Highlights:
- Pretrained ShuffleNetV2 x0.5 (stage 4) fine-tuned end-to-end at compact spatial resolution
- Learned 2D spatial positional embeddings
- Factorized word embeddings + singular fallback for high vocabulary coverage at ~130k params
- Bidirectional Depthwise-Separable Dilated Conv1D + Dual-View Attention Question Encoder
- 2-Glimpse Question-Guided Spatial Attention + Global Mean Residual + Gated Tanh Fusion (GLU)
- Low-Rank Factorized 1000-way Answer Classifier
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
    def __init__(self, vocab_size, emb_rank=32, q_dim=96, word_drop=0.0):
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
        valid_f = (~pad_mask).unsqueeze(-1).float()             # [B, L, 1]
        valid_c = valid_f.transpose(1, 2)                       # [B, 1, L]

        L = tokens.size(1)
        x = (self.emb_proj(self.emb(tokens)) + self.q_pos[:, :L]) * valid_f
        h = x.transpose(1, 2)                                   # [B, D, L]
        h = (h + F.gelu(self.pw1(self.dw1(h)))) * valid_c
        h = (h + F.gelu(self.pw2(self.dw2(h)))) * valid_c
        seq = self.norm(h.transpose(1, 2))                      # [B, L, D]

        scores = self.q_att(seq).masked_fill(pad_mask.unsqueeze(-1), -1e9)  # [B, L, 2]
        w = F.softmax(scores, dim=1)
        return torch.einsum("blh,bld->bhd", w, seq).flatten(1)  # [B, 2*D]


class GatedDense(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return torch.tanh(val) * torch.sigmoid(gate)


class V5TinyHead(nn.Module):
    def __init__(self, n_words, n_answers, v_dim=192, n_tokens=9, emb_rank=32,
                 q_dim=96, hid=144, cls_rank=128, drop=0.25):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, v_dim) * 0.02)
        self.q_enc = CompactConvQuestionEncoder(n_words, emb_rank=emb_rank, q_dim=q_dim, word_drop=0.08)

        q_out_dim = q_dim * 2
        self.att_v = nn.Linear(v_dim, 64)
        self.att_q = nn.Linear(q_out_dim, 64)
        self.att_out = nn.Linear(64, 2)

        self.v_gated = GatedDense(v_dim * 2, hid)
        self.q_gated = GatedDense(q_out_dim, hid)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hid),
            nn.Dropout(drop),
            nn.Linear(hid, hid + 80),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid + 80, cls_rank, bias=False),
            nn.Linear(cls_rank, n_answers),
        )

    def forward(self, v, tokens):
        # v: [B, N, 192]
        v_norm = F.normalize(v + self.pos_emb, dim=-1)
        q = self.q_enc(tokens)                                  # [B, 2*q_dim]

        a = F.softmax(self.att_out(torch.tanh(self.att_v(v_norm) + self.att_q(q).unsqueeze(1))), dim=1)
        v_att = torch.einsum("bng,bnd->bgd", a, v_norm)         # [B, 2, 192]
        v_mean = v_norm.mean(dim=1)                             # [B, 192]
        v_feat = torch.cat([v_att[:, 0], v_att[:, 1] + 0.5 * v_mean], dim=-1)

        fused = self.v_gated(v_feat) * self.q_gated(q)          # [B, hid]
        return self.classifier(fused)                           # [B, n_answers]


class V5TinyVQA(nn.Module):
    def __init__(self, words, answers, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.img_res = int(cfg.get("res", 80))
        n_tokens = int(cfg.get("n_tokens", 9))
        emb_rank = int(cfg.get("emb_rank", 32))
        q_dim = int(cfg.get("q_dim", 96))
        hid = int(cfg.get("hid", 144))
        cls_rank = int(cfg.get("cls_rank", 128))
        drop = float(cfg.get("drop", 0.25))

        self.words, self.answers = words, answers
        self.w2i = {w: i for i, w in enumerate(words)}

        m = torchvision.models.shufflenet_v2_x0_5(weights=None)
        self.backbone = nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.head = V5TinyHead(
            len(words), len(answers),
            v_dim=192, n_tokens=n_tokens, emb_rank=emb_rank,
            q_dim=q_dim, hid=hid, cls_rank=cls_rank, drop=drop,
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
        f = self.backbone(x)                                    # [B, 192, H', W']
        return f.flatten(2).transpose(1, 2)                     # [B, N, 192]

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
    return V5TinyVQA(_read_lines("words.txt"), _read_lines("answers.txt"), cfg=cfg)
