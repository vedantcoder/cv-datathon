"""MobileNetV3-Small VQA Model: frozen MobileNetV3-Small grid features + GRU question encoder
+ 2-glimpse spatial attention + gated fusion + 1000-way answer classifier.

Designed for maximum composite score efficiency on Codabench:
~4.3M parameters, ~135 MFLOPs (vs 18.15M params and 3.69 GFLOPs for the ResNet-18 baseline).
Yields a ~+13.7 point composite efficiency bonus.
"""
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


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)

    def forward(self, tokens, lengths):
        x = self.emb(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]                                            # [B, hidden]


class FrozenMobileNetV3SmallGrid(nn.Module):
    """MobileNetV3-Small features -> [B, 49, 576] grid features for 224x224 input. Frozen."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.mobilenet_v3_small(weights=None)
        self.features = m.features
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):                                 # stay in eval mode (frozen BN stats)
        return super().train(False)

    def forward(self, images):                                  # images in [0,1]
        x = (images - self.mean) / self.std
        f = self.features(x)                                    # [B, 576, 7, 7]
        return f.flatten(2).transpose(1, 2)                     # [B, 49, 576]


class VQAHead(nn.Module):
    def __init__(self, n_words, n_answers, v_dim=576, q_dim=256, att_dim=256, glimpses=2, hid=512, drop=0.3):
        super().__init__()
        self.q_enc = QuestionEncoder(n_words, emb_dim=256, hidden=q_dim)
        self.att_v = nn.Linear(v_dim, att_dim)
        self.att_q = nn.Linear(q_dim, att_dim)
        self.att_out = nn.Linear(att_dim, glimpses)
        self.v_proj = nn.Linear(v_dim * glimpses, hid)
        self.q_proj = nn.Linear(q_dim, hid)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hid),
            nn.Dropout(drop),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, n_answers),
        )

    def forward(self, v, tokens, lengths):
        v = F.normalize(v, dim=-1)                              # [B, 49, 576]
        q = self.q_enc(tokens, lengths)                         # [B, 256]
        a = self.att_out(torch.tanh(self.att_v(v) + self.att_q(q).unsqueeze(1)))   # [B, 49, G]
        a = a.softmax(dim=1)
        v_att = torch.einsum("bng,bnd->bgd", a, v).flatten(1)  # [B, G*576]
        fused = F.gelu(self.v_proj(v_att)) * F.gelu(self.q_proj(q))
        return self.classifier(fused)                           # [B, n_answers] logits


class MobileNetVQA(nn.Module):
    def __init__(self, words, answers):
        super().__init__()
        self.words, self.answers = words, answers
        self.w2i = {w: i for i, w in enumerate(words)}
        self.backbone = FrozenMobileNetV3SmallGrid()
        self.head = VQAHead(len(words), len(answers))

    def encode_questions(self, questions, device):
        ids = [[self.w2i.get(t, UNK) for t in tokenize(q)][:MAX_Q_LEN] or [UNK] for q in questions]
        lengths = torch.tensor([len(x) for x in ids])
        tokens = torch.full((len(ids), MAX_Q_LEN), PAD, dtype=torch.long)
        for i, x in enumerate(ids):
            tokens[i, : len(x)] = torch.tensor(x)
        return tokens.to(device), lengths

    def forward(self, images, tokens, lengths):
        return self.head(self.backbone(images), tokens, lengths)

    @torch.no_grad()
    def predict(self, images, questions):
        dev = next(self.parameters()).device
        tokens, lengths = self.encode_questions(questions, dev)
        logits = self.forward(images.to(dev).float(), tokens, lengths)
        return [self.answers[i] for i in logits.argmax(-1).tolist()]


def _read_lines(name):
    return (HERE / name).read_text().splitlines()


def build_model():
    return MobileNetVQA(_read_lines("words.txt"), _read_lines("answers.txt"))
