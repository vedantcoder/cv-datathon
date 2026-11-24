"""Model v3: Ultra-Lightweight ShuffleNetV2 x0.5 + 2D Coordinate Grid
+ Word-Dropout Regularized GRU + Gated Attention + Gated Tanh Multimodal Fusion (GLU)
+ 1000-way Answer Classifier.

Targeted for the #1 spot on Codabench:
~2.2M parameters (~+6.1 pt bonus), ~69 MFLOPs (~+11.5 pt bonus) = ~+17.5 pt formula efficiency bonus.
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
    def __init__(self, vocab_size, emb_dim=192, hidden=192, word_drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.word_drop = word_drop

    def forward(self, tokens, lengths):
        if self.training and self.word_drop > 0.0:
            mask = (torch.rand_like(tokens, dtype=torch.float) < self.word_drop) & (tokens != PAD)
            tokens = tokens.masked_fill(mask, UNK)
        x = self.emb(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]                                            # [B, hidden]


class FrozenShuffleNetCoordGrid(nn.Module):
    """ShuffleNetV2 x0.5 features up to stage4 + static 2D coordinates -> [B, 49, 194]. Frozen."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.shufflenet_v2_x0_5(weights=None)
        self.body = nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3, m.stage4)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        # Static 2D spatial coordinate grid: [1, 49, 2] in [-1, 1]
        y = torch.linspace(-1, 1, 7)
        x = torch.linspace(-1, 1, 7)
        coords = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).view(1, 49, 2)
        self.register_buffer("coords", coords, persistent=False)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, images):                                  # images in [0,1]
        x = (images - self.mean) / self.std
        f = self.body(x)                                        # [B, 192, 7, 7]
        grid = f.flatten(2).transpose(1, 2)                     # [B, 49, 192]
        c = self.coords.expand(grid.size(0), -1, -1)           # [B, 49, 2]
        return torch.cat([grid, c], dim=-1)                     # [B, 49, 194]


class GatedTanhDense(nn.Module):
    """Gated Tanh Unit: h = tanh(W_1 x + b_1) * sigmoid(W_2 x + b_2)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return torch.tanh(val) * torch.sigmoid(gate)


class V3GatedHead(nn.Module):
    def __init__(self, n_words, n_answers, v_dim=194, q_dim=192, att_dim=192, glimpses=2, hid=256, drop=0.3):
        super().__init__()
        self.q_enc = QuestionEncoder(n_words, emb_dim=192, hidden=q_dim, word_drop=0.1)
        self.att_v = nn.Linear(v_dim, att_dim)
        self.att_q = nn.Linear(q_dim, att_dim)
        self.att_out = nn.Linear(att_dim, glimpses)

        # Gated Tanh Projections for robust multimodal fusion
        self.v_gated = GatedTanhDense(v_dim * glimpses, hid)
        self.q_gated = GatedTanhDense(q_dim, hid)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hid),
            nn.Dropout(drop),
            nn.Linear(hid, 384),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(384, n_answers),
        )

    def forward(self, v, tokens, lengths):
        # v: [B, 49, 194]
        v_norm = F.normalize(v, dim=-1)
        q = self.q_enc(tokens, lengths)                         # [B, 192]

        # 2-glimpse question-guided spatial attention
        a = self.att_out(torch.tanh(self.att_v(v_norm) + self.att_q(q).unsqueeze(1)))  # [B, 49, G]
        a = a.softmax(dim=1)
        v_att = torch.einsum("bng,bnd->bgd", a, v_norm).flatten(1)  # [B, G*194]

        # Gated multimodal fusion
        h_v = self.v_gated(v_att)                               # [B, hid]
        h_q = self.q_gated(q)                                   # [B, hid]
        fused = h_v * h_q                                       # [B, hid]

        return self.classifier(fused)                           # [B, n_answers]


class V3ShuffleNetVQA(nn.Module):
    def __init__(self, words, answers):
        super().__init__()
        self.words, self.answers = words, answers
        self.w2i = {w: i for i, w in enumerate(words)}
        self.backbone = FrozenShuffleNetCoordGrid()
        self.head = V3GatedHead(len(words), len(answers))

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
    return V3ShuffleNetVQA(_read_lines("words.txt"), _read_lines("answers.txt"))
