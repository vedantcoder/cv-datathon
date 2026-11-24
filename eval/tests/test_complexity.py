import json
import math
import textwrap

import pytest
import torch
import yaml

from eval.complexity import measure
from eval.score import composite, load_score_cfg, validate_complexity

TINY_MODEL = textwrap.dedent('''
    import torch, torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 20)      # frozen: must still be counted
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.head = nn.Linear(20, 5)
            self.answers = ["yes", "no", "1", "2", "red"]

        def predict(self, images, questions):
            x = images.flatten(1)[:, :10]           # slicing: 0 FLOPs
            logits = self.head(torch.relu(self.backbone(x)))
            return [self.answers[i] for i in logits.argmax(1).tolist()]

    def build_model():
        return Tiny()
''')


@pytest.fixture
def tiny(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "model.py").write_text(TINY_MODEL)
    import importlib.util
    spec = importlib.util.spec_from_file_location("tiny_model_for_ckpt", d / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    torch.manual_seed(123)
    ckpt = d / "ckpt.pt"
    torch.save(m.build_model().state_dict(), ckpt)
    return d, ckpt


def test_tiny_params_and_flops(tiny):
    d, ckpt = tiny
    r = measure(d, ckpt)
    assert r["params"] == (10 * 20 + 20) + (20 * 5 + 5)        # 325, frozen backbone included
    assert r["flops"] == 2 * 1 * 10 * 20 + 2 * 1 * 20 * 5        # 600: multiply-add = 2 FLOPs, batch 1
    assert r["torch_version"] == torch.__version__
    assert len(r["checkpoint_sha256"]) == 64
    assert set(r) == {"params", "flops", "checkpoint_sha256", "torch_version"}


def test_deterministic(tiny):
    d, ckpt = tiny
    assert measure(d, ckpt) == measure(d, ckpt)


def test_bad_predict_output(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "model.py").write_text(TINY_MODEL.replace("return [self.answers", "return 0 and [self.answers"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("bad_model", d / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    torch.save(m.build_model().state_dict(), d / "c.pt")
    with pytest.raises(TypeError, match="list\\[str\\]"):
        measure(d, d / "c.pt")


def test_composite_formula():
    sc = {"alpha": 2.0, "beta": 1.0, "P_ref": 1000, "F_ref": 500}
    r = composite(50.0, 4000, 250, sc)
    assert r["score"] == pytest.approx(50.0 - 2.0 * math.log2(4) - 1.0 * math.log2(0.5))   # 50 - 4 + 1 = 47


def test_score_cfg_todo_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"score": {"alpha": None, "beta": 1.0, "P_ref": 1, "F_ref": 1}}))
    with pytest.raises(ValueError, match="not set"):
        load_score_cfg(p)


def test_validate_complexity():
    good = {"params": 1, "flops": 2, "checkpoint_sha256": "ab", "torch_version": "x"}
    assert validate_complexity(good) == good
    with pytest.raises(ValueError):
        validate_complexity({**good, "params": "1"})
    with pytest.raises(ValueError):
        validate_complexity({k: v for k, v in good.items() if k != "flops"})
