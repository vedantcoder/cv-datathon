"""Complexity counter for a student submission.

    python eval/complexity.py --model-dir <dir with model.py> --ckpt <checkpoint.pt> [--out complexity.json]

Student interface (fixed):
    model = build_model()                         # defined in <model-dir>/model.py
    model.load_state_dict(torch.load(ckpt_path))
    answers = model.predict(images, questions)    # FloatTensor[B,3,224,224], list[str] -> list[str]

Params : sum of p.numel() over ALL parameters (frozen/pretrained included; shared tensors counted once).
FLOPs  : torch.utils.flop_counter.FlopCounterMode around ONE predict() call, batch size 1, with the fixed
         dummy image + dummy question from config.yaml, on CPU, under torch.no_grad().
         Convention: one multiply-add = 2 FLOPs. FlopCounterMode counts matmul / conv / attention-type ops;
         elementwise ops (activations, norms, additions) are not counted.
Output : {"params": int, "flops": int, "checkpoint_sha256": str, "torch_version": str}
FLOP counts depend on the torch version, so the version is recorded.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
BUFFER_WARN_FRACTION = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_student_module(model_dir: Path):
    model_py = model_dir / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"{model_py} not found")
    sys.path.insert(0, str(model_dir))  # allow model.py to import its own helper modules
    spec = importlib.util.spec_from_file_location("student_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_model"):
        raise AttributeError(f"{model_py} must define build_model()")
    return mod


def dummy_inputs(cfg: dict):
    c = cfg["complexity"]
    img = torch.full(c["dummy_image"]["shape"], float(c["dummy_image"]["fill"]), dtype=torch.float32)
    assert img.shape[0] == c["batch_size"] == 1
    return img, [c["dummy_question"]]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model, images, questions) -> int:
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        out = model.predict(images, questions)
    if not (isinstance(out, list) and len(out) == len(questions) and all(isinstance(a, str) for a in out)):
        raise TypeError(f"predict() must return list[str] of length {len(questions)}, got {out!r}")
    return int(fc.get_total_flops())


def measure(model_dir, ckpt, config_path=ROOT / "config.yaml") -> dict:
    model_dir, ckpt = Path(model_dir).resolve(), Path(ckpt).resolve()
    cfg = yaml.safe_load(open(config_path))
    torch.manual_seed(0)
    mod = import_student_module(model_dir)
    model = mod.build_model()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to("cpu").eval()
    if not callable(getattr(model, "predict", None)):
        raise AttributeError("model returned by build_model() must have a predict(images, questions) method")

    params = count_params(model)
    n_buf = sum(b.numel() for b in model.buffers())
    if params and n_buf > BUFFER_WARN_FRACTION * params:
        print(f"WARNING: model has {n_buf:,} buffer elements ({n_buf / params:.1%} of params). Buffers are not "
              f"counted as params; check that weights are not being hidden in buffers.", file=sys.stderr)

    images, questions = dummy_inputs(cfg)
    flops = count_flops(model, images, questions)
    return {"params": int(params), "flops": flops, "checkpoint_sha256": sha256_file(ckpt),
            "torch_version": torch.__version__}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count params / FLOPs of a student VQA model")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default="complexity.json")
    args = ap.parse_args(argv)
    res = measure(args.model_dir, args.ckpt, args.config)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
