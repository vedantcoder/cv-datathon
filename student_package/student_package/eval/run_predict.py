"""Run a submission through the fixed interface and write predictions.json.

    python eval/run_predict.py --model-dir <dir with model.py> --ckpt <ckpt.pt> \
        --images <split>/images --questions <split>/questions.json --out predictions.json

Image convention (fixed for everyone): each image is loaded with PIL, converted to RGB, and turned into a
FloatTensor[3,224,224] with values in [0, 1] (torchvision to_tensor; no mean/std normalization — models
normalize internally). Batches are FloatTensor[B,3,224,224]. Questions are passed as raw strings.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from eval.complexity import import_student_module  # noqa: E402


def load_image(path) -> torch.Tensor:
    with Image.open(path) as im:
        t = to_tensor(im.convert("RGB"))
    if t.shape != (3, 224, 224):
        raise ValueError(f"{path}: expected 3x224x224, got {tuple(t.shape)}")
    return t


class _QDataset(torch.utils.data.Dataset):
    def __init__(self, questions, image_dir):
        self.q, self.dir = questions, Path(image_dir)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        q = self.q[i]
        return load_image(self.dir / f"{q['image_id']}.jpg"), q["question"], q["question_id"]


def _collate(batch):
    imgs, qs, ids = zip(*batch)
    return torch.stack(imgs), list(qs), list(ids)


def load_model(model_dir, ckpt, device):
    mod = import_student_module(Path(model_dir).resolve())
    model = mod.build_model()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def predict_split(model, questions, image_dir, batch_size=64, num_workers=4, device="cpu", log_every=50):
    dl = torch.utils.data.DataLoader(_QDataset(questions, image_dir), batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, collate_fn=_collate)
    preds, t0 = [], time.time()
    with torch.no_grad():
        for b, (imgs, qs, ids) in enumerate(dl):
            answers = model.predict(imgs.to(device), qs)
            if len(answers) != len(qs):
                raise RuntimeError(f"predict returned {len(answers)} answers for {len(qs)} questions")
            preds += [{"question_id": int(i), "answer": str(a)} for i, a in zip(ids, answers)]
            if log_every and b % log_every == 0:
                print(f"  batch {b}/{len(dl)}  {time.time() - t0:.0f}s", flush=True)
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    questions = json.load(open(args.questions))
    model = load_model(args.model_dir, args.ckpt, args.device)
    t0 = time.time()
    preds = predict_split(model, questions, args.images, args.batch_size, args.num_workers, args.device)
    assert len({p["question_id"] for p in preds}) == len(questions)
    Path(args.out).write_text(json.dumps(preds))
    print(f"wrote {len(preds)} predictions to {args.out} in {time.time() - t0:.1f}s on {args.device}")


if __name__ == "__main__":
    main()
